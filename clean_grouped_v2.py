"""Pre-training dataset construction; never writes source images or assigns labels.

Requires numpy, pandas and Pillow (already required by CNN_resnet18.ipynb).
Candidate components are isolation units, NOT assertions of duplicate identity.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageOps, ImageDraw

CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
POLICY = dict(protocol='clean_grouped_v2', schema=1, seed=42,
              ratios=[0.70, 0.15, 0.15], candidate_hamming=4,
              candidate_gray_mae=3.0, reliable_hamming=1,
              reliable_rgb_mae=1.0, reliable_rgb_p99=4.0,
              conflict_policy='quarantine_entire_candidate_component',
              uncertain_policy='retain_but_isolate_entire_candidate_component',
              exact_policy='retain_lexicographically_first_path',
              resize='bilinear', comparison_size=48)
BLOCKS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Components:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def features(root, row):
    with Image.open(root / row.path) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB')
        actual = hashlib.sha256(str(rgb.size).encode() + b'|' + rgb.tobytes()).hexdigest()
        if actual != row.pixel_sha256:
            raise ValueError(f'Source changed since scan: {row.path}')
        gray = rgb.convert('L')
        hp = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
        h = int.from_bytes(np.packbits((hp[:, 1:] > hp[:, :-1]).ravel()).tobytes(), 'big')
        return h, np.asarray(gray.resize((48, 48), Image.Resampling.BILINEAR)), np.asarray(
            rgb.resize((48, 48), Image.Resampling.BILINEAR)), rgb.size


def assign_splits(frame, seed=42):
    """Deterministic per-class greedy sample balancing with indivisible groups.

    Approximate stratification: large groups can prevent exact target ratios.
    Require >=3 groups/class, reserve one group per split; fail instead of falling
    back to image-level splitting when the constraints are infeasible.
    """
    assignments = {}
    for label in CLASSES:
        sizes = frame[frame.label == label].groupby('group_id').size().to_dict()
        if len(sizes) < 3:
            raise ValueError(f'{label}: need at least three eligible isolation groups')
        groups = sorted(sizes, key=lambda g: (-sizes[g], digest([seed, g])))
        target = np.asarray(POLICY['ratios']) * sum(sizes.values())
        used = np.zeros(3)
        for pos, group in enumerate(groups):
            empty = np.flatnonzero(used == 0)
            choices = empty if len(groups) - pos == len(empty) else range(3)
            split = min(choices, key=lambda k: (float(np.sum(
                ((used + np.eye(3)[k] * sizes[group] - target) ** 2) / np.maximum(target, 1))), k))
            used[split] += sizes[group]
            assignments[group] = ['train', 'validation', 'test'][split]
    result = frame.copy()
    result['split'] = result.group_id.map(assignments)
    verify_splits(result)
    return result


def verify_splits(frame):
    if not frame.path.is_unique or frame.split.isna().any():
        raise ValueError('Missing or repeated split membership')
    for key in ['group_id', 'reliable_group_id', 'pixel_sha256']:
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError(f'{key} crosses split boundaries')
    for split in ['train', 'validation', 'test']:
        if set(frame.loc[frame.split == split, 'label']) != set(CLASSES):
            raise ValueError(f'{split} does not contain all five classes')
    if not (frame.target == frame.label.map(dict(zip(CLASSES, range(5))))).all():
        raise ValueError('Incorrect class mapping')


def prepare(manifest, root, output, seed=42, archive_sha256=None):
    """Build an immutable proposal in a NEW directory; caller reviews hashes.

    Input is the complete readable-image manifest, before baseline exclusions.
    Exact hashes supplied by the notebook are verified by decoding every image.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Output must be outside source dataset')
    output.mkdir(parents=True, exist_ok=False)
    frame = manifest.sort_values('path').reset_index(drop=True).copy()
    if frame.empty or not frame.path.is_unique or not set(frame.label) <= set(CLASSES):
        raise ValueError('Invalid input manifest')
    for path in frame.path:
        if not (root / path).resolve().is_relative_to(root):
            raise ValueError('Source path escapes dataset')
    cfg = dict(POLICY, seed=seed)
    cfg['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg['versions'] = dict(python=platform.python_version(), numpy=np.__version__,
                           pandas=pd.__version__, pillow=PIL.__version__)
    hashes, grays, rgbs, sizes = [], [], [], []
    for row in frame.itertuples():
        h, gray, rgb, size = features(root, row)
        hashes.append(h); grays.append(gray); rgbs.append(rgb); sizes.append(size)
    reliable, isolation = Components(len(frame)), Components(len(frame))
    exact = defaultdict(list)
    for i, h in enumerate(frame.pixel_sha256):
        exact[h].append(i)
    representatives = []
    for members in exact.values():
        representatives.append(members[0])
        for j in members[1:]:
            reliable.union(members[0], j); isolation.union(members[0], j)
    buckets, pairs = defaultdict(list), []
    for i in sorted(representatives):
        candidates = set()
        for b, (shift, width) in enumerate(BLOCKS):
            candidates.update(buckets[b, (hashes[i] >> shift) & ((1 << width) - 1)])
        for j in sorted(candidates):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance > cfg['candidate_hamming']:
                continue
            mae = float(np.abs(grays[i].astype(np.int16) - grays[j]).mean())
            if mae > cfg['candidate_gray_mae']:
                continue
            difference = np.abs(rgbs[i].astype(np.int16) - rgbs[j])
            rgb_mae, p99 = float(difference.mean()), float(np.percentile(difference, 99))
            high = (sizes[i] == sizes[j] and distance <= cfg['reliable_hamming'] and
                    rgb_mae <= cfg['reliable_rgb_mae'] and p99 <= cfg['reliable_rgb_p99'] and
                    min(float(grays[i].std()), float(grays[j].std())) >= 10)
            isolation.union(i, j)
            if high:
                reliable.union(i, j)
            pairs.append(dict(path_a=frame.at[j, 'path'], path_b=frame.at[i, 'path'],
                              label_a=frame.at[j, 'label'], label_b=frame.at[i, 'label'],
                              hamming=distance, gray_mae=mae, rgb_mae=rgb_mae, rgb_p99=p99,
                              high_confidence=high, label_conflict=frame.at[i, 'label'] != frame.at[j, 'label']))
        for b, (shift, width) in enumerate(BLOCKS):
            buckets[b, (hashes[i] >> shift) & ((1 << width) - 1)].append(i)
    def ids(components, prefix):
        members = defaultdict(list)
        for i in range(len(frame)):
            members[components.find(i)].append(frame.at[i, 'path'])
        names = {k: prefix + digest(v) for k, v in members.items()}
        return [names[components.find(i)] for i in range(len(frame))]
    frame['reliable_group_id'] = ids(reliable, 'dup_')
    frame['group_id'] = ids(isolation, 'isolate_')
    frame['target'] = frame.label.map(dict(zip(CLASSES, range(5))))
    conflict = frame.groupby('group_id').label.transform('nunique') > 1
    frame['decision'] = np.where(conflict, 'quarantine_label_conflict', 'eligible')
    redundant = frame.duplicated('pixel_sha256') & ~conflict
    frame.loc[redundant, 'decision'] = 'redundant_exact_copy'
    eligible = frame[frame.decision == 'eligible'].copy()
    # Report affected samples BEFORE constructing splits or allowing training.
    distribution = pd.DataFrame(index=CLASSES)
    for name, subset in [('original_readable', frame), ('proposed_retained', eligible),
                         ('quarantine_label_conflict', frame[conflict]),
                         ('redundant_exact_copy', frame[redundant])]:
        distribution[name] = subset.label.value_counts().reindex(CLASSES, fill_value=0)
    distribution['quarantine_fraction'] = distribution.quarantine_label_conflict / distribution.original_readable
    distribution.to_csv(output / 'class_distribution.csv', index_label='label')
    columns = ['path_a', 'path_b', 'label_a', 'label_b', 'hamming', 'gray_mae', 'rgb_mae', 'rgb_p99', 'high_confidence', 'label_conflict']
    pair_df = pd.DataFrame(pairs, columns=columns)
    pair_df.to_csv(output / 'candidate_pairs.csv', index=False)
    pair_df.loc[pair_df.label_conflict].to_csv(output / 'cross_label_candidates.csv', index=False)
    frame.loc[conflict].to_csv(output / 'conflicting_groups_manifest.csv', index=False)
    groups = frame.groupby('group_id').agg(images=('path', 'size'), labels=('label', lambda x: '|'.join(sorted(set(x)))), reliable_groups=('reliable_group_id', 'nunique'))
    groups['manual_review'] = (groups.images > 1)
    groups.to_csv(output / 'groups.csv')
    frame.to_csv(output / 'decision_manifest.csv', index=False)
    # Paginated contact sheets cover ALL multi-image components, including exact conflicts.
    review = frame[frame.group_id.isin(groups.index[groups.manual_review])].sort_values(['group_id', 'path'])
    sheets = output / 'contact_sheets'; sheets.mkdir()
    for start in range(0, len(review), 36):
        page = review.iloc[start:start + 36]
        canvas = Image.new('RGB', (960, 6 * 150), 'white'); draw = ImageDraw.Draw(canvas)
        for k, row in enumerate(page.itertuples()):
            x, y = (k % 6) * 160, (k // 6) * 150
            with Image.open(root / row.path) as src:
                thumbnail = ImageOps.exif_transpose(src).convert('RGB'); thumbnail.thumbnail((150, 105))
                canvas.paste(thumbnail, (x, y))
            draw.text((x, y + 106), f'{start+k}: {row.label}\n{row.group_id[8:20]}', fill='black')
        canvas.save(sheets / f'{start // 36:04d}.png')
    review.reset_index(drop=True).to_csv(output / 'contact_sheet_index.csv', index_label='image_index')
    split = assign_splits(eligible, seed)
    integrity = dict(
        passed=True, unique_paths=bool(split.path.is_unique),
        eligible_images=len(eligible), assigned_images=len(split),
        excluded_images=len(frame) - len(split),
        crossing_groups={key: int((split.groupby(key).split.nunique() > 1).sum())
                         for key in ['group_id', 'reliable_group_id', 'pixel_sha256']},
        all_five_classes_per_split=True, class_mapping=dict(zip(CLASSES, range(5))),
        final_test_evaluation_enabled=False)
    (output / 'split_integrity.json').write_text(json.dumps(integrity, indent=2, sort_keys=True))
    split.to_csv(output / 'split_manifest.csv', index=False)
    for name in ['train', 'validation', 'test']:
        split[split.split == name].to_csv(output / f'{name}_manifest.csv', index=False)
    table = pd.crosstab(split.label, split.split).reindex(CLASSES)[['train', 'validation', 'test']]
    table.to_csv(output / 'split_distribution.csv')
    summary = dict(protocol=cfg['protocol'], archive_sha256=archive_sha256,
                   config_sha256=digest(cfg), dataset_sha256=digest(frame[['path', 'label', 'pixel_sha256']].to_dict('records')),
                   split_sha256=hashlib.sha256((output / 'split_manifest.csv').read_bytes()).hexdigest(),
                   readable_images=len(frame), retained_images=len(split), candidate_pairs=len(pairs),
                   high_confidence_pairs=sum(p['high_confidence'] for p in pairs),
                   isolation_groups=len(groups), multi_image_groups=int((groups.images > 1).sum()),
                   reliable_groups=frame.reliable_group_id.nunique(),
                   conflicting_groups=int((frame[conflict].group_id.nunique())),
                   proposed_conflict_exclusions=int(conflict.sum()), exact_redundant=int(redundant.sum()),
                   split_sizes=split.split.value_counts().to_dict(),
                   group_integrity_verified=True, final_test_evaluation_enabled=False)
    (output / 'configuration.json').write_text(json.dumps(cfg, indent=2, sort_keys=True))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    return split, summary
