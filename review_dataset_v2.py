"""Read-only review of a saved clean_grouped_v2 audit; never calls prepare/split.

Colab: upload this file and call review_saved_audit(AUDIT_DIR, DATA_DIR,
OUTPUT_DIR, expected_split_sha256=...). Only quarantined images are decoded.
"""
import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import textwrap
import tempfile
import zipfile

import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
EXPECTED_SPLIT_SHA256 = '7e9742da12ed2caa4652e23af7a78215f57fd1e44e37cec2e7c70cd664b66ec8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Components:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def validate(audit_dir, expected_hash):
    """Reconcile all readable rows, exclusions, saved graphs and split bytes."""
    audit = Path(audit_dir)
    split_path = audit / 'split_manifest.csv'
    require(sha(split_path) == expected_hash, 'Saved split bytes do not match locked SHA-256')
    summary = json.loads((audit / 'summary.json').read_text())
    require(summary['split_sha256'] == expected_hash, 'Summary split hash differs')
    manifest = pd.read_csv(audit / 'decision_manifest.csv', keep_default_na=False)
    split = pd.read_csv(split_path, keep_default_na=False)
    pairs = pd.read_csv(audit / 'candidate_pairs.csv', keep_default_na=False)
    require(manifest.path.is_unique and split.path.is_unique, 'Duplicate source paths')
    require(set(manifest.label) == set(CLASSES), 'Unexpected class labels')
    require(set(manifest.decision) <= {'eligible', 'quarantine_label_conflict', 'redundant_exact_copy'}, 'Unknown decision')
    for frame in (manifest, split):
        require((frame.target == frame.label.map(dict(zip(CLASSES, range(5))))).all(), 'Class target mismatch')
    eligible = manifest[manifest.decision == 'eligible']
    excluded = manifest[manifest.decision != 'eligible']
    conflicts = manifest[manifest.decision == 'quarantine_label_conflict']
    require(set(eligible.path) == set(split.path), 'Eligible/split paths differ')
    require(not set(excluded.path) & set(split.path), 'Excluded images entered split')
    require(len(eligible) + len(excluded) == len(manifest), 'Incomplete decision coverage')
    pd.testing.assert_frame_equal(
        eligible.sort_values('path').reset_index(drop=True),
        split[eligible.columns].sort_values('path').reset_index(drop=True), check_dtype=False)
    require(set(split.split) == {'train', 'validation', 'test'}, 'Unexpected splits')
    crossings = {key: int((split.groupby(key).split.nunique() > 1).sum())
                 for key in ['group_id', 'reliable_group_id', 'pixel_sha256']}
    require(not any(crossings.values()), 'Duplicate component crosses splits')
    for name, subset in split.groupby('split'):
        require(set(subset.label) == set(CLASSES), f'{name} missing class')
    mixed = manifest.groupby('group_id').label.nunique()
    conflict_ids = set(mixed.index[mixed > 1])
    require(set(conflicts.group_id) == conflict_ids, 'Conflict decision/group mismatch')
    require(set(conflicts.path) == set(manifest[manifest.group_id.isin(conflict_ids)].path), 'Partially quarantined component')
    redundant = manifest[manifest.decision == 'redundant_exact_copy']
    require(set(redundant.pixel_sha256) <= set(eligible.pixel_sha256), 'Redundant copy lacks retained exact representative')
    require(eligible.pixel_sha256.is_unique, 'Repeated eligible exact hashes')
    expected_counts = dict(readable_images=len(manifest), retained_images=len(split),
                           candidate_pairs=len(pairs), conflicting_groups=len(conflict_ids),
                           proposed_conflict_exclusions=len(conflicts), exact_redundant=len(redundant))
    for key, count in expected_counts.items():
        require(summary[key] == count, f'Summary count mismatch: {key}')
    require(summary['split_sizes'] == split.split.value_counts().to_dict(), 'Split counts differ')
    rows = manifest[['path', 'label', 'pixel_sha256']].sort_values('path').to_dict('records')
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(summary['dataset_sha256'] == fingerprint, 'Readable dataset fingerprint differs')
    # Reconstruct connected components from saved edges, without image detection.
    graph = Components(manifest.path)
    reliable_graph = Components(manifest.path)
    for _, members in manifest.groupby('pixel_sha256', sort=False):
        paths = members.path.tolist()
        for p in paths[1:]:
            graph.union(paths[0], p)
            reliable_graph.union(paths[0], p)
    by_path = manifest.set_index('path')
    for pair in pairs.itertuples():
        require(pair.path_a in by_path.index and pair.path_b in by_path.index, 'Pair path missing')
        require(by_path.at[pair.path_a, 'label'] == pair.label_a and
                by_path.at[pair.path_b, 'label'] == pair.label_b, 'Pair labels differ from manifest')
        graph.union(pair.path_a, pair.path_b)
        if str(pair.high_confidence).lower() == 'true':
            reliable_graph.union(pair.path_a, pair.path_b)
    check = manifest[['path', 'group_id']].copy()
    check['reconstructed'] = check.path.map(graph.find)
    require((check.groupby('group_id').reconstructed.nunique() == 1).all() and
            (check.groupby('reconstructed').group_id.nunique() == 1).all(), 'Saved edges cannot reproduce isolation components')
    reliable_check = manifest[['path', 'reliable_group_id']].copy()
    reliable_check['reconstructed'] = reliable_check.path.map(reliable_graph.find)
    require((reliable_check.groupby('reliable_group_id').reconstructed.nunique() == 1).all() and
            (reliable_check.groupby('reconstructed').reliable_group_id.nunique() == 1).all(), 'Saved edges cannot reproduce reliable components')
    saved_conflicts = audit / 'conflicting_groups_manifest.csv'
    if saved_conflicts.exists():
        pd.testing.assert_frame_equal(conflicts.sort_values('path').reset_index(drop=True),
            pd.read_csv(saved_conflicts, keep_default_na=False).sort_values('path').reset_index(drop=True), check_dtype=False)
    scan_path = audit.parent / 'audit_manifest.csv'
    scan_checked = scan_path.exists()
    if scan_checked:
        scan = pd.read_csv(scan_path, keep_default_na=False)
        cols = ['path', 'label', 'pixel_sha256']
        pd.testing.assert_frame_equal(manifest[cols].sort_values('path').reset_index(drop=True),
            scan[cols].sort_values('path').reset_index(drop=True), check_dtype=False)
    checks = dict(passed=True, split_sha256=expected_hash, **expected_counts,
                  all_exclusions_accounted_for=True, graph_reconstruction_passed=True,
                  crossing_groups=crossings, original_readable_scan_checked=scan_checked,
                  source_file_existence_for_all_exclusions_checked=False,
                  invalid_files_scope='Not in readable manifest; inspect parent invalid_images.csv separately')
    return manifest, split, pairs, checks


def conflict_evidence(manifest, pairs):
    """Separate exact/reliable conflicts from uncertain links and chaining risks."""
    result = []
    conflict = manifest[manifest.decision == 'quarantine_label_conflict']
    path_group = manifest.set_index('path').group_id
    pair_groups = pairs.path_a.map(path_group)
    for group_id, group in conflict.groupby('group_id', sort=True):
        edges = pairs[pair_groups == group_id]
        exact_conflicts = int((group.groupby('pixel_sha256').label.nunique() > 1).sum())
        reliable_conflicts = int((group.groupby('reliable_group_id').label.nunique() > 1).sum())
        high = edges.high_confidence.astype(str).str.lower().eq('true')
        reliable_count = group.reliable_group_id.nunique()
        exact_count = group.pixel_sha256.nunique()
        # Connectivity after exact collapse: non-complete graphs require transitive links.
        total_pairs = exact_count * (exact_count - 1) // 2
        chain = len(edges) < total_pairs
        reason = ('exact_pixel_label_conflict' if exact_conflicts else
                  'high_confidence_component_label_conflict' if reliable_conflicts else
                  'uncertain_candidate_component_label_conflict')
        result.append(dict(group_id=group_id, images=len(group),
            labels='|'.join(sorted(set(group.label))), reason=reason,
            exact_pixel_groups=exact_count, exact_conflicting_groups=exact_conflicts,
            reliable_components=reliable_count, reliable_conflicting_components=reliable_conflicts,
            candidate_edges=len(edges), uncertain_edges=int((~high).sum()),
            chain_risk=chain, expanded_beyond_reliable_component=reliable_count > 1,
            priority='first' if chain or reliable_count > 1 or not reliable_conflicts else 'standard',
            **{label: int((group.label == label).sum()) for label in CLASSES}))
    return pd.DataFrame(result, columns=['group_id', 'images', 'labels', 'reason', 'exact_pixel_groups',
        'exact_conflicting_groups', 'reliable_components', 'reliable_conflicting_components',
        'candidate_edges', 'uncertain_edges', 'chain_risk', 'expanded_beyond_reliable_component', 'priority', *CLASSES])


def contact_sheets(conflicts, source_root, output):
    """One group per sheet/page, large 320px panels, full ID and original label."""
    root = Path(source_root).resolve()
    folder = output / 'conflict_contact_sheets'; folder.mkdir()
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default(size=18)
    status, pages = [], []
    for gid, group in conflicts.groupby('group_id', sort=True):
        group = group.sort_values(['label', 'path'])
        for start in range(0, len(group), 9):
            subset = group.iloc[start:start + 9]
            canvas = Image.new('RGB', (1050, 125 + 490 * ((len(subset) + 2) // 3)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 10), '\n'.join(textwrap.wrap(gid, 85)), fill='black', font=font)
            draw.text((12, 65), f'Original labels; quarantine only; page {start // 9 + 1}; images {len(group)}', fill='black', font=font)
            for k, item in enumerate(subset.itertuples()):
                x, y = (k % 3) * 350 + 12, 125 + (k // 3) * 490
                path = (root / item.path).resolve()
                require(path.is_relative_to(root), 'Image path escapes source root')
                state, detail = 'verified', ''
                try:
                    with Image.open(path) as src:
                        image = ImageOps.exif_transpose(src).convert('RGB')
                        actual = hashlib.sha256(str(image.size).encode() + b'|' + image.tobytes()).hexdigest()
                        require(actual == item.pixel_sha256, 'Source pixels differ from saved audit')
                        image.thumbnail((320, 320))
                        # Enlarge small originals only for viewing; no source or detector changes.
                        image = ImageOps.contain(image, (320, 320))
                        canvas.paste(image, (x, y))
                except Exception as error:
                    state, detail = 'unavailable_or_changed', str(error)
                    draw.text((x, y + 60), 'IMAGE UNAVAILABLE / CHANGED', font=font, fill='red')
                caption = (f'Original label: {item.label}\nImage {start+k+1}/{len(group)}\n' +
                           '\n'.join(textwrap.wrap(item.path, 30)))
                lines = caption.splitlines()
                draw.multiline_text((x, y + 325), '\n'.join(lines[:7]), fill='black', font=font)
                status.append(dict(group_id=gid, path=item.path, label=item.label,
                                   status=state, detail=detail))
            name = f'{gid}_{start // 9 + 1:02d}.png'
            canvas.save(folder / name)
            pages.append(dict(group_id=gid, file=f'conflict_contact_sheets/{name}'))
    pd.DataFrame(status).to_csv(output / 'conflict_image_status.csv', index=False)
    return pages, status



def extract_conflict_sources(audit_dir, zip_path, expected_split_sha256=EXPECTED_SPLIT_SHA256):
    """For a fresh runtime: extract ONLY quarantined sources into a new temp dir.

    No grouping, cleaning or splitting is rerun. The ZIP and audit stay read-only.
    """
    manifest, _, _, _ = validate(audit_dir, expected_split_sha256)
    summary = json.loads((Path(audit_dir) / 'summary.json').read_text())
    require(sha(zip_path) == summary['archive_sha256'], 'Original archive SHA-256 mismatch')
    selected = manifest[manifest.decision == 'quarantine_label_conflict']
    root = Path(tempfile.mkdtemp(prefix='fer_conflict_sources_')).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for path in selected.path:
            target = (root / path).resolve()
            require(target.is_relative_to(root), 'Image path escapes temporary folder')
            matches = [name for name in names if name == path or name.endswith('/' + path)]
            require(len(matches) == 1, f'Archive source missing or ambiguous: {path}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(matches[0]) as source, target.open('xb') as dest:
                import shutil
                shutil.copyfileobj(source, dest)
    return root


def review_saved_audit(audit_dir, source_root, output_dir, expected_split_sha256=EXPECTED_SPLIT_SHA256):
    audit, output = Path(audit_dir).resolve(), Path(output_dir).resolve()
    require(not output.is_relative_to(audit) and not audit.is_relative_to(output),
            'Review output must be separate from audit directory')
    if source_root is not None:
        root = Path(source_root).resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output), 'Review output overlaps image sources')
    tracked = sorted(p for p in audit.iterdir() if p.is_file())
    if (audit.parent / 'audit_manifest.csv').exists():
        tracked.append(audit.parent / 'audit_manifest.csv')
    before = {str(p): sha(p) for p in tracked}
    manifest, split, pairs, checks = validate(audit, expected_split_sha256)
    output.mkdir(parents=True, exist_ok=False)
    excluded = manifest[manifest.decision != 'eligible']
    conflicts = manifest[manifest.decision == 'quarantine_label_conflict']
    evidence = conflict_evidence(manifest, pairs).sort_values(['priority', 'images', 'group_id'], ascending=[True, False, True])
    evidence.to_csv(output / 'conflict_group_review.csv', index=False)
    conflict_edges = pairs[pairs.path_a.isin(conflicts.path)].copy()
    conflict_edges['group_id'] = conflict_edges.path_a.map(manifest.set_index('path').group_id)
    conflict_edges.to_csv(output / 'conflict_edges_review.csv', index=False)
    # All exclusions retained in a derivative ledger; original files unchanged.
    excluded.to_csv(output / 'all_exclusions_ledger.csv', index=False)
    conflicts.to_csv(output / 'conflict_members_review.csv', index=False)
    distribution = pd.DataFrame(index=CLASSES)
    distribution['original_readable'] = manifest.label.value_counts().reindex(CLASSES, fill_value=0)
    distribution['retained'] = split.label.value_counts().reindex(CLASSES, fill_value=0)
    distribution['quarantined'] = conflicts.label.value_counts().reindex(CLASSES, fill_value=0)
    distribution['quarantine_percent'] = 100 * distribution.quarantined / distribution.original_readable
    distribution.to_csv(output / 'conflict_class_impact.csv', index_label='label')
    template = evidence[['group_id', 'images', 'reason', 'priority']].copy()
    for field in ['reviewer', 'review_status', 'visual_relationship', 'recommendation', 'notes']:
        template[field] = ''
    template.to_csv(output / 'review_decisions_template.csv', index=False)
    pages, status = (contact_sheets(conflicts, source_root, output) if source_root is not None else ([], []))
    checks['conflict_images_verified'] = sum(s['status'] == 'verified' for s in status)
    checks['conflict_images_unavailable_or_changed'] = sum(s['status'] != 'verified' for s in status)
    checks['visual_review_complete'] = False
    checks['review_output_does_not_apply_decisions'] = True
    links = ['<meta charset="utf-8"><h1>Conflict groups only — original labels</h1>',
             '<p>Review findings only. No labels, exclusions or splits are edited.</p>']
    for page in pages:
        links.append(f'<p>{html.escape(page["group_id"])}</p><a href="{page["file"]}"><img width="1050" src="{page["file"]}"></a>')
    (output / 'index.html').write_text('\n'.join(links), encoding='utf-8')
    reasons = evidence.reason.value_counts().to_dict()
    report = ['# Saved Dataset V2 conflict review', '',
        f'Locked split SHA-256: `{expected_split_sha256}`',
        f'Integrity recheck: PASSED; {len(manifest)} readable images; {len(split)} retained.',
        f'{len(evidence)} conflict groups / {len(conflicts)} quarantined images.',
        f'All {len(excluded)} excluded readable images accounted for in the decision manifest.',
        '', '## Evidence categories', json.dumps(reasons, indent=2), '',
        distribution.to_csv(index_label='label'),
        f'Groups with uncertain expansion: {int(evidence.expanded_beyond_reliable_component.sum())}.',
        f'Groups with non-complete candidate graph after exact collapse: {int(evidence.chain_risk.sum())}.',
        '', '## Decision',
        'Retain the frozen split provisionally. Visual approval is still pending.',
        'Exact cross-label identity supports quarantine but does not identify the correct label.',
        'For uncertain expansion or chains, inspect every member and saved edge metrics.',
        'Large exact-copy groups alone do not demonstrate over-grouping; compare unique pixel groups.',
        'If different photographs/expressions were connected, record a recommendation for a separately',
        'versioned protocol. Never edit this benchmark or reuse its hash for revised membership.',
        'No model predictions or test performance should inform these recommendations.',
        '', f'Conflict image bytes checked: {checks["conflict_images_verified"]}; unavailable/changed: {checks["conflict_images_unavailable_or_changed"]}.',
        'Image verification does not substitute for human visual review.',
        'Review excluded groups only; do not browse retained test images for model selection.']
    (output / 'review_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    require(before == {str(p): sha(p) for p in tracked}, 'Input audit changed during review')
    require(sha(audit / 'split_manifest.csv') == expected_split_sha256, 'Split changed during review')
    checks['audit_files_unchanged'] = True
    (output / 'input_hashes.json').write_text(json.dumps(before, indent=2))
    (output / 'review_integrity.json').write_text(json.dumps(checks, indent=2, sort_keys=True))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--source-root', help='Original extracted Data directory; omit for tables-only review')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-split-sha256', default=EXPECTED_SPLIT_SHA256)
    args = parser.parse_args()
    print(json.dumps(review_saved_audit(args.audit_dir, args.source_root, args.output_dir,
                                       args.expected_split_sha256), indent=2))
