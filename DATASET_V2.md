# clean_grouped_v2: construction and review

This is a **new benchmark**, not an accuracy improvement claim. The historical
70.69% validation accuracy used different evaluation membership. Do not compute
an improvement delta against it. ResNet18 combined, full layer4, smoothing 0.05,
augmentation, AdamW, scheduler and validation macro-F1 checkpoint selection stay
unchanged. Test evaluation and training remain disabled by default.

## Findings and evidence limits

The existing notebook decodes every image with EXIF correction, hashes dimensions
and RGB bytes, excludes exact cross-label groups and keeps the first path for
same-label exact duplicates. It then splits individual images 70/15/15 using
seed 42. Only afterward does it audit train/validation with dHash <=4 and
48x48 grayscale MAE <=3. That audit neither groups components nor examines the
future test partition, so it cannot enforce near-duplicate split isolation.

The user supplied previous audit totals: 49 candidate pairs, 39 same-label,
10 cross-label, 46 training and 38 validation images. These are screening
results, not confirmed duplicates or evidence of which label is correct.
No dataset archive, saved audit CSVs/contact sheets or audit outputs were found
in the checkout. Notebook outputs contain no audit totals. The stated original
59,099 and baseline 28,818 images and the supplied candidate totals have therefore
not been remeasured. Actual v2 groups, exclusions, class distributions and split
sizes are **pending the Colab audit**, not zero. They cannot be inferred from the
49 pairs because v2 scans all readable images before splitting.

## Protocol and functions

`clean_grouped_v2.py` is the single supporting module. The notebook downloads it
from branch `salma`, resolves the branch to an immutable commit URL and verifies
the notebook-pinned module SHA-256 before importing it. No `main` fallback or
automatic import from the current working directory is allowed. Provenance is
saved as `RUN_DIR/v2_module_source.json`. If the remote revision is unavailable
or different, the loader stops with an actionable error. An explicit uploaded
file via `V2_LOCAL_MODULE_PATH` must pass the same checksum. `prepare` consumes the complete readable
manifest, including all original exact copies/conflicts. It re-decodes and checks
the input pixel hashes, builds proposals in a new output directory and never
writes, deletes or relabels a source image.

- `features`: EXIF-normalized RGB, dHash64 from grayscale 9x8, grayscale and RGB
  comparison images at 48x48, bilinear resize. Exact identity includes dimensions.
- Candidate retrieval: five disjoint hash blocks enumerate every pair of exact
  representatives with dHash distance <=4, followed by grayscale MAE <=3.
  Exact-hash members are attached to their representative's components.
- High-confidence edges: same native dimensions, dHash <=1, RGB MAE <=1,
  99th-percentile RGB difference <=4, both grayscale standard deviations >=10.
  These are provisional dataset rules, not a validated duplicate classifier.
  No prediction, validation score or test score informs the thresholds.
- `Components`: deterministic connected components for high-confidence duplicate
  groups and separate candidate isolation groups. Transitive chains can join
  dissimilar endpoints. All candidate components remain indivisible in the split,
  including uncertain edges; this is a conservative isolation constraint, not a
  claim that all members express the same emotion or are the same photograph.
- Cross-label candidate components: quarantine the **whole component** in the
  proposed benchmark, including uncertain conflicts. No majority-vote or automatic
  relabeling. This can exclude false-positive matches and must be reviewed.
  `decision_manifest.csv` and `class_distribution.csv` are written before splitting,
  showing all proposed exclusions. Review before authorizing training.
- Same-label exact copies: select the first sorted source path; flag the others
  as redundant in the manifest. Same-label non-exact candidates remain present,
  grouped within a split. This avoids uncertain deletions but may overweight groups.
- `assign_splits`: deterministic per-class greedy balancing by sample count with
  stable hash tie breaks and 70/15/15 targets. Every eligible isolation group is
  label-homogeneous. Require >=3 eligible groups per class and all five classes
  in every split; otherwise stop. Ratios are approximate for indivisible groups.
- `verify_splits`: unique paths, correct targets (Angry=0, Fear=1, Happy=2, Sad=3,
  Surprise=4), class coverage, and no exact, reliable or isolation group crossing.
  Folder `Suprise` remains mapped to label `Surprise` by the notebook.

No reviewed group decisions are imported automatically. To revise a false-positive
constraint or conflict exclusion, explicitly revise/version the construction policy,
rerun audit and review a new fingerprint. Do not silently hand-edit split manifests.

## Audit artifacts and class impact

All v2 artifacts live under `RUN_DIR/clean_grouped_v2/`:

- `candidate_pairs.csv`: metrics, confidence and cross-label flags.
- `cross_label_candidates.csv`, `conflicting_groups_manifest.csv`: candidate
  cross-label pairs and all members of conflicting components, including exact
  conflicts whose representative pair may have the same label.
- `split_integrity.json`: unique paths, assigned/excluded counts, class mapping
  and zero crossing counts for exact, reliable and isolation groups.
- `groups.csv`: every isolation group, size, labels, number of reliable components.
- `decision_manifest.csv`: every readable source, IDs, original label and decision.
- `class_distribution.csv`: original readable, retained, proposed conflict exclusions,
  redundant exact copies and quarantine fraction per class. Compare both counts and
  fractions for Fear, Sad, Angry and all other classes before adopting the protocol.
- `split_distribution.csv`, `split_manifest.csv` and per-split manifests: exact sizes,
  labels, targets, original paths and group IDs. No source images are copied.
- `contact_sheets/*.png`, `contact_sheet_index.csv`: all multi-image components,
  paginated at 36 images/page with original labels and an index back to full paths.
- `configuration.json`, `summary.json`: rules, versions, code hash, archive fingerprint,
  readable dataset fingerprint, configuration hash and exact split-file hash.

The earlier notebook cells also save historical exact-cleaning reports and a plot
in the new run directory. Those are baseline diagnostics; use the **nested v2**
reports for v2 exclusions, distributions and membership. Invalid image details
remain in the parent `invalid_images.csv`; the v2 fingerprint covers readable
images, while the archive SHA-256 identifies the full original archive.

## Colab: audit first

1. Open the matching revision of `CNN_resnet18.ipynb` from branch `salma`. Use
   a CPU runtime. When the matching module is on GitHub, keep
   `V2_LOCAL_MODULE_PATH=None` for automatic verified download. These local edits
   have not been committed or pushed: until the matching revision is available
   remotely, upload the revised notebook and `clean_grouped_v2.py` to Colab and
   set `V2_LOCAL_MODULE_PATH='/content/clean_grouped_v2.py'`. Do not alter the
   pinned module checksum to bypass a mismatch.
2. Set `ZIP_PATH` and `OUTPUT_ROOT`. Keep the expected archive fingerprint; a
   different archive must be treated as another documented dataset version.
3. Set `DATASET_PROTOCOL='clean_grouped_v2'`, `RUN_TRAINING=False`,
   `RESUME_RUN_DIR=None`, `REVIEWED_V2_SPLIT_SHA256=None`. Run all.
4. Inspect printed counts, the nested decision/group/pair tables and **all** contact
   sheets. Compare per-class exclusion fractions and actual split proportions.
   Resolve whether the proposed rules and exclusions are appropriate before training.
5. Preserve the entire audit folder, configuration, versions and summary. Record the
   approved `split_sha256` and configuration/code hashes. Repeated audits with the same
   data, code, versions and seed must produce identical manifests. Unique run IDs
   are intentionally excluded from dataset construction hashes.

## Colab: training after review

Keep `combined`, `full_layer4`, smoothing 0.05 and all current training settings.
Set `REVIEWED_V2_SPLIT_SHA256` to the approved audit's exact hash, select GPU T4,
then set `RUN_TRAINING=True`. The notebook reconstructs the proposal in a fresh
`clean_grouped_v2_...` experiment directory and rejects a mismatching split before
creating a model or loading pretrained weights. V2 resume is deliberately disabled;
no previous checkpoint, report or history is overwritten. The baseline switch
remains the default and retains its historical count and split-hash checks.

Keep the final test cell disabled. Construction may inspect the future test images
and labels only under this predefined construction policy. After construction,
restrict test manifests/contact sheets to the dataset custodian and do not inspect
test examples or labels during model selection. This notebook provides procedural
separation, not access-control enforcement: artifacts contain source labels, and
an unused test loader is constructed by the existing training setup. Test labels
and performance must not guide threshold, model or hyperparameter selection.

## Limitations and local verification

Repeated subjects are not detected: identity-disjoint evaluation requires subject
IDs or a separately reviewed identity protocol. Crops, poses, reflections, alignment,
compression and color shifts can escape dHash/MAE; similar or low-detail faces can
be false matches. RGB checks reduce some false positives without proving identity.
Connected-component chaining may make large groups and bias exclusion by class.
Inspect group sizes and quarantine fractions. Group isolation cannot guarantee
absence of unidentified duplicates. A fresh Colab extraction is preferable: the
existing extraction cache is keyed by archive hash, but its completion marker does
not recheck cached files against ZIP entries. V2 verifies scan-to-feature consistency,
not authenticity of a previously edited extraction cache.

Candidate generation has worst-case quadratic time for large shared hash buckets;
all contact sheets and features can take CPU time and disk space on the full dataset.
No silent candidate cap is imposed. Pin recorded library versions for replication.

Changed files: notebook config/split/audit branches and validation count/comparison
guards; supporting module functions listed above; this document; synthetic regression
checks in `tests/test_clean_grouped_v2.py`. SimpleCNN notebooks are unchanged.

Local verification uses tiny synthetic images only: transitive grouping, exact and
near conflicts, deterministic manifests under reordered inputs, source bytes unchanged,
class mapping/coverage, indivisible groups, infeasible split rejection and output
non-overwrite checks. Notebook JSON and every code cell are parsed. No real-data audit,
GPU training, test evaluation, accuracy measurement, commit, push or merge is performed.

Final integration verification additionally checks the notebook-pinned module hash,
audit-only defaults, salma commit resolution with mocked network responses, checksum
rejection, explicit local loading and failure without a fallback branch. No live
Colab session or real-data audit has been executed locally.
