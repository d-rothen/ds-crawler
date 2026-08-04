# Configuration and artifact reference

This reference covers the on-disk contract used by `ds-crawler`. For a first
index, start with the [README quick start](../README.md#quick-start).

## Metadata directories

The default artifact set lives below the dataset root:

```text
dataset/
├── .ds_crawler/
│   ├── dataset-head.json
│   ├── ds-crawler.json
│   ├── index.json
│   └── split_train.json
└── ...
```

For several logical modalities in one root, use scopes:

```text
dataset/
├── .ds_crawler/
│   ├── scopes.json
│   ├── rgb/
│   │   ├── dataset-head.json
│   │   ├── ds-crawler.json
│   │   └── index.json
│   └── camera_extrinsics/
│       ├── dataset-head.json
│       ├── ds-crawler.json
│       └── index.json
└── ...
```

Pass the same `metadata_scope` to every operation that should address a scoped
artifact set. Omitting it always addresses the unscoped layout.

## `dataset-head.json`

The dataset head is owned by `euler-dataset-contract`, not by the crawler. Its
top-level fields are:

| Field | Meaning |
|---|---|
| `contract` | Must identify `dataset_head` and its contract version. |
| `dataset.id` | Stable machine-readable dataset identifier. |
| `dataset.name` | Human-readable name. |
| `dataset.attributes` | Optional dataset-wide attributes. |
| `modality.key` | Modality identifier such as `rgb` or `depth`. |
| `modality.meta` | Modality-specific semantic metadata. |
| `addons` | Optional namespaced integrations such as `euler_train`. |

Use `build_dataset_head(...)` to apply modality defaults and validate this
mapping programmatically.

## `ds-crawler.json`

A crawler configuration has four core areas:

```json
{
  "contract": {"kind": "ds_crawler_config", "version": "2.0"},
  "head_file": "dataset-head.json",
  "source": {"path": "."},
  "indexing": {}
}
```

`source.prebuilt_index_file` is optional. It lets a generated dataset load an
existing index without retaining regexes that can recreate it.

### IDs

`indexing.id.regex` must contain at least one capture group. For named groups,
each part becomes `<name>-<value>` and parts are joined with `join_char`:

```json
{
  "id": {
    "regex": "^(?P<scene>[^/]+)/(?P<frame>\\d+)\\.png$",
    "join_char": "+"
  }
}
```

The path `scene_01/0001.png` receives the ID
`scene-scene_01+frame-0001`. Unnamed capture groups contribute only their
values.

`id.override` replaces the captured value after the regex matches. It is useful
for a single semantic file such as `calib.json` that should be indexed as
`intrinsics`. Do not use one override for several files at the same hierarchy
level: they would become duplicates.

### Hierarchy

`indexing.hierarchy.regex` maps captures to nested `children` keys. Named
groups require `indexing.hierarchy.separator`:

```json
{
  "hierarchy": {
    "regex": "^(?P<scene>[^/]+)/(?P<camera>[^/]+)/",
    "separator": ":"
  }
}
```

This produces keys such as `scene:scene_01` and `camera:front`. Without a
hierarchy regex, matching files are stored in the root node. That is the right
model for a calibration file shared by every sample.

### Extracted properties

Properties preserve useful capture groups without changing identity:

| Field | Matched against | Saved as |
|---|---|---|
| `indexing.properties.path.regex` | Relative path | `path_properties` |
| `indexing.properties.basename.regex` | File basename | `basename_properties` |

Only named capture groups become properties.

### File selection

`indexing.files.extensions` is a list such as `[".png", ".npy"]`. Extensions
are normalized to include the leading dot.

`indexing.files.path_filters` supports:

| Field | Purpose |
|---|---|
| `include_regex` | Keep paths matching at least one regex. |
| `exclude_regex` | Reject paths matching any regex. |
| `include_terms` | Keep paths containing at least one term. |
| `exclude_terms` | Reject paths containing any term. |
| `term_match_mode` | `substring` or `path_segment`. |
| `case_sensitive` | Toggle case-sensitive term matching. |

Exclusions take precedence over inclusions.

### Duplicate constraints

`indexing.constraints.flat_ids_unique` controls duplicate scope:

- `true`: a file ID may appear only once in the entire dataset.
- `false`: a file ID may repeat in different hierarchy nodes, but not within
  the same node.

Duplicate entries are skipped with a warning. `strict=True` turns a duplicate
into an error.

## `index.json`

The saved index is intentionally small:

```json
{
  "contract": {"kind": "dataset_index", "version": "1.0"},
  "generator": {"name": "ds_crawler", "version": "2.10.0"},
  "execution": {},
  "index": {
    "files": [],
    "children": {}
  }
}
```

Each file entry contains `path`, `id`, `path_properties`, and
`basename_properties`; optional per-file data lives under `attributes`.
`children` recursively contains more index nodes.

Loading an index hydrates it with the sibling head and crawler config, so the
in-memory result also contains `head`, `head_file`, and `indexing`.

## Split artifacts

`split_<name>.json` uses the `dataset_split` contract and records:

- the split name and source index filename;
- generator and execution provenance (for example ratio, seed, or sampling);
- the filtered recursive index node.

Split artifacts do not duplicate the dataset head or crawler config. Loading a
split hydrates those sibling artifacts exactly like loading the full index.

Split names and metadata scopes are validated before they become path
components. Invalid names, missing mapped IDs, and cross-split duplicate IDs
fail before any artifact is written.

## Hierarchical modality patterns

### One ancestor file shared by descendants

Index a root-level or per-scene calibration as a separate modality. Its
hierarchy should end at the level where it applies. A root-level calibration
has no `hierarchy` block; a per-scene calibration captures only the scene.
Consumers can then apply that ancestor entry to all descendant samples.

### Several variants for one sample ID

Suppose RGB augmentations use `abc/aug_1.png` and `abc/aug_2.png`, while depth
uses `abc.png`. Give RGB the parent `abc` as a hierarchy key and each
augmentation its own leaf ID. Give depth the same `abc` hierarchy key. This
preserves every augmentation while allowing a hierarchy-aware consumer to join
the one depth file to both.

`align_datasets` is not appropriate for this pattern because it aligns only by
leaf ID. See [`examples/augmented_rgb_example.py`](../examples/augmented_rgb_example.py)
for a complete runnable setup.

## ZIP behavior

Directories and ZIP archives use the same metadata filenames and contracts.
ZIP metadata is stored under `.ds_crawler/` inside the archive. When an archive
contains a single wrapping directory whose name matches the ZIP stem,
`ds-crawler` normalizes that prefix while reading.

Metadata updates rewrite the archive once per batch. Already-compressed media
formats are stored without an unnecessary second compression pass.
