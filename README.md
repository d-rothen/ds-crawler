# ds-crawler

Regex-based dataset crawler, indexer, and dataset artifact toolkit.

`ds-crawler` indexes files by regex, stores the crawl result as a minimal
`index.json`, and keeps semantic dataset metadata in a separate
`dataset-head.json` contract shared with other packages such as
`euler-loading` and `euler-train`. The dataset head is validated through
`euler-dataset-contract`.

It works with both directories and `.zip` archives.

```bash
pip install .
pip install ".[progress]"
pip install ".[dev]"
uv pip install "ds-crawler @ git+https://github.com/d-rothen/ds-crawler"
```

Requires Python `>=3.9`.

## What lives on disk

Every dataset uses a `.ds_crawler/` metadata directory:

```text
dataset_root/
├── .ds_crawler/
│   ├── dataset-head.json
│   ├── ds-crawler.json
│   ├── index.json
│   ├── split_train.json
│   └── split_val.json
└── ...
```

The files have distinct roles:

| File | Purpose |
|---|---|
| `.ds_crawler/dataset-head.json` | Shared semantic dataset contract: identity, modality, modality metadata, namespaced addon metadata. |
| `.ds_crawler/ds-crawler.json` | Crawl recipe: source path, regexes, path filters, file extensions, hierarchy rules, or a prebuilt index reference. |
| `.ds_crawler/index.json` | Materialized full dataset index. No duplicated head/config metadata. |
| `.ds_crawler/split_<name>.json` | Named split artifact with its own contract, provenance, and filtered `index` node. |

When several logical modalities share one physical root or zip, store each
metadata set under a scope:

```text
dataset_root/
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
├── calib.json
└── ...
```

Pass `metadata_scope="rgb"` or `metadata_scope="camera_extrinsics"` to the
path-based APIs to read/write that scoped artifact set. Scoped writes update
`.ds_crawler/scopes.json`, a discoverability manifest listing known scopes and
their artifact filenames. Omitting `metadata_scope` preserves the legacy
unscoped layout exactly.

The in-memory objects returned by `index_dataset_from_path(...)` and
`load_dataset_split(...)` are hydrated outputs. They include `head` and
`indexing` for convenience, even though the on-disk `index.json` only stores
the minimal index artifact.

## Quick start

### 1. Create dataset metadata

`dataset-head.json`:

```json
{
  "contract": {
    "kind": "dataset_head",
    "version": "1.0"
  },
  "dataset": {
    "id": "foggy_rgb",
    "name": "Foggy RGB"
  },
  "modality": {
    "key": "rgb",
    "meta": {
      "range": [0, 255],
      "dimensions": {
        "height": 375,
        "width": 1242,
        "channels": 3
      }
    }
  },
  "addons": {
    "euler_train": {
      "version": "1.0",
      "used_as": "input",
      "slot": "dehaze.input.rgb"
    }
  }
}
```

`ds-crawler.json`:

```json
{
  "contract": {
    "kind": "ds_crawler_config",
    "version": "2.0"
  },
  "head_file": "dataset-head.json",
  "source": {
    "path": "."
  },
  "indexing": {
    "id": {
      "regex": "^(?P<scene>[^/]+)/(?P<frame>\\d+)\\.png$",
      "join_char": "+"
    },
    "hierarchy": {
      "regex": "^(?P<scene>[^/]+)/(?P<frame>\\d+)\\.png$",
      "separator": ":"
    },
    "properties": {
      "basename": {
        "regex": "^(?P<frame>\\d+)\\.(?P<ext>png)$"
      },
      "path": {
        "regex": "^(?P<scene>[^/]+)/"
      }
    },
    "files": {
      "extensions": [".png"],
      "path_filters": {
        "include_terms": ["fog"],
        "term_match_mode": "path_segment"
      }
    },
    "constraints": {
      "flat_ids_unique": true
    }
  }
}
```

### 2. Index the dataset

```python
from ds_crawler import index_dataset_from_path

output = index_dataset_from_path("/data/foggy_rgb", save_index=True)

print(output["head"]["dataset"]["name"])
print(output["head"]["modality"]["key"])
print(output["index"].keys())
```

`save_index=True` writes:

- `.ds_crawler/dataset-head.json`
- `.ds_crawler/ds-crawler.json`
- `.ds_crawler/index.json`

For a shared physical root, pass a metadata scope:

```python
rgb = index_dataset_from_path(
    "/data/muses",
    save_index=True,
    metadata_scope="rgb",
)
extrinsics = index_dataset_from_path(
    "/data/muses",
    save_index=True,
    metadata_scope="camera_extrinsics",
)
```

### 3. Align multiple modalities

```python
from ds_crawler import align_datasets

aligned = align_datasets(
    {"modality": "rgb", "source": "/data/foggy_rgb"},
    {"modality": "depth", "source": "/data/foggy_depth"},
)

for file_id, modalities in aligned.items():
    if "rgb" in modalities and "depth" in modalities:
        print(file_id, modalities["rgb"]["path"], modalities["depth"]["path"])
```

### Recipe: augmented modality (file-id as folder) matched against per-id GT

When one modality stores its file-id as a *parent folder* containing several
augmentation files, while a sibling modality keeps the file-id as a *filename
stem*, the two indices can still share hierarchy keys so that
`euler-loading`'s `hierarchical_modalities=` joins each augmentation to the
matching per-id GT file at sample-load time.

Layout:

```
rgb_root/
  abc/aug_1.png  abc/aug_2.png
  xyz/aug_1.png  xyz/aug_2.png
depth_root/
  abc.png  xyz.png
```

Augmented RGB indexing (`flat_ids_unique` is `false` so two augs of the same
file-id at the same hierarchy level are not flagged as duplicates):

```json
{
  "indexing": {
    "id":        {"regex": "^[^/]+/(?P<aug>[^/]+)\\.png$", "join_char": "+"},
    "hierarchy": {"regex": "^(?P<file_id>[^/]+)/[^/]+\\.png$", "separator": ":"},
    "files":     {"extensions": [".png"]},
    "constraints": {"flat_ids_unique": false}
  }
}
```

Per-id depth indexing (uses the same `file_id` named group so both indices
end up keyed by `file_id:<id>` at the same hierarchy depth):

```json
{
  "indexing": {
    "id":        {"regex": "^(?P<file_id>[^/]+)\\.png$", "join_char": "+"},
    "hierarchy": {"regex": "^(?P<file_id>[^/]+)\\.png$", "separator": ":"},
    "files":     {"extensions": [".png"]}
  }
}
```

Note that `align_datasets` is *not* the right tool here — it flattens by leaf
id only and would collapse augmentations.  The cross-modality join happens
inside `euler-loading` by passing depth as a hierarchical modality:

```python
MultiModalDataset(
    modalities={"rgb": Modality(path=rgb_root, ...)},
    hierarchical_modalities={"depth": Modality(path=depth_root, ...)},
)
```

A runnable end-to-end example is in
[`examples/augmented_rgb_example.py`](examples/augmented_rgb_example.py).

### Recipe: root-level calibration shared by every sample

Some archives keep calibration at the dataset root, for example
`rgb.zip/calib.json`, while regular samples live below path-based hierarchy
folders. Model that calibration as a hierarchical modality with no
`indexing.hierarchy` block. ds-crawler then stores `calib.json` directly in
the root `index.files` list, i.e. at hierarchy path `()`. Consumers such as
`euler-loading` treat root-level hierarchical files as ancestors of every
sample, so the calibration applies globally.

```json
{
  "indexing": {
    "id": {"regex": "^(calib)\\.json$", "join_char": "+"},
    "files": {
      "extensions": [".json"],
      "path_filters": {"include_regex": ["^calib\\.json$"]}
    },
    "constraints": {"flat_ids_unique": true}
  }
}
```

This is the same hierarchy model as per-scene or per-camera calibration: the
only difference is that the applicable hierarchy level is the root rather than
`scene:<id>` or `camera:<id>`. Do not add a `hierarchy.regex` for this case;
there is no path segment to extract.

Use `indexing.id.override` when the matched file's natural name is not the
logical calibration slot. For example, a file named `calib.json` or a UUID JSON
can be indexed as `intrinsics`, `extrinsics`, or `calibration`:

```json
{
  "indexing": {
    "id": {
      "regex": "^(calib)\\.json$",
      "join_char": "+",
      "override": "intrinsics"
    }
  }
}
```

The override replaces the ID extracted by `id.regex` after the regex has
matched. This matters for hierarchical modalities because downstream consumers
use the file ID as the semantic key, and repeated IDs at different ancestor
levels intentionally mean "the deeper file overrides the shallower one". Do not
set `id.override` when multiple distinct files can exist at the same hierarchy
level; they would all receive the same ID and ds-crawler will treat all but one
as duplicates.

### 4. Create named split artifacts

```python
from ds_crawler import create_dataset_splits, load_dataset_split

create_dataset_splits(
    "/data/foggy_rgb",
    split_names=["train", "val"],
    ratios=[80, 20],
    seed=42,
)

train_output = load_dataset_split("/data/foggy_rgb", "train")
print(train_output["split"]["name"])
print(train_output["execution"]["split"])
```

Scoped split artifacts use the same namespace as the index:

```python
create_dataset_splits(
    "/data/muses",
    split_names=["train", "val"],
    ratios=[80, 20],
    metadata_scope="rgb",
)

train_output = load_dataset_split("/data/muses", "train", metadata_scope="rgb")
```

To reuse an existing partition on a sibling dataset (same qualified file
IDs), copy the splits over instead of recomputing them:

```python
from ds_crawler import copy_dataset_splits

copy_dataset_splits(
    "/data/foggy_rgb",
    "/data/foggy_depth",
    split_names=["test"],   # or None to copy every split
    override=False,         # pass True to replace existing target splits
)
```

The target dataset's `index.json` is used to resolve IDs. Any ID from a
source split that is missing on the target raises a `ValueError` — no
partial split is ever written.

To split by hierarchy keys, use `create_hierarchy_dataset_splits`. Level
indices are zero-based and values match the exact `children` keys in
`index.json` (for named captures this includes the configured separator,
for example `weather:fog`):

```python
from ds_crawler import create_hierarchy_dataset_splits

create_hierarchy_dataset_splits(
    "/data/foggy_rgb.zip",
    {
        "exclusive": True,
        "splits": [
            {
                "name": "front_fog",
                "clauses": [
                    {"levelIndex": 0, "values": ["camera_0"]},
                    {"levelIndex": 1, "values": ["fog"]},
                ],
            },
            {
                "name": "remaining_fog",
                "clauses": [
                    {"levelIndex": 1, "values": ["fog"]},
                ],
            },
        ],
    },
)
```

Directory and `.zip` datasets are both supported. With multiple source paths,
the split rules are applied to the intersection of hierarchy-qualified IDs so
the generated split artifacts stay aligned across modalities.

To split from an explicit JSON mapping of split name to full qualified file
IDs, use `create_mapped_dataset_splits`:

```python
from ds_crawler import create_mapped_dataset_splits

create_mapped_dataset_splits(
    "/data/foggy_rgb",
    {
        "fog": ["camera_0~fog~day~0001", "camera_1~fog~day~0002"],
        "clear": ["camera_0~clear~day~0003"],
    },
)
```

String IDs are split by `qualified_id_separator="~"` into
`(*hierarchy_keys, file_id)`. If a path segment itself contains `~`, pass that
ID as a JSON array of path segments instead. All requested IDs are validated
before writing, so an invalid mapping never leaves partial split artifacts.

### 5. Write generated outputs back to disk

```python
from ds_crawler import DatasetWriter

head = {
    "contract": {"kind": "dataset_head", "version": "1.0"},
    "dataset": {"id": "pred_rgb", "name": "Predicted RGB"},
    "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
    "addons": {
        "euler_train": {
            "version": "1.0",
            "used_as": "output",
            "slot": "dehaze.output.rgb"
        }
    }
}

writer = DatasetWriter("/tmp/predictions", head=head)

path = writer.get_path("/scene:Scene01/0001", "0001.png")
path.write_bytes(b"data")

writer.save_index()
```

`DatasetWriter.save_index()` writes the current artifact set, not a legacy
`output.json`.

## The current schemas

### `dataset-head.json`

Core keys:

- `contract`
- `dataset`
- `modality`
- `addons`

Notes:

- `dataset.id` is the stable dataset identifier.
- `modality.key` replaces the old root-level `type`.
- `modality.meta` replaces the old root-level `meta`.
- `modality.meta.file_types` is inferred from indexed files when the crawler
  or writer can determine it.
- `addons` is namespaced. Payloads such as `addons.euler_train` and
  `addons.euler_loading` are owned by their respective packages.

### `ds-crawler.json`

Core keys:

- `contract`
- `head_file`
- `source`
- `indexing`

`source.prebuilt_index_file` is optional. When set, the dataset can be loaded
from an existing `index.json` without re-crawling source files.

Supported indexing areas:

- `indexing.id.regex`
- `indexing.id.join_char`
- `indexing.id.override`
- `indexing.hierarchy.regex`
- `indexing.hierarchy.separator`
- `indexing.properties.path.regex`
- `indexing.properties.basename.regex`
- `indexing.files.extensions`
- `indexing.files.path_filters`
- `indexing.constraints.flat_ids_unique`

There are no camera-specific config fields anymore.

Supported `indexing.files.path_filters` keys:

- `include_regex`
- `exclude_regex`
- `include_terms`
- `exclude_terms`
- `term_match_mode` with `substring` or `path_segment`
- `case_sensitive`

### `index.json`

On disk, the full index artifact is intentionally minimal:

```json
{
  "contract": {
    "kind": "dataset_index",
    "version": "1.0"
  },
  "generator": {
    "name": "ds_crawler",
    "version": "0"
  },
  "execution": {},
  "index": {
    "files": [],
    "children": {}
  }
}
```

`index` is a recursive node structure:

- `files`: leaf file entries
- `children`: nested hierarchy nodes

Each file entry has:

- `path`
- `id`
- `path_properties`
- `basename_properties`

### `split_<name>.json`

Split artifacts are versioned and self-describing:

```json
{
  "contract": {
    "kind": "dataset_split",
    "version": "1.0"
  },
  "split": {
    "name": "train",
    "source_index_file": "index.json"
  },
  "generator": {
    "name": "ds_crawler",
    "version": "0"
  },
  "execution": {
    "ratio": 80,
    "seed": 42
  },
  "index": {
    "files": [],
    "children": {}
  }
}
```

They do not duplicate `dataset-head.json` or `ds-crawler.json`. Loading a
split hydrates those sibling artifacts automatically.

## CLI

Indexing:

```bash
ds-crawler CONFIG.json
ds-crawler index CONFIG.json
```

Main flags:

| Flag | Description |
|---|---|
| `-o, --output PATH` | Write a single JSON output file instead of per-dataset `.ds_crawler/index.json`. |
| `-w, --workdir PATH` | Base directory for relative config paths. |
| `-s, --strict` | Abort on duplicate IDs or excessive regex misses. |
| `--sample N` | Keep every `N`th matched file. |
| `--match-index PATH` | Only keep IDs present in another hydrated output/index. |
| `-v, --verbose` | Enable debug logging. |

Metadata migration:

```bash
ds-crawler migrate-metadata /data/legacy_dataset
ds-crawler migrate-metadata /data/archive.zip
ds-crawler migrate-metadata /data/datasets --scan-zips
```

Copy splits between sibling datasets:

```bash
ds-crawler copy-splits /data/foggy_rgb /data/foggy_depth
ds-crawler copy-splits /data/foggy_rgb /data/foggy_depth --split test
ds-crawler copy-splits /data/foggy_rgb /data/foggy_depth --override
```

`--split NAME` can be repeated to pick a subset; without it every split
found on the source is copied. `--override` replaces existing target
splits of the same name. Missing IDs on the target abort the command.

Migration notes:

- `--scan-zips` recursively scans subfolders for `.zip` archives by default.
- `--top-level-only` disables recursion.
- `--no-index` writes `dataset-head.json` and `ds-crawler.json` without
  rewriting `index.json`.
- Archive migration fails loudly when a `.zip` does not contain usable
  `.ds_crawler/` metadata.

## Python API

The main public entry points are re-exported from `ds_crawler`.

Indexing and config:

- `build_crawler_config(...)`
- `build_dataset_artifacts_from_files(...)`
- `build_dataset_head(...)`
- `index_dataset(...)`
- `index_dataset_from_files(...)`
- `index_dataset_from_path(...)`
- `load_dataset_config(...)`
- `validate_crawler_config(...)`
- `validate_dataset(...)`
- `validate_output(...)`

`index_dataset(...)`, `index_dataset_from_path(...)`, `load_dataset_config(...)`,
and `validate_dataset(...)` accept `metadata_scope=...` for scoped metadata.
`build_dataset_artifacts_from_files(...)` also accepts `metadata_scope=...` and
returns scoped artifact names plus `scopes.json`.

Dataset metadata:

- `get_dataset_contract(source)`
- `get_dataset_properties(source)`
- `extract_dataset_properties(mapping)`
- `list_metadata_scopes(path)`

`get_dataset_contract(...)` returns a `DatasetHeadContract`. Use
`contract.get_namespace("euler_train")` to access addon payloads.
`get_dataset_contract(path, metadata_scope=...)` and
`get_dataset_properties(path, metadata_scope=...)` read scoped heads.

Traversal and filtering:

- `collect_qualified_ids(...)`
- `filter_index_by_qualified_ids(...)`
- `get_files(...)`
- `split_qualified_ids(...)`

Operations:

- `align_datasets(...)`
- `copy_dataset(...)`
- `copy_dataset_splits(...)`
- `extract_datasets(...)`
- `split_dataset(...)`
- `split_datasets(...)`
- `create_dataset_splits(...)`
- `create_aligned_dataset_splits(...)`
- `create_hierarchy_dataset_splits(...)`
- `create_mapped_dataset_splits(...)`
- `list_dataset_splits(...)`
- `load_dataset_split(...)`

Split, copy, align, and extraction operations accept `metadata_scope=...`.
`copy_dataset(...)`, `copy_dataset_splits(...)`, and `split_dataset(...)` also
accept source/target scope overrides when the two sides use different scope
names. `split_datasets(...)` and `extract_datasets(...)` accept per-source or
per-config scope lists for multi-dataset workflows.

Writers:

- `DatasetWriter(...)`
- `ZipDatasetWriter(...)`

Both writers accept `metadata_scope=...` and write scoped artifacts plus the
`scopes.json` manifest.

Migration helpers:

- `migrate_dataset_metadata(...)`
- `migrate_dataset_zip(...)`
- `migrate_dataset_zips_in_folder(...)`
- `migrate_inline_splits(...)`

Migration helpers and `ds-crawler migrate` also accept `metadata_scope=...`
or `--metadata-scope` for scoped legacy metadata conversion.

## ZIP support

All core workflows support `.zip` archives:

- indexing from zipped datasets
- writing metadata back into archives
- loading inline split artifacts from archives
- migrating legacy archive metadata in place

When updating metadata inside a `.zip`, `ds-crawler` rewrites the archive once
per metadata batch, not once per file.

## Validation behavior

Current datasets are expected to follow the new schema. `ds-crawler` does not
attempt legacy fallback when loading normal datasets anymore.

If a dataset is malformed, loading and validation fail with explicit errors.
Use `ds-crawler migrate-metadata ...` to rewrite older datasets into the
current layout first.

```sh
ds-crawler migrate-metadata --scan-zips .
```

## Examples

See [`examples/`](examples/) for small usage snippets. The test suite under
[`tests/`](tests/) is also a good source of current-schema examples.
