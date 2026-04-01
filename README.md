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

- `index_dataset(...)`
- `index_dataset_from_files(...)`
- `index_dataset_from_path(...)`
- `load_dataset_config(...)`
- `validate_crawler_config(...)`
- `validate_dataset(...)`
- `validate_output(...)`

Dataset metadata:

- `get_dataset_contract(source)`
- `get_dataset_properties(source)`
- `extract_dataset_properties(mapping)`

`get_dataset_contract(...)` returns a `DatasetHeadContract`. Use
`contract.get_namespace("euler_train")` to access addon payloads.

Traversal and filtering:

- `collect_qualified_ids(...)`
- `filter_index_by_qualified_ids(...)`
- `get_files(...)`
- `split_qualified_ids(...)`

Operations:

- `align_datasets(...)`
- `copy_dataset(...)`
- `extract_datasets(...)`
- `split_dataset(...)`
- `split_datasets(...)`
- `create_dataset_splits(...)`
- `create_aligned_dataset_splits(...)`
- `list_dataset_splits(...)`
- `load_dataset_split(...)`

Writers:

- `DatasetWriter(...)`
- `ZipDatasetWriter(...)`

Migration helpers:

- `migrate_dataset_metadata(...)`
- `migrate_dataset_zip(...)`
- `migrate_dataset_zips_in_folder(...)`

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
