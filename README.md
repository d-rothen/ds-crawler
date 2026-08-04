# ds-crawler

[![CI](https://github.com/d-rothen/ds-crawler/actions/workflows/ci.yml/badge.svg)](https://github.com/d-rothen/ds-crawler/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ds-crawler.svg)](https://pypi.org/project/ds-crawler/)
[![Python](https://img.shields.io/pypi/pyversions/ds-crawler.svg)](https://pypi.org/project/ds-crawler/)

Build stable, queryable metadata for datasets whose structure is encoded in
file paths.

`ds-crawler` scans directories or ZIP archives, extracts IDs, hierarchy, and
properties with regular expressions, and writes a small set of versioned JSON
artifacts. It also aligns modalities, creates reproducible splits, copies or
extracts subsets, and indexes generated outputs.

- Directory and `.zip` inputs use the same API.
- Regex capture groups define IDs, hierarchy, and file properties.
- Include/exclude filters keep irrelevant files out of the index.
- Named splits and metadata scopes are first-class artifacts.
- Dataset semantics live in the shared `euler-dataset-contract` format.

## Installation

```bash
pip install ds-crawler
```

Progress bars are optional:

```bash
pip install "ds-crawler[progress]"
```

Python 3.9 or newer is required.

## Quick start

Given this dataset:

```text
foggy_rgb/
├── .ds_crawler/
│   ├── dataset-head.json
│   └── ds-crawler.json
├── scene_01/
│   ├── 0001.png
│   └── 0002.png
└── scene_02/
    └── 0001.png
```

describe its semantic identity in `.ds_crawler/dataset-head.json`:

```json
{
  "contract": {"kind": "dataset_head", "version": "1.0"},
  "dataset": {"id": "foggy_rgb", "name": "Foggy RGB"},
  "modality": {
    "key": "rgb",
    "meta": {"range": [0, 255]}
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

Then describe how paths become index entries in
`.ds_crawler/ds-crawler.json`:

```json
{
  "contract": {"kind": "ds_crawler_config", "version": "2.0"},
  "head_file": "dataset-head.json",
  "source": {"path": "."},
  "indexing": {
    "id": {
      "regex": "^(?P<scene>[^/]+)/(?P<frame>\\d+)\\.png$",
      "join_char": "+"
    },
    "hierarchy": {
      "regex": "^(?P<scene>[^/]+)/",
      "separator": ":"
    },
    "properties": {
      "basename": {
        "regex": "^(?P<frame>\\d+)\\.(?P<ext>png)$"
      }
    },
    "files": {"extensions": [".png"]},
    "constraints": {"flat_ids_unique": true}
  }
}
```

Index and persist the dataset:

```python
from ds_crawler import get_files, index_dataset_from_path

output = index_dataset_from_path("/data/foggy_rgb", save_index=True)

print(output["head"]["dataset"]["name"])
print(get_files(output))
```

The returned object is hydrated with the dataset head and indexing recipe for
convenience. The saved `index.json` stays minimal.

For a self-contained example that creates a temporary dataset, see
[`examples/basic_usage.py`](examples/basic_usage.py).

## Artifact layout

Every indexed dataset stores metadata below `.ds_crawler/`:

| File | Purpose |
|---|---|
| `dataset-head.json` | Dataset identity, modality metadata, and namespaced addons. |
| `ds-crawler.json` | Source and indexing recipe. |
| `index.json` | Materialized recursive file index. |
| `split_<name>.json` | A named, filtered index with provenance. |

Several logical modalities can share one physical root. Pass
`metadata_scope="rgb"` (or another safe scope name) to path-based APIs to use
`.ds_crawler/rgb/`; scoped writes also maintain `.ds_crawler/scopes.json`.

See [the configuration reference](docs/configuration.md) for every supported
field, artifact shape, path filter, and hierarchy pattern.

## Common workflows

### Use the CLI

A CLI config can reference one or more datasets that already contain crawler
metadata:

```json
{
  "datasets": [
    {"path": "/data/foggy_rgb"},
    {"path": "/data/foggy_depth"}
  ]
}
```

```bash
ds-crawler index datasets.json
```

`index` is optional for backwards compatibility, so `ds-crawler datasets.json`
is equivalent. Useful flags include:

| Flag | Effect |
|---|---|
| `--strict` | Abort on duplicate IDs. |
| `--sample N` | Keep every Nth matched file deterministically. |
| `--match-index PATH` | Keep only hierarchy-qualified IDs found in another index. |
| `--metadata-scope SCOPE` | Read and write a scoped artifact set. |
| `--workdir PATH` | Resolve relative dataset paths from a specific directory. |
| `--verbose` | Log each skipped file. |

### Align modalities

```python
from ds_crawler import align_datasets

aligned = align_datasets(
    {"modality": "rgb", "source": "/data/foggy_rgb"},
    {"modality": "depth", "source": "/data/foggy_depth"},
)

rgb_path = aligned["scene-scene_01+frame-0001"]["rgb"]["path"]
```

Alignment is by leaf ID. For modalities that apply at an ancestor hierarchy
level, preserve the hierarchy and join in the consumer instead; the runnable
[`augmented RGB example`](examples/augmented_rgb_example.py) demonstrates that
layout.

### Create reproducible splits

```python
from ds_crawler import create_dataset_splits, load_dataset_split

create_dataset_splits(
    "/data/foggy_rgb",
    split_names=["train", "validation"],
    ratios=[80, 20],
    seed=42,
)

train = load_dataset_split("/data/foggy_rgb", "train")
```

Use `create_aligned_dataset_splits` for several modalities,
`create_hierarchy_dataset_splits` for hierarchy rules, or
`create_mapped_dataset_splits` for an explicit qualified-ID mapping.
`copy_dataset_splits` transfers an existing partition to a sibling dataset and
fails before writing if any target ID is missing.

### Write generated data

```python
from ds_crawler import DatasetWriter

head = {
    "contract": {"kind": "dataset_head", "version": "1.0"},
    "dataset": {"id": "predicted_rgb", "name": "Predicted RGB"},
    "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
}

writer = DatasetWriter("/data/predicted_rgb", head=head)
path = writer.get_path("/scene:scene_01/frame-0001", "0001.png")
path.write_bytes(encoded_image)
writer.save_index()
```

`ZipDatasetWriter` provides the same index-building behavior for a new ZIP
archive and accepts bytes or writable file-like objects.

### Migrate legacy metadata

Normal loading is intentionally strict about the current schema. Migrate older
`output.json`-based datasets explicitly:

```bash
ds-crawler migrate-metadata /data/legacy_dataset
ds-crawler migrate-metadata /data/archive.zip
ds-crawler migrate-metadata /data/datasets --scan-zips
```

Run `ds-crawler migrate-metadata --help` for archive scanning, inline split,
and metadata-scope options.

## Python API

The main entry points are re-exported from `ds_crawler`:

| Area | Entry points |
|---|---|
| Build | `build_dataset_head`, `build_crawler_config`, `build_dataset_artifacts_from_files` |
| Index | `index_dataset`, `index_dataset_from_files`, `index_dataset_from_path` |
| Inspect | `get_files`, `collect_qualified_ids`, `get_dataset_contract`, `get_dataset_properties` |
| Split | `create_dataset_splits`, `create_aligned_dataset_splits`, `create_hierarchy_dataset_splits`, `create_mapped_dataset_splits`, `load_dataset_split` |
| Transform | `align_datasets`, `copy_dataset`, `extract_datasets`, `split_dataset`, `split_datasets` |
| Write | `DatasetWriter`, `ZipDatasetWriter` |
| Validate | `validate_crawler_config`, `validate_dataset`, `validate_output` |

All core path-based workflows support directories and ZIP archives. Scope-aware
operations accept `metadata_scope`; multi-source operations also expose
per-source scope arguments where needed.

## Development

```bash
git clone https://github.com/d-rothen/ds-crawler.git
cd ds-crawler
uv sync --extra dev
uv run pytest
uv run ruff check .
uv build
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change. To
report a vulnerability, follow [SECURITY.md](SECURITY.md).

## License status

This repository does not yet declare a software license. A maintainer must
select and add an open-source license before the project is represented or
distributed as open-source software.
