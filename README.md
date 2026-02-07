# ds-crawler

Regex-based dataset metadata crawler and indexer. Extracts structured
identifiers from file paths, organises them into a hierarchical index
(`output.json`), and provides utilities for aligning, copying, splitting
and writing multi-modal datasets.

```
pip install .                 # core
pip install ".[progress]"     # with tqdm progress bars
pip install ".[dev]"          # with pytest + tqdm
```

Requires Python >= 3.9. No runtime dependencies.

---

## Quick start

### Index a dataset

Every dataset needs a small JSON config (`ds-crawler.json`) that tells
the crawler how to extract file IDs and (optionally) a hierarchy from
paths. Place it either inside `.ds_crawler/ds-crawler.json` at the
dataset root, or pass the config dict directly.

```python
from ds_crawler import index_dataset_from_path

# Reads ds-crawler.json from inside the dataset, returns an output dict
output = index_dataset_from_path("/data/rgb", save_index=True)
```

### Align modalities by ID

```python
from ds_crawler import align_datasets

aligned = align_datasets(
    {"modality": "rgb",   "source": "/data/rgb"},
    {"modality": "depth", "source": "/data/depth"},
)

for file_id, mods in aligned.items():
    if "rgb" in mods and "depth" in mods:
        print(mods["rgb"]["path"], mods["depth"]["path"])
```

### Split into train / val

```python
from ds_crawler import split_datasets

result = split_datasets(
    source_paths=["/data/rgb", "/data/depth"],
    suffixes=["train", "val"],
    ratios=[80, 20],
    seed=42,
)
```

### Write model outputs back to disk

```python
from ds_crawler import DatasetWriter

writer = DatasetWriter(
    "/output/segmentation",
    name="segmentation",
    type="segmentation",
    euler_train={"used_as": "target", "modality_type": "semantic"},
)

for sample in dataloader:
    pred = model(sample["rgb"])
    path = writer.get_path(sample["full_id"], f"{sample['id']}.png")
    save_image(pred, path)

writer.save_index()  # writes output.json for later re-indexing
```

---

## Configuration

### `ds-crawler.json`

Placed at `<dataset_root>/.ds_crawler/ds-crawler.json` (or passed as a
dict). Minimal example:

```json
{
  "name": "my_rgb",
  "path": "/data/my_rgb",
  "type": "rgb",
  "id_regex": "^frame_(\\d+)\\.png$",
  "properties": {
    "euler_train": {
      "used_as": "input",
      "modality_type": "rgb"
    }
  }
}
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Human-readable dataset name |
| `path` | yes | Root directory or `.zip` archive |
| `type` | yes | One of `"rgb"`, `"depth"`, `"segmentation"`, `"metadata"` |
| `id_regex` | yes | Regex applied to each file's relative path. Capture groups form the file ID (joined by `id_regex_join_char`). |
| `properties.euler_train.used_as` | yes | `"input"`, `"target"`, or `"condition"` |
| `properties.euler_train.modality_type` | yes | Identifier token (e.g. `"rgb"`, `"depth"`) |
| `hierarchy_regex` | no | Regex with named groups to build a hierarchy tree |
| `named_capture_group_value_separator` | no | Character joining group name and value in hierarchy keys (default `":"`) |
| `basename_regex` | no | Regex applied to basename only; properties stored per file |
| `path_regex` | no | Regex applied to full relative path; properties stored per file |
| `intrinsics_regex` | no | Regex matching camera intrinsics files |
| `extrinsics_regex` | no | Regex matching camera extrinsics files |
| `id_regex_join_char` | no | Join character for multi-group IDs (default `"+"`) |
| `file_extensions` | no | Restrict to these extensions (e.g. `[".png", ".jpg"]`) |
| `flat_ids_unique` | no | If `true`, IDs must be globally unique (not just within hierarchy node) |
| `output_json` | no | Path to a pre-existing `output.json` to load instead of crawling |

### Multi-dataset config

For the CLI, wrap multiple dataset configs in:

```json
{
  "datasets": [
    { "name": "rgb",   "path": "/data/rgb",   "type": "rgb",   "id_regex": "..." },
    { "name": "depth", "path": "/data/depth", "type": "depth", "id_regex": "..." }
  ]
}
```

---

## Output format (`output.json`)

The index produced by the crawler:

```json
{
  "name": "my_rgb",
  "type": "rgb",
  "id_regex": "...",
  "id_regex_join_char": "+",
  "euler_train": { "used_as": "input", "modality_type": "rgb" },
  "named_capture_group_value_separator": ":",
  "dataset": {
    "files": [
      { "path": "frame_001.png", "id": "001", "path_properties": {}, "basename_properties": {} }
    ],
    "children": {
      "scene:Scene01": {
        "files": [ ... ],
        "children": { ... }
      }
    }
  }
}
```

`dataset` is a recursive node: each node has `files` (leaf entries) and
`children` (named sub-nodes). Hierarchy keys follow the pattern
`<group_name><separator><value>` (e.g. `scene:Scene01`).

---

## CLI

```
ds-crawler CONFIG [OPTIONS]
```

| Flag | Description |
|---|---|
| `-o, --output PATH` | Write a single combined output file (otherwise writes per-dataset) |
| `-w, --workdir PATH` | Prepend to relative dataset paths |
| `-s, --strict` | Abort on duplicate IDs or >20% regex misses |
| `--sample N` | Keep every Nth matched file |
| `--match-index PATH` | Only include file IDs present in this output.json |
| `-v, --verbose` | Log every skipped file |

---

## Python API

All public symbols are re-exported from the top-level `ds_crawler`
package. They live in four submodules:

### Indexing (`ds_crawler.parser`)

#### `index_dataset(config, *, strict, save_index, sample, match_index) -> dict`

Index a single dataset from a config dict.

#### `index_dataset_from_path(path, *, strict, save_index, force_reindex, sample, match_index) -> dict`

Index a dataset by path, reading `ds-crawler.json` from the dataset root.
Returns a cached `output.json` when available (unless `force_reindex=True`).

#### `index_dataset_from_files(config, files, *, base_path, strict, sample, match_index) -> dict`

Index from pre-collected file paths (useful when files aren't on the
local filesystem).

#### `DatasetParser(config, *, strict, sample, match_index)`

Lower-level class wrapping the full `Config` object. Methods:

- `parse_all() -> list[dict]`
- `parse_dataset(ds_config, ...) -> dict`
- `parse_dataset_from_files(ds_config, files, ...) -> dict`
- `write_output(output_path)`
- `write_outputs_per_dataset(filename="output.json") -> list[Path]`

---

### Traversal & filtering (`ds_crawler.traversal`)

#### `get_files(output_json) -> list[str]`

Flat list of every file path in an output dict (or list of output dicts).

#### `collect_qualified_ids(output_json) -> set[tuple[str, ...]]`

Set of `(*hierarchy_keys, file_id)` tuples. Qualified IDs distinguish
files that share the same raw ID but live at different hierarchy levels.

#### `filter_index_by_qualified_ids(output_json, qualified_ids) -> dict`

Return a pruned copy of the output dict keeping only the given IDs.

#### `split_qualified_ids(qualified_ids, ratios, *, seed=None) -> list[set]`

Partition qualified IDs by percentage (e.g. `[80, 20]`). Deterministic
when `seed` is `None` (sorted order) or fixed.

---

### Operations (`ds_crawler.operations`)

#### `align_datasets(*args) -> dict[str, dict[str, dict]]`

Align multiple modalities by file ID. Each positional argument is a dict
with `"modality"` (label) and `"source"` (path or output dict). Returns
`{file_id: {modality: file_entry, ...}, ...}`.

#### `copy_dataset(input_path, output_path, *, index=None, sample=None) -> dict`

Copy dataset files to a new location (directory or `.zip`), preserving
structure. Returns `{"copied": int, "missing": int, "missing_files": [...]}`.

#### `split_dataset(source_path, ratios, target_paths, *, qualified_ids, seed) -> dict`

Split a single dataset into multiple targets by percentage.

#### `split_datasets(source_paths, suffixes, ratios, *, seed) -> dict`

Split multiple aligned datasets using their common ID intersection.
Target paths are derived by appending the suffix
(`/data/rgb` + `"train"` -> `/data/rgb_train`).

---

### Writer (`ds_crawler.writer`)

#### `DatasetWriter(root, *, name, type, euler_train, separator=":", **properties)`

Stateful helper that turns `(full_id, basename)` pairs into filesystem
paths while accumulating an `output.json`-compatible index.

| Method | Description |
|---|---|
| `get_path(full_id, basename, *, source_meta=None) -> Path` | Register a file and get the absolute path to write to. Directories are created automatically. |
| `build_output() -> dict` | Return the accumulated index as an output dict. |
| `save_index(filename="output.json") -> Path` | Persist the index to `<root>/.ds_crawler/<filename>`. |

`full_id` follows the format produced by euler-loading:
`/scene:Scene01/camera:Cam0/scene-Scene01+camera-Cam0+frame-00001`
where each `/key:value` segment maps to a directory level and the final
component is the ds-crawler file ID.

---

### Validation (`ds_crawler.validation`)

#### `validate_crawler_config(config, workdir=None) -> DatasetConfig`

Validate a `ds-crawler.json` dict. Raises `ValueError` on failure.

#### `validate_output(output) -> dict`

Validate an `output.json` object (single dict or list). Raises
`ValueError` on failure.

#### `validate_dataset(path) -> dict`

Check a dataset path for valid metadata files. Returns
`{"path", "has_config", "has_output", "config", "output"}`.

---

### Schema (`ds_crawler.schema`)

#### `DatasetDescriptor(name, path, type, properties={})`

Minimal dataset description dataclass. Class methods:

- `from_output(data, path) -> DatasetDescriptor`
- `from_output_file(path, dataset_root) -> list[DatasetDescriptor]`

---

### Config (`ds_crawler.config`)

#### `DatasetConfig`

Full dataset configuration (extends `DatasetDescriptor` with regex
fields). Usually created via `DatasetConfig.from_dict(data, workdir)` or
`load_dataset_config(data, workdir)`.

#### `Config(datasets: list[DatasetConfig])`

Container loaded with `Config.from_file(path, workdir)`.

---

## ZIP support

All operations work transparently with `.zip` archives. Paths like
`/data/dataset.zip` are handled the same as directories: the crawler
reads file listings from the archive, `copy_dataset` writes into a new
archive, and `save_index` / `write_outputs_per_dataset` embed
`output.json` inside the zip.

---

## Examples

See the [`examples/`](examples/) directory:

- **`example.py`** -- index a dataset, match against an existing index,
  and copy a subset.
- **`sample_realdrivesim.py`** -- subsample a multi-modal RealDriveSim
  dataset (RGB, depth, segmentation) preserving cross-modality alignment.
