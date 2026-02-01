# ds-crawler

Regex-based dataset metadata crawler and indexer for computer vision datasets. Automatically indexes files, extracts deterministic IDs via regex capture groups, and organises them into a hierarchical JSON structure. Designed for multi-modality alignment (e.g. matching RGB frames to depth maps by ID) and for feeding structured metadata into dataloaders.

## Installation

```bash
pip install ds-crawler

# with optional progress bars
pip install ds-crawler[progress]

uv pip install "ds-crawler @ git+https://github.com/d-rothen/ds-crawler"
```

Requires Python >= 3.9. No runtime dependencies (only the standard library). `tqdm` is optional for progress bars.

## Quick start

### CLI

```bash
# Index datasets defined in a config file, write a single output JSON
ds-crawler config.json -o output.json

# Write output.json into each dataset's root directory instead
ds-crawler config.json

# Verbose logging / strict mode (abort on errors) / custom workdir
ds-crawler config.json -v -s -w /data/
```

| Flag | Description |
|---|---|
| `config` | Path to a JSON configuration file (required, positional) |
| `-o, --output PATH` | Write all results to a single JSON file |
| `-v, --verbose` | Enable verbose (DEBUG) logging |
| `-s, --strict` | Abort on errors instead of warning and continuing |
| `-w, --workdir PATH` | Prepend this directory to all relative dataset paths |

### Python API

```python
from ds_crawler import index_dataset, index_dataset_from_path, index_dataset_from_files, get_files
```

#### `index_dataset(config, *, strict=False, save_index=False) -> dict`

Index a single dataset from a configuration dict.

| Parameter | Type | Description |
|---|---|---|
| `config` | `dict[str, Any]` | Dataset configuration (see [Configuration](#configuration)) |
| `strict` | `bool` | If `True`, raise on errors; if `False`, warn and skip. Default `False` |
| `save_index` | `bool` | If `True`, write `output.json` into the dataset root. Default `False` |

**Returns:** A dict containing the hierarchical index (see [Output format](#output-format)).

#### `index_dataset_from_files(config, files, *, base_path=None, strict=False) -> dict`

Index a dataset from a pre-collected iterable of file paths, bypassing filesystem/ZIP discovery.

| Parameter | Type | Description |
|---|---|---|
| `config` | `dict[str, Any]` | Dataset configuration |
| `files` | `Iterable[str \| Path]` | File paths to index |
| `base_path` | `str \| Path \| None` | Base path to compute relative paths from. Defaults to `config["path"]` |
| `strict` | `bool` | Abort on errors. Default `False` |

**Returns:** Hierarchical index dict.

#### `index_dataset_from_path(path, *, strict=False, save_index=False, force_reindex=False) -> dict`

Index a dataset by its root path. Configuration is loaded from a `ds-crawler.json` file inside the dataset directory (or ZIP archive).

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the dataset root directory or `.zip` file |
| `strict` | `bool` | Abort on errors. Default `False` |
| `save_index` | `bool` | Write `output.json` into the dataset root. Default `False` |
| `force_reindex` | `bool` | Re-index even if a cached `output.json` exists. Default `False` |

**Returns:** Hierarchical index dict (served from cache when available and `force_reindex=False`).

#### `get_files(output_json) -> list[str]`

Extract a flat list of all file paths from a crawler output.

| Parameter | Type | Description |
|---|---|---|
| `output_json` | `dict \| list[dict]` | A single output dict or a list of output dicts |

**Returns:** `list[str]` of relative file paths.

## Configuration

A configuration file is a JSON object with a `datasets` array. Each entry describes one dataset to index.

```jsonc
{
  "datasets": [
    {
      "name": "my_dataset",          // Human-readable name (required)
      "path": "/data/my_dataset",    // Root directory or .zip file path (required)
      "type": "rgb",                 // Modality: "rgb", "depth", or "segmentation" (required)
      "id_regex": "...",             // Regex with capture groups to extract a unique file ID (required)

      // Optional fields
      "basename_regex": "...",       // Regex matched against the filename only
      "path_regex": "...",           // Regex matched against the relative path
      "hierarchy_regex": "...",      // Regex whose capture groups define the hierarchy tree
      "named_capture_group_value_separator": ":",  // Separator for hierarchy keys (required when hierarchy_regex uses named groups)
      "intrinsics_regex": "...",     // Regex to match camera intrinsics files
      "extrinsics_regex": "...",     // Regex to match camera extrinsics files
      "flat_ids_unique": false,      // true = IDs must be globally unique; false = unique per hierarchy level
      "id_regex_join_char": "+",     // Character joining capture group values into the ID string
      "file_extensions": [".png"],   // Override default extensions for this type
      "output_json": "custom.json",  // Custom output path (overrides default)
      "properties": {}               // Arbitrary user metadata, merged into output
    }
  ]
}
```

### Default file extensions by type

| Type | Extensions |
|---|---|
| `rgb` | `.png`, `.jpg`, `.jpeg` |
| `depth` | `.png`, `.exr`, `.npy`, `.pfm` |
| `segmentation` | `.png` |

### Embedded configuration

Instead of a multi-dataset config file, you can place a `ds-crawler.json` inside a dataset's root directory (or inside a `.zip` archive). This is used by `index_dataset_from_path()`.

## Regex fields explained

All regex fields are matched against the **relative path** from the dataset root (except `basename_regex`, which matches only the filename).

| Field | Matched against | Purpose |
|---|---|---|
| `id_regex` | Relative path | Named capture groups are joined with `id_regex_join_char` to form a deterministic ID (e.g. `scene-Scene01+camera-Camera_0+frame-00001`). Files in different datasets that produce the same ID can be aligned across modalities. |
| `basename_regex` | Filename only | Capture groups are stored as `basename_properties` on each file entry. |
| `path_regex` | Relative path | Capture groups are stored as `path_properties` on each file entry. |
| `hierarchy_regex` | Relative path | Capture groups define nested hierarchy levels. With named groups and a separator (e.g. `":"`), keys are formatted as `name:value` (e.g. `scene:Scene01 > camera:Camera_0 > frame:00001`). |
| `intrinsics_regex` | Relative path | Matched files are attached as `camera_intrinsics` at the deepest hierarchy level whose groups match. |
| `extrinsics_regex` | Relative path | Same as above for `camera_extrinsics`. |

## Output format

Each indexed dataset produces a dict with this structure:

```jsonc
{
  "name": "VKITTI2",
  "type": "rgb",
  "id_regex": "...",
  "id_regex_join_char": "+",
  "hierarchy_regex": "...",
  "named_capture_group_value_separator": ":",

  // User-defined properties from config are merged here
  "gt": true,

  "dataset": {
    "children": {
      "scene:Scene01": {
        "children": {
          "camera:Camera_0": {
            "camera_intrinsics": "Scene01/clone/intrinsics/Camera_0_intrinsics.txt",
            "camera_extrinsics": "Scene01/clone/extrinsics/Camera_0_extrinsics.txt",
            "children": {
              "frame:00001": {
                "files": [
                  {
                    "path": "Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg",
                    "id": "scene-Scene01+variation-clone+camera-Camera_0+frame-00001",
                    "path_properties": { "scene": "Scene01", "variation": "clone", "camera": "Camera_0" },
                    "basename_properties": { "frame": "00001", "ext": "jpg" }
                  }
                ]
              }
            }
          }
        }
      }
    }
  }
}
```

Each **file entry** contains:

| Key | Type | Description |
|---|---|---|
| `path` | `str` | Relative path from the dataset root |
| `id` | `str` | Deterministic ID built from `id_regex` capture groups joined by `id_regex_join_char` |
| `path_properties` | `dict[str, str]` | Named capture groups from `path_regex` |
| `basename_properties` | `dict[str, str]` | Named capture groups from `basename_regex` |

Each **hierarchy node** may contain:

| Key | Type | Description |
|---|---|---|
| `children` | `dict[str, node]` | Child nodes keyed by `name:value` |
| `files` | `list[entry]` | File entries at this level |
| `camera_intrinsics` | `str \| None` | Relative path to the intrinsics file for this level |
| `camera_extrinsics` | `str \| None` | Relative path to the extrinsics file for this level |

## ZIP support

Datasets stored as `.zip` archives are handled transparently. Point `path` to the `.zip` file and the crawler will enumerate entries inside it. Embedded `ds-crawler.json` and cached `output.json` files are read from / written into the archive.

## Cross-modality matching

Files across datasets that produce the same deterministic ID can be matched:

```python
from ds_crawler import index_dataset

rgb_output = index_dataset(rgb_config)
depth_output = index_dataset(depth_config)

rgb_files = {f["id"]: f for f in get_all_files(rgb_output)}
depth_files = {f["id"]: f for f in get_all_files(depth_output)}

for fid in rgb_files:
    if fid in depth_files:
        print(rgb_files[fid]["path"], "<->", depth_files[fid]["path"])
```

## License

See repository for license information.
