# Using ds-crawler with Dataloaders

## Concept

Datasets of the same domain often share an identical directory structure but hold
different modalities (RGB, depth, segmentation, ...).  `ds-crawler` indexes each
modality independently and produces output objects whose **file IDs are
deterministic** — derived from the regex capture groups in each dataset's config.
When two datasets share the same hierarchy (scene/frame/camera/...), their file
IDs will match, giving you a ready-made mapping across modalities.

## Quick Start

```
pip install ds_crawler
```

### 1. Place a `ds-crawler.config` in each dataset root

```
/data/vkitti2/rgb/ds-crawler.config        # type: rgb
/data/vkitti2/depth/ds-crawler.config      # type: depth
```

Each config is a JSON object with the regex rules for that modality:

```json
{
  "name": "vkitti2-rgb",
  "path": "/data/vkitti2/rgb",
  "type": "rgb",
  "basename_regex": "^rgb_(?P<frame>\\d+)\\.(?P<ext>jpg|png)$",
  "id_regex": "^(?P<scene>Scene\\d+)/(?P<variation>[^/]+)/frames/rgb/(?P<camera>Camera_\\d+)/rgb_(?P<frame>\\d+)\\.(?:jpg|png)$",
  "hierarchy_regex": "^(?P<scene>Scene\\d+)/(?P<variation>[^/]+)/frames/rgb/(?P<camera>Camera_\\d+)/rgb_(?P<frame>\\d+)\\.(?:jpg|png)$",
  "named_capture_group_value_separator": ":"
}
```

> **Key rule:** design the `id_regex` capture groups so that the same
> physical sample (scene + frame + camera) produces the **same ID string**
> across all modalities.

### 2. Index both datasets

```python
from ds_crawler import index_dataset_from_path

rgb_index   = index_dataset_from_path("/data/vkitti2/rgb")
depth_index = index_dataset_from_path("/data/vkitti2/depth")
```

Each returned dict has the shape of a single `output.json` entry — a
hierarchical tree under the `"dataset"` key, with file entries at the leaves.

### 3. Build an ID-to-file lookup per modality

```python
def collect_files(node):
    """Recursively collect all file entries from an index."""
    files = list(node.get("files", []))
    for child in node.get("children", {}).values():
        files.extend(collect_files(child))
    return files

rgb_by_id   = {f["id"]: f for f in collect_files(rgb_index["dataset"])}
depth_by_id = {f["id"]: f for f in collect_files(depth_index["dataset"])}
```

### 4. Match and load paired samples

```python
common_ids = rgb_by_id.keys() & depth_by_id.keys()

for sample_id in sorted(common_ids):
    rgb_path   = f"/data/vkitti2/rgb/{rgb_by_id[sample_id]['path']}"
    depth_path = f"/data/vkitti2/depth/{depth_by_id[sample_id]['path']}"
    # load and use both files ...
```

File paths in the index are **relative to the dataset root**, so prepend the
root when opening files.

## Using a config dict instead of a config file

If you prefer not to place config files on disk, pass the config dict directly:

```python
from ds_crawler import index_dataset

rgb_index = index_dataset({
    "name": "vkitti2-rgb",
    "path": "/data/vkitti2/rgb",
    "type": "rgb",
    "basename_regex": "...",
    "id_regex": "...",
})
```

## Extra metadata on file entries

Each file entry carries additional properties extracted by the regexes:

| Key                   | Description                                    |
|-----------------------|------------------------------------------------|
| `path`                | Relative path from dataset root                |
| `id`                  | Deterministic ID built from `id_regex` groups  |
| `path_properties`     | Named groups captured by `path_regex`          |
| `basename_properties` | Named groups captured by `basename_regex`      |

These can be used for filtering (e.g. only frames from a specific scene) or for
additional matching logic beyond the ID.
