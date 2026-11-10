# Changelog

## Unreleased

- Regenerate `meta/schema.json` from euler-dataset-contract: the shared
  `invalid`, `sparse` and `source_grid` meta fields, and `skyclass` accepting `null`,
  which the checked-in copy had fallen behind on
  (`test_checked_in_schema_is_current` failed). The test compares against the
  installed contract, so it needs the contract release that adds `invalid`,
  `sparse` and `source_grid`.
- **Fixed:** `save_index(filename=...)` on `DatasetWriter` and
  `ZipDatasetWriter` records that filename as `ds-crawler.json`'s
  `source.prebuilt_index_file`. It always recorded `index.json`, so an index
  saved under another name (euler-loading's output writers used
  `output.json`) left a dataset that could not be loaded back.
- `DatasetWriter.set_modality_meta(meta)` (also on `ZipDatasetWriter`) replaces
  the head's `modality.meta` before the index is saved, validated like the head.
  It is the meta counterpart of `set_head_addon`, for producers that learn
  their metadata while writing (e.g. euler-preprocess's crop boxes).
- Writers take `path_prefix`: a directory below the root that holds the
  dataset's files, included in every index path. With `metadata_scope`,
  several datasets can share one root without their files colliding, e.g.
  `rgb/scene/0001.png` and `depth/scene/0001.png` under scopes `rgb` and
  `depth`. Absolute prefixes and `..` are rejected.

## 2.11.0 (unreleased)

Add public detached-head APIs, validated byte commits, atomic metadata generations/ZIP publication, resume, and receipt-preserving copy/split/scopes. No tensor execution semantics.
