# Changelog

## Unreleased

- `build_index_output(head=..., index=..., indexing=None, generator=None,
  execution=None)` and `save_dataset_index(path, output, metadata_scope=None)`
  are public. Together they write a validated artifact set for files that
  already exist, with IDs and hierarchy chosen by the caller, as a prebuilt
  index (`source.prebuilt_index_file`). `save_dataset_index` also persists
  `index_dataset_from_files` output, regexes included, into another root.
  Callers no longer need the internal `ds_crawler.artifacts.save_output_artifacts`.
- Regenerate `meta/schema.json` from euler-dataset-contract 0.13.0: the
  shared `invalid`, `sparse` and `source_grid` meta fields, and `skyclass`
  accepting `null`.
- Require `euler-dataset-contract>=0.13.0`, the version `meta/schema.json` is
  generated from, and lock it. `test_checked_in_schema_is_current` compares
  the file with the installed contract's schema, and the lock still installed
  0.8.0, so the regenerated file failed in CI.
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
