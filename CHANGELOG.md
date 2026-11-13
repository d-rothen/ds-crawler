# Changelog

## Unreleased

- A prebuilt index keeps a hierarchy separator declared without a regex
  (`indexing: {"hierarchy": {"separator": ":"}}`) when it is loaded back.
  It used to be dropped, so consumers could not split the `name:value`
  hierarchy keys of prebuilt scopes, nor of `DatasetWriter` outputs.
- Regenerate `meta/schema.json` from euler-dataset-contract 0.15.0, which
  registers optional geometry meta fields (camera models, transform
  directions and frames, point-cloud columns and units), and require
  `euler-dataset-contract>=0.15.0`. Refresh `uv.lock` once 0.15.0 is published.
- The `euler_loading` addon rules are always registered; the guard for
  contracts older than 0.14 is gone.
- Dataset heads are checked against the rules of the `euler_loading` addon
  (from euler-dataset-contract 0.14, registered on import). They used to be
  checked only when euler-loading was imported too, so a head ds-crawler
  wrote or accepted could be one euler-loading rejects, e.g. `hierarchy_scope`
  without `used_as: condition`. Older contracts keep the previous behaviour.
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
