# Changelog

## Unreleased

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
