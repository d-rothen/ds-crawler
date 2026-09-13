# Validated publication (2.11.0, unreleased)

`DatasetWriter` and `ZipDatasetWriter` retain their legacy APIs. The new
`ValidatedDatasetWriter` stages completed encoded bytes and validated records,
and requires explicit finalization. It never executes tensor algorithms.

```python
from ds_crawler import ValidatedDatasetWriter

with ValidatedDatasetWriter("output.zip", head=detached_head, plan=output_plan,
                            expected_full_ids=["/scene/frame/variant"]) as writer:
    # Encode successfully before committing. Attributes carry the one selected
    # artifact receipt (inline or a content-addressed record reference).
    writer.commit_bytes("/scene/frame/variant", "result.npy", encoded_bytes,
                        attributes=attributes, records=records)
    writer.finalize()
```

A plan is the contract's `OutputPlan` 2.0 and selects one logical output field.
Crawler validates internal receipt references, plan identity, output IDs, counts,
index digests and actual encoded artifact bytes. It validates the producer's
assertions; it does not certify source-file verification or tensor execution.
A producer can supply a `publication_validator(output, read_record, read_bytes)`
callback for additional checks; executable callbacks are never persisted.
The loading strict producer uses this boundary for source-backed replay.

`head` returns a detached snapshot. Legacy writers now expose
`set_head_addon()` and `set_hierarchy_separator()` so consumers need not mutate
private state. A validated writer freezes semantic head data and rejects
changing it after writes. `get_path()` is refused on this writer because it
cannot establish that encoding succeeded; use `commit_bytes()` instead.

Directory bytes use content-addressed `artifacts/` paths, independent of logical
qualified IDs. Metadata and records are staged in immutable generation folders
inside the selected `.ds_crawler` scope. `publication.json` is replaced atomically
and names/digests the complete generation. `read_metadata_json()` resolves this
pointer and checks published metadata digests. Partial/interrupted finalization
leaves the old publication intact. A missing expected output prevents completion.
This is a single-owner process-interruption guarantee, not a distributed writer.

Directory resume requires `resume=True` and the identical plan/revision. An
identical qualified-ID/receipt/artifact write is idempotent; conflicts fail.
Repeated directory finalization is supported. ZIP output stages privately,
replaces the archive as a unit, and allows one finalization. Existing ZIPs and
ZIP resume are refused; `close()` without finalization publishes nothing.

`resolve_receipt()` in the contract and crawler's `read_record()` require one
location: inline **or** `records/<canonical-sha256>.json` under the same metadata
scope. Missing/corrupt records and traversal fail. Copy and split preserve each
selected mapping and its records, even across ZIP/directory/scope relocation.
Subsets get new count/index digests; source/recipe/artifact identities stay pinned.
Sampling materialized outputs uses qualified IDs so identical bytes do not
collapse distinct samples. Inline splits refer to records in the owning complete
publication and do not claim to replace its full-dataset membership summary.

`finalize_output(root, output, records=..., expected_full_ids=...)` is the public
validated directory finalizer for callers that already constructed an index.
Plain filesystem data is files-only until an explicit finalization. Legacy
`save_index()` is not retroactively an encoding transaction.

The contract resolves from PyPI through the normal dependency declaration; no
local source override or sibling checkout is required. When maintainers update
the lock with `uv lock --upgrade-package euler-dataset-contract`, it selects the
latest compatible public release. Wheel metadata requires
`euler-dataset-contract>=0.8.0`.
