from copy import deepcopy

import pytest
from euler_dataset_contract import (
    parse_dataset_head,
    receipt_location,
)
from euler_dataset_contract.testing import descriptor_fixture

from ds_crawler import (
    DatasetWriter,
    ValidatedDatasetWriter,
    copy_dataset,
    index_dataset_from_path,
    split_dataset,
    validate_output_records,
)
from ds_crawler.zip_utils import read_metadata_json


def setup_writer(root, *, sidecar=False, scope=None, expected=None):
    vector = descriptor_fixture("phase2-five-field")
    head = {
        "contract": {"kind": "dataset_head", "version": "1.0"},
        "dataset": {"id": "derived_rgb", "name": "Derived RGB"},
        "modality": {
            "key": "rgb",
            "meta": {
                "range": [0, 1],
                "dimensions": {"height": 320, "width": 640, "channels": 3},
            },
        },
    }
    writer = ValidatedDatasetWriter(
        root,
        head=head,
        plan=vector["plan"],
        metadata_scope=scope,
        expected_full_ids=expected,
    )
    return writer, vector


def commit(writer, vector, fid="/changed/frame_0", *, sidecar=False):
    receipt = deepcopy(vector["receipt"])
    receipt["output_full_id"] = fid
    location, records = receipt_location(receipt, sidecar=sidecar)
    return writer.commit_bytes(
        fid,
        "sample.npy",
        b"synthetic-artifact",
        attributes={"custom": "retained", "euler_transforms": location},
        records=records,
    )


def test_detach_source_head(tmp_path):
    source = parse_dataset_head(
        {
            "contract": {"kind": "dataset_head", "version": "1.0"},
            "dataset": {"id": "source", "name": "source"},
            "modality": {"key": "rgb", "meta": {"range": [0, 255]}},
        }
    )
    writer = DatasetWriter(tmp_path, head=source)
    writer.set_head_addon("opaque", {"version": "1.0", "value": [1]})
    detached = writer.head
    detached["addons"]["opaque"]["value"].append(2)
    assert writer.head["addons"]["opaque"]["value"] == [1]
    assert not source.addons


@pytest.mark.parametrize("scope", [None, "rgb"])
@pytest.mark.parametrize("sidecar", [False, True])
def test_directory_zip_copy_split_preserve_records(tmp_path, scope, sidecar):
    writer, vector = setup_writer(tmp_path / "out", scope=scope)
    for i in range(4):
        commit(writer, vector, f"/changed/frame_{i}", sidecar=sidecar)
    writer.finalize()
    index = index_dataset_from_path(tmp_path / "out", metadata_scope=scope)
    validate_output_records(tmp_path / "out", index, metadata_scope=scope)
    copy_dataset(
        tmp_path / "out",
        tmp_path / "copy.zip",
        input_metadata_scope=scope,
        output_metadata_scope="moved",
    )
    zip_index = index_dataset_from_path(tmp_path / "copy.zip", metadata_scope="moved")
    validate_output_records(tmp_path / "copy.zip", zip_index, metadata_scope="moved")
    assert zip_index["head"]["addons"]["euler_transforms"]["mapping"]["count"] == 4
    split_dataset(
        tmp_path / "copy.zip",
        [0.5, 0.5],
        [tmp_path / "train", tmp_path / "test.zip"],
        metadata_scope="moved",
        seed=4,
    )
    for target in (tmp_path / "train", tmp_path / "test.zip"):
        out = index_dataset_from_path(target, metadata_scope="moved")
        validate_output_records(target, out, metadata_scope="moved")
        assert out["head"]["addons"]["euler_transforms"]["mapping"]["count"] == 2


def test_failure_during_finalization_keeps_old_publication(tmp_path, monkeypatch):
    import ds_crawler.records as records

    writer, vector = setup_writer(tmp_path / "out")
    commit(writer, vector)
    writer.finalize()
    before = read_metadata_json(tmp_path / "out", "publication.json")
    commit(writer, vector, "/changed/frame_1", sidecar=True)
    real = records._atomic_bytes

    def fail(path, data):
        if path.name == "publication.json":
            raise OSError("interrupted publication")
        return real(path, data)

    monkeypatch.setattr(records, "_atomic_bytes", fail)
    with pytest.raises(OSError, match="interrupted"):
        writer.finalize()
    assert read_metadata_json(tmp_path / "out", "publication.json") == before
    assert (
        index_dataset_from_path(tmp_path / "out")["head"]["addons"]["euler_transforms"][
            "mapping"
        ]["count"]
        == 1
    )
    monkeypatch.setattr(records, "_atomic_bytes", real)
    writer.finalize()
    assert (
        index_dataset_from_path(tmp_path / "out")["head"]["addons"]["euler_transforms"][
            "mapping"
        ]["count"]
        == 2
    )


def test_zip_unfinished_not_published_and_single_finalize(tmp_path):
    writer, vector = setup_writer(
        tmp_path / "out.zip", expected=["/changed/frame_0", "/changed/frame_1"]
    )
    commit(writer, vector)
    with pytest.raises(ValueError, match="incomplete"):
        writer.finalize()
    assert not (tmp_path / "out.zip").exists()
    writer.close()
    writer, vector = setup_writer(tmp_path / "out.zip")
    commit(writer, vector)
    writer.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        writer.finalize()
    with pytest.raises(ValueError, match="new destination"):
        setup_writer(tmp_path / "out.zip")


def test_records_bytes_and_counts_must_agree(tmp_path):
    writer, vector = setup_writer(tmp_path / "out")
    with pytest.raises(ValueError, match="bytes disagree"):
        writer.commit_bytes(
            "/changed/frame_0",
            "sample.npy",
            b"wrong",
            attributes={"euler_transforms": vector["inline"]},
        )
    assert len(writer) == 0
    with pytest.raises(ValueError):
        writer.commit_bytes("/changed/frame_0", "sample.npy", b"synthetic-artifact")
    commit(writer, vector, sidecar=True)
    path = next((tmp_path / "out/artifacts").iterdir())
    path.write_bytes(b"overwritten")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        writer.finalize()
    assert read_metadata_json(tmp_path / "out", "dataset-head.json") is None


def test_inline_splits_keep_receipts_in_owning_scope(tmp_path):
    from ds_crawler import (
        copy_dataset_splits,
        create_dataset_splits,
        list_dataset_splits,
        load_dataset_split,
    )

    writer, vector = setup_writer(tmp_path / "out", scope="a")
    for i in range(4):
        commit(writer, vector, f"/changed/frame_{i}", sidecar=True)
    writer.finalize()
    create_dataset_splits(
        tmp_path / "out", ["train", "test"], [0.5, 0.5], metadata_scope="a", seed=2
    )
    subset = load_dataset_split(tmp_path / "out", "train", metadata_scope="a")
    validate_output_records(tmp_path / "out", subset, metadata_scope="a", subset=True)
    copy_dataset(
        tmp_path / "out",
        tmp_path / "copy.zip",
        input_metadata_scope="a",
        output_metadata_scope="b",
    )
    copy_dataset_splits(
        tmp_path / "out",
        tmp_path / "copy.zip",
        source_metadata_scope="a",
        target_metadata_scope="b",
    )
    assert list_dataset_splits(tmp_path / "copy.zip", metadata_scope="b") == [
        "test",
        "train",
    ]
    moved = load_dataset_split(tmp_path / "copy.zip", "train", metadata_scope="b")
    validate_output_records(
        tmp_path / "copy.zip", moved, metadata_scope="b", subset=True
    )


def test_published_records_cannot_fall_back_to_unpublished_metadata(tmp_path):
    from euler_dataset_contract import canonical_digest, canonical_json

    writer, vector = setup_writer(tmp_path / "out")
    commit(writer, vector, sidecar=True)
    writer.finalize()
    publication = read_metadata_json(tmp_path / "out", "publication.json")
    record = next(path for path in publication["files"] if path.startswith("records/"))
    stale = tmp_path / "out/.ds_crawler" / record
    stale.parent.mkdir(parents=True)
    stale.write_text(canonical_json(next(iter(vector["records"].values()))))
    publication["files"].remove(record)
    del publication["digests"][record]
    publication["generation"] = canonical_digest(publication["digests"])[7:]
    (tmp_path / "out/.ds_crawler/publication.json").write_text(
        canonical_json(publication)
    )
    with pytest.raises(
        ValueError, match="missing receipt record in published generation"
    ):
        read_metadata_json(tmp_path / "out", record)
