"""Split the RDS clear/hazy datasets into train/val with aligned IDs."""

from ds_crawler import split_datasets

rgb_clear = "/cluster/scratch/drothenpiele/data/real-drive-sim/sample_10_zips/rgb_10.zip"
rgb_hazy = "/cluster/scratch/drothenpiele/data/real-drive-sim/sample_10_zips/radial_foggy_10.zip"
depth_hazy = "/cluster/scratch/drothenpiele/data/real-drive-sim/sample_10_zips/depth_10.zip"

result = split_datasets(
    source_paths=[rgb_clear, rgb_hazy, depth_hazy],
    suffixes=["train.zip", "val.zip", "test.zip"],
    ratios=[80, 10, 10],
    seed=42,
)

print(f"Common IDs: {len(result['common_ids'])}")
for src in result["per_source"]:
    print(f"\n{src['source']}:")
    print(f"  total: {src['total_ids']}, excluded: {src['excluded_ids']}")
    for s in src["splits"]:
        print(f"  {s['suffix']}: {s['num_ids']} IDs, {s['copied']} copied")
