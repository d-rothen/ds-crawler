"""Split the RDS clear/hazy datasets into train/val with aligned IDs."""

from ds_crawler import split_datasets

rds_clear = "/path/to/rds_clear"
rds_hazy = "/path/to/rds_hazy"

result = split_datasets(
    source_paths=[rds_clear, rds_hazy],
    suffixes=["train", "val"],
    ratios=[80, 20],
    seed=42,
)

print(f"Common IDs: {len(result['common_ids'])}")
for src in result["per_source"]:
    print(f"\n{src['source']}:")
    print(f"  total: {src['total_ids']}, excluded: {src['excluded_ids']}")
    for s in src["splits"]:
        print(f"  {s['suffix']}: {s['num_ids']} IDs, {s['copied']} copied")
