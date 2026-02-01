import json
import logging

from ds_crawler.parser import copy_dataset, index_dataset_from_path

logging.basicConfig(level=logging.INFO)

test_kitti = "/Volumes/Volume/Datasets/vkitti2/test_kitti"

base_index = "/Volumes/Volume/Datasets/vkitti2/hazy_depth_06/output.json"

copy_target = "/Volumes/Volume/Datasets/vkitti2/test_kitti_(copy)"

# 1) Index test_kitti with sampling, save the output.json
result1 = index_dataset_from_path(
    test_kitti,
    strict=False,
    save_index=True,
)
print("=== Result 1 (test_kitti) ===")
print(json.dumps(result1, indent=2)[:500])

# 2) Load output.json from test, then index test_kitti matching against it
with open(base_index) as f:
    match_index = json.load(f)

result2 = index_dataset_from_path(
    test_kitti,
    strict=False,
    save_index=True,
    match_index=match_index
)
print("\n=== Result 2 (test_kitti with match_index) ===")
print(json.dumps(result2, indent=2)[:500])

# # 3) Copy kitti dataset with sampling
result3 = copy_dataset(
    test_kitti,
    copy_target,
    sample=1,
)
print("\n=== Result 3 (copy_dataset) ===")
print(json.dumps(result3, indent=2))
