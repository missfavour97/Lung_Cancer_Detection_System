import os
import hashlib

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

def file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def collect_hashes(folder):
    hashes = {}
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith("."):
                continue
            path = os.path.join(root, file)
            try:
                h = file_hash(path)
                hashes[path] = h
            except Exception as e:
                print(f"Skipped {path}: {e}")
    return hashes

def remove_duplicates(reference_hashes, target_folder, target_name):
    removed = 0
    target_hashes = collect_hashes(target_folder)

    for path, h in target_hashes.items():
        if h in reference_hashes:
            print(f"Removing duplicate from {target_name}: {path}")
            os.remove(path)
            removed += 1

    return removed

# Collect hashes
train_hashes = collect_hashes(TRAIN_DIR)
train_hash_values = set(train_hashes.values())

# Remove from VAL anything duplicated in TRAIN
removed_val = remove_duplicates(train_hash_values, VAL_DIR, "VAL")

# Recollect VAL after cleanup
val_hashes = collect_hashes(VAL_DIR)
train_val_hash_values = set(train_hash_values) | set(val_hashes.values())

# Remove from TEST anything duplicated in TRAIN or VAL
removed_test = remove_duplicates(train_val_hash_values, TEST_DIR, "TEST")

print("\nCleanup complete.")
print(f"Removed from VAL: {removed_val}")
print(f"Removed from TEST: {removed_test}")