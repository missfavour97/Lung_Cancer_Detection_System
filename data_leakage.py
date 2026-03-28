import os
import hashlib

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

def find_duplicates(hash_dict_1, hash_dict_2, name1, name2):
    reverse_1 = {}
    reverse_2 = {}

    for path, h in hash_dict_1.items():
        reverse_1.setdefault(h, []).append(path)

    for path, h in hash_dict_2.items():
        reverse_2.setdefault(h, []).append(path)

    common_hashes = set(reverse_1.keys()) & set(reverse_2.keys())

    if not common_hashes:
        print(f"\n✅ No duplicates between {name1} and {name2}")
        return

    print(f"\n⚠️ Duplicates found between {name1} and {name2}:")
    for h in common_hashes:
        for p1 in reverse_1[h]:
            print(f"{name1}: {p1}")
        for p2 in reverse_2[h]:
            print(f"{name2}: {p2}")
        print("-" * 50)

train_hashes = collect_hashes("dataset/train")
val_hashes = collect_hashes("dataset/val")
test_hashes = collect_hashes("dataset/test")

find_duplicates(train_hashes, val_hashes, "TRAIN", "VAL")
find_duplicates(train_hashes, test_hashes, "TRAIN", "TEST")
find_duplicates(val_hashes, test_hashes, "VAL", "TEST")