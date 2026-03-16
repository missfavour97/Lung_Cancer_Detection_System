import os
import random
import shutil

random.seed(42)

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

classes = ["cancer", "no_cancer"]

for class_name in classes:
    source_folder = os.path.join(train_dir, class_name)
    images = os.listdir(source_folder)
    random.shuffle(images)

    total = len(images)
    val_count = int(total * 0.15)
    test_count = int(total * 0.15)

    val_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]

    for img in val_images:
        shutil.move(
            os.path.join(source_folder, img),
            os.path.join(val_dir, class_name, img)
        )

    for img in test_images:
        shutil.move(
            os.path.join(source_folder, img),
            os.path.join(test_dir, class_name, img)
        )

print("Dataset split complete!")