from pathlib import Path
import random
import shutil

SOURCE = Path(r"E:\SPCK-CSI18\data\standardized_256")
DEST = Path(__file__).parent / "dataset"

random.seed(42)

for class_dir in SOURCE.iterdir():

    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.jpg"))

    random.shuffle(images)

    n = len(images)

    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]:

        target = DEST / split / class_dir.name
        target.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, target / image.name)

print("Đã chia dataset!")