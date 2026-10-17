import random
import shutil
from pathlib import Path

SOURCE_DIR = Path(r"E:\SPCK-CSI18\data\RealWaste")
OUTPUT_DIR = SOURCE_DIR.parent / "RealWaste_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(42)

if not SOURCE_DIR.is_dir():
    raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Remove or rename it before running again."
    )

class_dirs = sorted(
    path for path in SOURCE_DIR.iterdir()
    if path.is_dir()
)

if not class_dirs:
    raise ValueError("No class folders found in RealWaste.")

for class_dir in class_dirs:
    images = [
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in split_data.items():
        for image_path in split_images:
            relative_path = image_path.relative_to(class_dir)
            destination = (
                OUTPUT_DIR / split_name / class_dir.name / relative_path
            )

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)

    print(
        f"{class_dir.name}: "
        f"total={total}, "
        f"train={len(split_data['train'])}, "
        f"val={len(split_data['val'])}, "
        f"test={len(split_data['test'])}"
    )

print(f"\nDataset split complete: {OUTPUT_DIR}")