import os
from PIL import Image

DATASET_DIR = "dataset"

def list_images(path):
    imgs = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(root, f))
    return imgs

def debug_dataset():
    print("\n====== DATASET DEBUG REPORT ======\n")

    for split in ["train", "test"]:
        print(f"\n📁 Checking '{split}' folder...\n")

        split_path = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_path):
            print(f" Missing folder: {split_path}")
            continue

        for class_name in ["real", "deepfake"]:
            class_path = os.path.join(split_path, class_name)
            print(f" ▶ Class: {class_name}")

            if not os.path.exists(class_path):
                print(f"   ❌ Missing: {class_path}")
                continue

            images = list_images(class_path)
            print(f"   - Found {len(images)} image(s)")

            # Try reading the first image
            if images:
                try:
                    img = Image.open(images[0])
                    img.verify()  # Check for corruption
                    img = Image.open(images[0])  # reopen fully
                    print(f"   - Example: {images[0]}")
                    print(f"   - Image size: {img.size}")
                    print(f"   - Image mode: {img.mode}")
                except Exception as e:
                    print(f" Error opening image '{images[0]}': {e}")

    print("\n====== END OF REPORT ======\n")


if __name__ == "__main__":
    debug_dataset()
