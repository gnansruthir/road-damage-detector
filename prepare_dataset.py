import os
import zipfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil
import argparse

# Configure paths
DATA_DIR = Path("data").resolve()
ZIP_URL = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip"
ZIP_PATH = DATA_DIR / "RDD2022_India.zip"

CLASS_MAP = {
    "D40": 0,  # Pothole
    "D00": 1,  # Longitudinal Crack
    "D10": 2,  # Transverse Crack
}

def download_dataset():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {ZIP_URL}...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Zip file already exists. Skipping download.")

def extract_dataset():
    extracted_dir = DATA_DIR / "RDD2022_India"
    if not extracted_dir.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

def convert_voc_to_yolo(xml_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_boxes = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Calculate YOLO coordinates
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_boxes

def get_image_brightness(image_path):
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())

def _image_files(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _copy_yolo_split(image_files, source_labels_dir, destination_split, dry_run=False):
    records = []
    for image_path in image_files:
        label_path = Path(source_labels_dir) / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        labels = [line for line in labels if line.split()[0].isdigit() and int(line.split()[0]) in range(3)]
        if not labels:
            continue
        records.append((image_path, labels, get_image_brightness(image_path)))
        if not dry_run:
            shutil.copy(image_path, DATA_DIR / "images" / destination_split / image_path.name)
            (DATA_DIR / "labels" / destination_split / label_path.name).write_text(
                "\n".join(labels), encoding="utf-8"
            )
    return records


def _report_class_balance(records):
    counts = [0, 0, 0]
    for _, labels, _ in records:
        for label in labels:
            class_id = int(label.split()[0])
            if class_id < len(counts):
                counts[class_id] += 1
    names = ["Pothole", "Longitudinal Crack", "Transverse Crack"]
    print("Class instances: " + ", ".join(f"{name}={count}" for name, count in zip(names, counts)))
    nonzero = [count for count in counts if count]
    if nonzero and min(nonzero) * 10 < max(nonzero):
        print("WARNING: a class has fewer than 10% of the instances of the largest class.")


def prepare_yolo_splits(source_dir, limit=None, dry_run=False):
    source_dir = Path(source_dir)
    train_images = source_dir / "train" / "images"
    train_labels = source_dir / "train" / "labels"
    val_images = source_dir / "valid" / "images"
    val_labels = source_dir / "valid" / "labels"
    if not val_images.exists():
        val_images = source_dir / "val" / "images"
        val_labels = source_dir / "val" / "labels"
    if not train_images.exists() or not train_labels.exists() or not val_images.exists() or not val_labels.exists():
        raise FileNotFoundError("Roboflow export must contain train/images, train/labels, and valid/images, valid/labels.")

    if limit is not None:
        train_files = _image_files(train_images)[:limit]
        val_files = _image_files(val_images)[:limit]
    else:
        train_files = _image_files(train_images)
        val_files = _image_files(val_images)
    train_records = _copy_yolo_split(train_files, train_labels, "train", dry_run)
    val_records = _copy_yolo_split(val_files, val_labels, "val", dry_run)
    _report_class_balance(train_records + val_records)
    val_records.sort(key=lambda record: record[2])
    night_records = val_records[:150]
    if not dry_run:
        for image_path, labels, _ in night_records:
            shutil.copy(image_path, DATA_DIR / "images" / "val_night" / image_path.name)
            (DATA_DIR / "labels" / "val_night" / f"{image_path.stem}.txt").write_text(
                "\n".join(labels), encoding="utf-8"
            )
    print(f"Roboflow YOLO split: train={len(train_records)}, val={len(val_records)}, val_night={len(night_records)}")


def prepare_splits(limit=None, dry_run=False, source_dir=None):
    # Setup directories
    if not dry_run:
        for split in ["train", "val", "val_night"]:
            (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    if source_dir is not None:
        prepare_yolo_splits(source_dir, limit=limit, dry_run=dry_run)
        return

    src_images_dir = DATA_DIR / "India" / "train" / "images"
    src_xmls_dir = DATA_DIR / "India" / "train" / "annotations" / "xmls"
    
    if not src_images_dir.exists():
        # Sometimes structure is India/train/images or India/India/train/images depending on extraction
        alternatives = list(DATA_DIR.glob("**/train/images"))
        if alternatives:
            src_images_dir = alternatives[0]
            src_xmls_dir = src_images_dir.parent / "annotations" / "xmls"
            
    print(f"Source images directory: {src_images_dir}")
    print(f"Source XMLs directory: {src_xmls_dir}")
    
    image_files = sorted(
        path for path in src_images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    random.seed(42)
    random.shuffle(image_files)
    if limit is not None:
        image_files = image_files[:limit]
    
    # Train/Val split (85% train, 15% val)
    split_idx = int(len(image_files) * 0.85)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total train images: {len(train_files)}")
    print(f"Total validation images: {len(val_files)}")
    
    # Convert and copy train files
    print("Processing training split...")
    for img_path in train_files:
        xml_path = src_xmls_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue
        
        # Get image size safely with PIL
        from PIL import Image
        with Image.open(img_path) as img:
            w, h = img.size
            
        boxes = convert_voc_to_yolo(xml_path, w, h)
        if not boxes:  # Skip images without targets of interest
            continue
            
        if dry_run:
            continue

        # Copy image
        shutil.copy(img_path, DATA_DIR / "images" / "train" / img_path.name)
        # Save labels
        label_path = DATA_DIR / "labels" / "train" / (img_path.stem + ".txt")
        label_path.write_text("\n".join(boxes), encoding="utf-8")
        
    # Convert and copy val files
    print("Processing validation split...")
    val_brightness_list = []
    for img_path in val_files:
        xml_path = src_xmls_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue
            
        from PIL import Image
        with Image.open(img_path) as img:
            w, h = img.size
            
        boxes = convert_voc_to_yolo(xml_path, w, h)
        if not boxes:
            continue
            
        if dry_run:
            val_brightness_list.append((img_path, boxes, get_image_brightness(img_path)))
            continue

        dest_img_path = DATA_DIR / "images" / "val" / img_path.name
        shutil.copy(img_path, dest_img_path)
        
        label_path = DATA_DIR / "labels" / "val" / (img_path.stem + ".txt")
        label_path.write_text("\n".join(boxes), encoding="utf-8")
        
        # Measure brightness for night split
        brightness = get_image_brightness(dest_img_path)
        val_brightness_list.append((img_path, boxes, brightness))
        
    # Build val_night split (take darkest 150 images)
    print("Creating night-mode validation split...")
    val_brightness_list.sort(key=lambda x: x[2])
    night_candidates = val_brightness_list[:150]

    if dry_run:
        print(f"Dry run: would create {len(night_candidates)} night validation images.")
        return
    
    for img_path, boxes, brightness in night_candidates:
        # Copy to val_night
        shutil.copy(img_path, DATA_DIR / "images" / "val_night" / img_path.name)
        label_path = DATA_DIR / "labels" / "val_night" / (img_path.stem + ".txt")
        label_path.write_text("\n".join(boxes), encoding="utf-8")
    
    darkest = f" Darkest brightness mean: {night_candidates[0][2]:.2f}." if night_candidates else ""
    print(f"val_night split created with {len(night_candidates)} images.{darkest}")

def create_yaml_configs(config_dir=Path(".")):
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    # Main YAML config
    rdd2022_yaml = f"""# RDD2022 India Dataset Configuration
path: {DATA_DIR}
train: images/train
val: images/val
names:
  0: Pothole
  1: Longitudinal Crack
  2: Transverse Crack
"""
    (config_dir / "rdd2022.yaml").write_text(rdd2022_yaml, encoding="utf-8")
    print(f"Created {config_dir / 'rdd2022.yaml'}")

    # Night YAML config
    rdd2022_night_yaml = f"""# RDD2022 India Night Validation Configuration
path: {DATA_DIR}
train: images/train
val: images/val_night
names:
  0: Pothole
  1: Longitudinal Crack
  2: Transverse Crack
"""
    (config_dir / "rdd2022_night.yaml").write_text(rdd2022_night_yaml, encoding="utf-8")
    print(f"Created {config_dir / 'rdd2022_night.yaml'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RDD2022 India annotations for Ultralytics YOLO.")
    parser.add_argument("--limit", type=int, help="Process at most this many images (useful for verification).")
    parser.add_argument("--source-dir", help="Existing Roboflow YOLO export directory, such as data/rdd2022.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and convert annotations without writing split files or YAML.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.source_dir:
        if not Path(args.source_dir).exists():
            raise FileNotFoundError(f"Source directory not found: {args.source_dir}")
    elif args.dry_run:
        if not (DATA_DIR / "India").exists() and not list(DATA_DIR.glob("**/train/images")):
            raise FileNotFoundError("Dry run requires an already extracted dataset under data/.")
    else:
        download_dataset()
        extract_dataset()
    prepare_splits(limit=args.limit, dry_run=args.dry_run, source_dir=args.source_dir)
    if not args.dry_run:
        create_yaml_configs(args.source_dir or Path("."))
    print("All preparation completed successfully!")
