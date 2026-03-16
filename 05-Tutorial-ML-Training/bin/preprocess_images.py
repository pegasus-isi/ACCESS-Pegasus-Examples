#!/usr/bin/env python3

"""
Preprocess crop images for disease classification.

Performs:
1. Image resizing to standard dimensions
2. Normalization
3. Data augmentation (optional)
4. Train/validation split
5. Feature extraction preparation

Usage:
    ./preprocess_images.py --input crop_catalog.csv --output-dir processed/ \
        --image-size 224 --split 0.8
"""

import argparse
import json
import logging
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_catalog(catalog_file: str) -> pd.DataFrame:
    """Load image catalog CSV."""
    df = pd.read_csv(catalog_file)
    logger.info(f"Loaded {len(df)} image records")
    return df


def resize_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Load and resize image to target dimensions.

    Args:
        image_path: Path to image file
        target_size: (width, height) tuple

    Returns:
        Numpy array of shape (height, width, 3)
    """
    try:
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize with antialiasing
        img = img.resize(target_size, Image.LANCZOS)

        return np.array(img)

    except Exception as e:
        logger.warning(f"Failed to process {image_path}: {e}")
        return None


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.

    Uses ImageNet normalization for transfer learning compatibility.
    """
    # Convert to float32
    img = img.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std

    return img


def create_label_mapping(categories: List[str]) -> Tuple[Dict, Dict]:
    """
    Create mapping between category names and numeric labels.

    Returns:
        label_to_idx: category name -> index
        idx_to_label: index -> category name
    """
    unique_categories = sorted(set(categories))
    label_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    idx_to_label = {idx: cat for cat, idx in label_to_idx.items()}

    return label_to_idx, idx_to_label


def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Create augmented versions of image for training.

    Returns list of augmented images.
    """
    augmented = []

    # Original
    augmented.append(img)

    # Horizontal flip
    augmented.append(np.fliplr(img))

    # Small rotation simulation (by shifting)
    # Note: Using numpy operations to avoid PIL dependency in augmentation
    augmented.append(np.roll(img, 10, axis=1))  # Shift right
    augmented.append(np.roll(img, -10, axis=1))  # Shift left

    return augmented


def preprocess_dataset(
    catalog: pd.DataFrame,
    output_dir: str,
    image_size: int = 224,
    train_split: float = 0.8,
    augment: bool = True
) -> Dict:
    """
    Preprocess entire dataset.

    Args:
        catalog: DataFrame with image paths and labels
        output_dir: Directory for preprocessed data
        image_size: Target image dimension (square)
        train_split: Fraction for training set
        augment: Whether to apply data augmentation

    Returns:
        Dictionary with preprocessing statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create label mapping
    label_to_idx, idx_to_label = create_label_mapping(catalog['category'].tolist())

    # Shuffle and split
    catalog_shuffled = catalog.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(catalog_shuffled) * train_split)

    train_df = catalog_shuffled[:split_idx]
    val_df = catalog_shuffled[split_idx:]

    logger.info(f"Train set: {len(train_df)}, Validation set: {len(val_df)}")

    # Process images
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    stats = {
        'processed': 0,
        'failed': 0,
        'train_samples': 0,
        'val_samples': 0,
    }

    # Process training set
    for idx, row in train_df.iterrows():
        img_path = row['image_path']
        category = row['category']

        # Check if file exists (skip sample placeholders)
        if not Path(img_path).exists():
            if row.get('is_sample', False):
                # Create dummy data for sample mode
                img = np.random.rand(image_size, image_size, 3).astype(np.float32)
            else:
                stats['failed'] += 1
                continue
        else:
            img = resize_image(img_path, (image_size, image_size))
            if img is None:
                stats['failed'] += 1
                continue
            img = normalize_image(img)

        label = label_to_idx[category]

        if augment:
            augmented = augment_image(img)
            for aug_img in augmented:
                train_images.append(aug_img)
                train_labels.append(label)
                stats['train_samples'] += 1
        else:
            train_images.append(img)
            train_labels.append(label)
            stats['train_samples'] += 1

        stats['processed'] += 1

    # Process validation set (no augmentation)
    for idx, row in val_df.iterrows():
        img_path = row['image_path']
        category = row['category']

        if not Path(img_path).exists():
            if row.get('is_sample', False):
                img = np.random.rand(image_size, image_size, 3).astype(np.float32)
            else:
                stats['failed'] += 1
                continue
        else:
            img = resize_image(img_path, (image_size, image_size))
            if img is None:
                stats['failed'] += 1
                continue
            img = normalize_image(img)

        label = label_to_idx[category]
        val_images.append(img)
        val_labels.append(label)
        stats['val_samples'] += 1
        stats['processed'] += 1

    # Convert to numpy arrays
    if train_images:
        train_X = np.array(train_images)
        train_y = np.array(train_labels)
    else:
        train_X = np.array([])
        train_y = np.array([])

    if val_images:
        val_X = np.array(val_images)
        val_y = np.array(val_labels)
    else:
        val_X = np.array([])
        val_y = np.array([])

    # Save preprocessed data
    np.savez_compressed(
        output_path / 'train_data.npz',
        images=train_X,
        labels=train_y
    )

    np.savez_compressed(
        output_path / 'val_data.npz',
        images=val_X,
        labels=val_y
    )

    # Save label mapping
    with open(output_path / 'label_mapping.json', 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {str(k): v for k, v in idx_to_label.items()},
            'num_classes': len(label_to_idx),
        }, f, indent=2)

    # Save preprocessing info
    preprocessing_info = {
        'image_size': image_size,
        'train_split': train_split,
        'augmentation': augment,
        'normalization': 'imagenet',
        'stats': stats,
        'train_shape': list(train_X.shape) if len(train_X) > 0 else [],
        'val_shape': list(val_X.shape) if len(val_X) > 0 else [],
    }

    with open(output_path / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)

    logger.info(f"Preprocessing complete:")
    logger.info(f"  Processed: {stats['processed']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Train samples: {stats['train_samples']}")
    logger.info(f"  Val samples: {stats['val_samples']}")

    return preprocessing_info


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess crop images for disease classification"
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input catalog CSV file'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for preprocessed data'
    )

    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Target image size (square)'
    )

    parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Training set fraction (0-1)'
    )

    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )

    parser.add_argument(
        '--images-archive',
        type=str,
        default=None,
        help='Optional tar.gz archive containing images directory'
    )

    args = parser.parse_args()

    if args.images_archive:
        if not os.path.exists(args.images_archive):
            logger.error(f"Images archive not found: {args.images_archive}")
            sys.exit(1)
        with tarfile.open(args.images_archive, "r:gz") as tar:
            tar.extractall(path=".")
        logger.info(f"Extracted images archive: {args.images_archive}")

    catalog = load_catalog(args.input)
    preprocess_dataset(
        catalog,
        args.output_dir,
        image_size=args.image_size,
        train_split=args.split,
        augment=not args.no_augment
    )


if __name__ == "__main__":
    main()
