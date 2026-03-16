#!/usr/bin/env python3

"""
Train CNN classifier for crop disease detection.

Uses transfer learning with pretrained models (ResNet, EfficientNet)
for efficient training on plant disease images.

Usage:
    ./train_classifier.py --input-dir processed/ --output-dir models/ \
        --epochs 20 --batch-size 32
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using sklearn fallback.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SimpleCNN(nn.Module):
    """Simple CNN for crop disease classification."""

    def __init__(self, num_classes: int, input_channels: int = 3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_preprocessed_data(input_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load preprocessed training and validation data."""
    input_path = Path(input_dir)

    train_data = np.load(input_path / 'train_data.npz')
    val_data = np.load(input_path / 'val_data.npz')

    with open(input_path / 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)

    train_X = train_data['images']
    train_y = train_data['labels']
    val_X = val_data['images']
    val_y = val_data['labels']

    logger.info(f"Loaded train: {train_X.shape}, val: {val_X.shape}")
    logger.info(f"Classes: {label_mapping['num_classes']}")

    return train_X, train_y, val_X, val_y, label_mapping


def train_pytorch_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'auto'
) -> Tuple[nn.Module, Dict]:
    """
    Train PyTorch CNN model.

    Returns:
        Trained model and training history
    """
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Convert to PyTorch tensors (channels first: NCHW)
    train_X_t = torch.FloatTensor(train_X).permute(0, 3, 1, 2)
    train_y_t = torch.LongTensor(train_y)
    val_X_t = torch.FloatTensor(val_X).permute(0, 3, 1, 2)
    val_y_t = torch.LongTensor(val_y)

    # Create data loaders
    train_dataset = TensorDataset(train_X_t, train_y_t)
    val_dataset = TensorDataset(val_X_t, val_y_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history


def train_sklearn_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> Tuple[object, Dict]:
    """
    Train sklearn Random Forest as fallback.

    Returns:
        Trained model and metrics
    """
    logger.info("Training sklearn Random Forest classifier...")

    # Flatten images for sklearn
    train_X_flat = train_X.reshape(train_X.shape[0], -1)
    val_X_flat = val_X.reshape(val_X.shape[0], -1)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
    model.fit(train_X_flat, train_y)

    # Evaluate
    train_pred = model.predict(train_X_flat)
    val_pred = model.predict(val_X_flat)

    train_acc = accuracy_score(train_y, train_pred)
    val_acc = accuracy_score(val_y, val_pred)

    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    history = {
        'train_acc': [train_acc],
        'val_acc': [val_acc],
        'model_type': 'RandomForest',
    }

    return model, history


def save_model(model, history: Dict, label_mapping: Dict, output_dir: str, use_pytorch: bool):
    """Save trained model and training info."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_pytorch and TORCH_AVAILABLE:
        # Save PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': label_mapping['num_classes'],
        }, output_path / 'disease_classifier.pt')
    else:
        # Save sklearn model with joblib
        import joblib
        joblib.dump(model, output_path / 'disease_classifier.joblib')

    # Save training info
    training_info = {
        'trained_at': datetime.now().isoformat(),
        'framework': 'pytorch' if (use_pytorch and TORCH_AVAILABLE) else 'sklearn',
        'num_classes': label_mapping['num_classes'],
        'label_mapping': label_mapping,
        'history': history,
        'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
        'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
    }

    with open(output_path / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)

    logger.info(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train crop disease classifier"
    )

    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory with preprocessed data'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for model'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for training'
    )

    parser.add_argument(
        '--use-sklearn',
        action='store_true',
        help='Use sklearn instead of PyTorch'
    )

    args = parser.parse_args()

    # Load data
    train_X, train_y, val_X, val_y, label_mapping = load_preprocessed_data(args.input_dir)

    if len(train_X) == 0:
        logger.error("No training data available")
        sys.exit(1)

    num_classes = label_mapping['num_classes']

    # Train model
    use_pytorch = TORCH_AVAILABLE and not args.use_sklearn

    if use_pytorch:
        model, history = train_pytorch_model(
            train_X, train_y, val_X, val_y,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device
        )
    elif SKLEARN_AVAILABLE:
        model, history = train_sklearn_model(train_X, train_y, val_X, val_y)
    else:
        logger.error("Neither PyTorch nor sklearn available")
        sys.exit(1)

    # Save model
    save_model(model, history, label_mapping, args.output_dir, use_pytorch)


if __name__ == "__main__":
    main()
