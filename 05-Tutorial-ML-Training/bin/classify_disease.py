#!/usr/bin/env python3

"""
Classify crop diseases from images using trained model.

Performs inference on new images and outputs disease predictions
with confidence scores and treatment recommendations.

Usage:
    ./classify_disease.py --model-dir models/ --input images/ \
        --output predictions.json
"""

import argparse
import json
import logging
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# Disease treatment recommendations
TREATMENT_RECOMMENDATIONS = {
    'Apple___Apple_scab': {
        'severity': 'moderate',
        'action': 'Apply fungicide (captan, mancozeb)',
        'prevention': 'Remove fallen leaves, improve air circulation',
        'fertilizer': 'Balanced NPK after treatment',
    },
    'Apple___Black_rot': {
        'severity': 'high',
        'action': 'Remove infected fruit and branches',
        'prevention': 'Prune for airflow, remove mummified fruit',
        'fertilizer': 'Potassium-rich fertilizer for disease resistance',
    },
    'Apple___healthy': {
        'severity': 'none',
        'action': 'No treatment needed',
        'prevention': 'Continue regular monitoring',
        'fertilizer': 'Standard balanced fertilizer program',
    },
    'Corn_(maize)___Common_rust': {
        'severity': 'moderate',
        'action': 'Apply fungicide if severe',
        'prevention': 'Plant resistant varieties',
        'fertilizer': 'Adequate nitrogen for healthy growth',
    },
    'Corn_(maize)___healthy': {
        'severity': 'none',
        'action': 'No treatment needed',
        'prevention': 'Rotate crops, monitor regularly',
        'fertilizer': 'Nitrogen-rich fertilizer during growth',
    },
    'Tomato___Early_blight': {
        'severity': 'moderate',
        'action': 'Apply copper-based fungicide',
        'prevention': 'Mulch around plants, avoid overhead watering',
        'fertilizer': 'Calcium and potassium to strengthen plants',
    },
    'Tomato___Late_blight': {
        'severity': 'critical',
        'action': 'Remove and destroy infected plants immediately',
        'prevention': 'Avoid wet conditions, use resistant varieties',
        'fertilizer': 'Hold fertilizer until healthy, then balanced NPK',
    },
    'Tomato___healthy': {
        'severity': 'none',
        'action': 'No treatment needed',
        'prevention': 'Maintain good cultural practices',
        'fertilizer': 'Balanced fertilizer with calcium for fruit set',
    },
    'Potato___Early_blight': {
        'severity': 'moderate',
        'action': 'Apply chlorothalonil or copper fungicide',
        'prevention': 'Rotate crops, destroy infected debris',
        'fertilizer': 'Potassium to improve disease tolerance',
    },
    'Potato___Late_blight': {
        'severity': 'critical',
        'action': 'Immediate fungicide application, destroy severely infected plants',
        'prevention': 'Use certified seed, avoid overhead irrigation',
        'fertilizer': 'Phosphorus for root development after recovery',
    },
    'Potato___healthy': {
        'severity': 'none',
        'action': 'No treatment needed',
        'prevention': 'Continue monitoring, hill properly',
        'fertilizer': 'High potassium fertilizer for tuber development',
    },
    'Grape___Black_rot': {
        'severity': 'high',
        'action': 'Apply mancozeb or myclobutanil fungicide',
        'prevention': 'Remove mummified fruit, prune for airflow',
        'fertilizer': 'Balanced fertilizer, avoid excess nitrogen',
    },
    'Grape___healthy': {
        'severity': 'none',
        'action': 'No treatment needed',
        'prevention': 'Regular canopy management',
        'fertilizer': 'Moderate nitrogen, adequate potassium',
    },
}


class SimpleCNN(nn.Module):
    """Simple CNN for crop disease classification (must match training)."""

    def __init__(self, num_classes: int, input_channels: int = 3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
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


def load_model(model_dir: str) -> Tuple[object, Dict, str]:
    """
    Load trained model and label mapping.

    Returns:
        model, label_mapping, framework
    """
    model_path = Path(model_dir)

    # Load training info
    with open(model_path / 'training_info.json', 'r') as f:
        training_info = json.load(f)

    label_mapping = training_info['label_mapping']
    framework = training_info['framework']

    if framework == 'pytorch' and TORCH_AVAILABLE:
        # Load PyTorch model
        checkpoint = torch.load(
            model_path / 'disease_classifier.pt',
            map_location='cpu'
        )
        num_classes = checkpoint['num_classes']

        model = SimpleCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        logger.info(f"Loaded PyTorch model with {num_classes} classes")

    elif framework == 'sklearn' and JOBLIB_AVAILABLE:
        model = joblib.load(model_path / 'disease_classifier.joblib')
        logger.info("Loaded sklearn model")

    else:
        raise RuntimeError(f"Cannot load {framework} model - dependencies not available")

    return model, label_mapping, framework


def preprocess_image(image_path: str, image_size: int = 224) -> np.ndarray:
    """Preprocess single image for inference."""
    try:
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((image_size, image_size), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        return img_array

    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return None


def classify_image_pytorch(
    model: nn.Module,
    image: np.ndarray,
    label_mapping: Dict
) -> Dict:
    """Classify image using PyTorch model."""
    # Convert to tensor (add batch dimension, channels first)
    img_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_idx = predicted.item()
    predicted_class = label_mapping['idx_to_label'][str(predicted_idx)]
    confidence_score = confidence.item()

    # Get top 3 predictions
    top_probs, top_indices = probabilities.topk(min(3, len(label_mapping['idx_to_label'])))
    top_predictions = [
        {
            'class': label_mapping['idx_to_label'][str(idx.item())],
            'confidence': prob.item()
        }
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]

    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'top_predictions': top_predictions,
    }


def classify_image_sklearn(
    model,
    image: np.ndarray,
    label_mapping: Dict
) -> Dict:
    """Classify image using sklearn model."""
    # Flatten image
    img_flat = image.reshape(1, -1)

    # Predict
    predicted_idx = model.predict(img_flat)[0]
    predicted_class = label_mapping['idx_to_label'][str(predicted_idx)]

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(img_flat)[0]
        confidence = probabilities[predicted_idx]

        # Top predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'class': label_mapping['idx_to_label'][str(idx)],
                'confidence': probabilities[idx]
            }
            for idx in top_indices
        ]
    else:
        confidence = 1.0
        top_predictions = [{'class': predicted_class, 'confidence': 1.0}]

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_predictions': top_predictions,
    }


def get_treatment_recommendation(predicted_class: str) -> Dict:
    """Get treatment recommendation for predicted disease."""
    # Try exact match
    if predicted_class in TREATMENT_RECOMMENDATIONS:
        return TREATMENT_RECOMMENDATIONS[predicted_class]

    # Try partial match
    for key, value in TREATMENT_RECOMMENDATIONS.items():
        if predicted_class.lower() in key.lower() or key.lower() in predicted_class.lower():
            return value

    # Default recommendation
    return {
        'severity': 'unknown',
        'action': 'Consult local agricultural extension service',
        'prevention': 'Maintain good cultural practices',
        'fertilizer': 'Standard balanced fertilizer',
    }


def classify_images(
    model,
    image_paths: List[str],
    label_mapping: Dict,
    framework: str
) -> List[Dict]:
    """Classify multiple images."""
    results = []

    for img_path in image_paths:
        logger.info(f"Classifying: {img_path}")

        image = preprocess_image(img_path)
        if image is None:
            results.append({
                'image_path': img_path,
                'error': 'Failed to preprocess image',
            })
            continue

        # Classify
        if framework == 'pytorch':
            prediction = classify_image_pytorch(model, image, label_mapping)
        else:
            prediction = classify_image_sklearn(model, image, label_mapping)

        # Extract disease info from class name
        predicted_class = prediction['predicted_class']
        parts = predicted_class.split('___')
        crop = parts[0].replace('_', ' ') if len(parts) > 0 else 'Unknown'
        disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
        is_healthy = 'healthy' in predicted_class.lower()

        # Get treatment
        treatment = get_treatment_recommendation(predicted_class)

        results.append({
            'image_path': img_path,
            'filename': Path(img_path).name,
            'predicted_class': predicted_class,
            'crop': crop,
            'disease': disease,
            'is_healthy': is_healthy,
            'confidence': round(prediction['confidence'], 4),
            'top_predictions': prediction['top_predictions'],
            'treatment': treatment,
        })

    return results


def run_inference(model_dir: str, input_path: str, output_file: str):
    """Run inference on input images."""
    # Load model
    model, label_mapping, framework = load_model(model_dir)

    # Find images
    input_path = Path(input_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = [
            str(f) for f in input_path.rglob('*')
            if f.suffix.lower() in image_extensions
        ]
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return

    logger.info(f"Found {len(image_paths)} images to classify")

    # Classify images
    results = classify_images(model, image_paths, label_mapping, framework)

    # Create output
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'model_dir': str(model_dir),
            'framework': framework,
            'total_images': len(results),
        },
        'predictions': results,
        'summary': {
            'total': len(results),
            'healthy': sum(1 for r in results if r.get('is_healthy', False)),
            'diseased': sum(1 for r in results if not r.get('is_healthy', True) and 'error' not in r),
            'errors': sum(1 for r in results if 'error' in r),
        }
    }

    # Disease breakdown
    disease_counts = {}
    for r in results:
        if 'error' not in r:
            disease = r.get('disease', 'Unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    output['summary']['disease_breakdown'] = disease_counts

    # Critical alerts
    critical = [r for r in results if r.get('treatment', {}).get('severity') == 'critical']
    if critical:
        output['critical_alerts'] = [
            {
                'image': r['filename'],
                'disease': r['disease'],
                'action': r['treatment']['action'],
            }
            for r in critical
        ]

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Predictions saved to {output_file}")
    logger.info(f"Summary: {output['summary']['healthy']} healthy, "
               f"{output['summary']['diseased']} diseased, "
               f"{output['summary']['errors']} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Classify crop diseases from images"
    )

    parser.add_argument(
        '--model-dir', '-m',
        type=str,
        required=True,
        help='Directory with trained model'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image file or directory'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output predictions JSON file'
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

    run_inference(args.model_dir, args.input, args.output)


if __name__ == "__main__":
    main()
