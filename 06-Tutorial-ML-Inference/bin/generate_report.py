#!/usr/bin/env python3

"""
Generate crop health report with visualizations.

Creates a comprehensive report including:
1. Disease detection summary
2. Treatment recommendations
3. Fertilizer guidance
4. Visualization of results

Usage:
    ./generate_report.py --predictions predictions.json \
        --output-dir reports/ --format all
"""

import argparse
import json
import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Severity colors
SEVERITY_COLORS = {
    'critical': '#d62728',  # Red
    'high': '#ff7f0e',      # Orange
    'moderate': '#ffbb78',  # Light orange
    'low': '#98df8a',       # Light green
    'none': '#2ca02c',      # Green
    'unknown': '#7f7f7f',   # Gray
}


def load_predictions(predictions_file: str) -> Dict:
    """Load predictions JSON file."""
    with open(predictions_file, 'r') as f:
        return json.load(f)


def create_disease_distribution_chart(predictions: List[Dict], output_path: Path):
    """Create pie chart of disease distribution."""
    # Count diseases
    disease_counts = {}
    for pred in predictions:
        if 'error' not in pred:
            disease = pred.get('disease', 'Unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

    if not disease_counts:
        logger.warning("No disease data to plot")
        return

    # Sort by count
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [d[0] for d in sorted_diseases]
    sizes = [d[1] for d in sorted_diseases]

    # Colors - green for healthy, other colors for diseases
    colors = []
    for label in labels:
        if 'healthy' in label.lower():
            colors.append(SEVERITY_COLORS['none'])
        elif any(x in label.lower() for x in ['blight', 'rot', 'virus']):
            colors.append(SEVERITY_COLORS['critical'])
        elif any(x in label.lower() for x in ['rust', 'spot', 'mold']):
            colors.append(SEVERITY_COLORS['moderate'])
        else:
            colors.append(SEVERITY_COLORS['unknown'])

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.8,
    )

    ax.set_title('Disease Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / 'disease_distribution.png', dpi=150)
    plt.close()

    logger.info(f"Saved disease distribution chart")


def create_severity_chart(predictions: List[Dict], output_path: Path):
    """Create bar chart of severity levels."""
    severity_counts = {
        'critical': 0,
        'high': 0,
        'moderate': 0,
        'low': 0,
        'none': 0,
    }

    for pred in predictions:
        if 'error' not in pred:
            severity = pred.get('treatment', {}).get('severity', 'unknown')
            if severity in severity_counts:
                severity_counts[severity] += 1

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = list(severity_counts.keys())
    values = list(severity_counts.values())
    colors = [SEVERITY_COLORS[cat] for cat in categories]

    bars = ax.bar(categories, values, color=colors, edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Severity Level', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Disease Severity Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 10)

    plt.tight_layout()
    plt.savefig(output_path / 'severity_distribution.png', dpi=150)
    plt.close()

    logger.info(f"Saved severity distribution chart")


def create_crop_health_summary(predictions: List[Dict], output_path: Path):
    """Create crop-wise health summary chart."""
    crop_health = {}

    for pred in predictions:
        if 'error' not in pred:
            crop = pred.get('crop', 'Unknown')
            is_healthy = pred.get('is_healthy', False)

            if crop not in crop_health:
                crop_health[crop] = {'healthy': 0, 'diseased': 0}

            if is_healthy:
                crop_health[crop]['healthy'] += 1
            else:
                crop_health[crop]['diseased'] += 1

    if not crop_health:
        return

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    crops = list(crop_health.keys())
    healthy = [crop_health[c]['healthy'] for c in crops]
    diseased = [crop_health[c]['diseased'] for c in crops]

    x = np.arange(len(crops))
    width = 0.6

    bars1 = ax.bar(x, healthy, width, label='Healthy', color=SEVERITY_COLORS['none'])
    bars2 = ax.bar(x, diseased, width, bottom=healthy, label='Diseased',
                  color=SEVERITY_COLORS['moderate'])

    ax.set_xlabel('Crop Type', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Crop Health Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(crops, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'crop_health_summary.png', dpi=150)
    plt.close()

    logger.info(f"Saved crop health summary chart")


def create_confidence_histogram(predictions: List[Dict], output_path: Path):
    """Create histogram of prediction confidence scores."""
    confidences = [
        pred.get('confidence', 0)
        for pred in predictions
        if 'error' not in pred
    ]

    if not confidences:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(confidences), color='red', linestyle='--',
              label=f'Mean: {np.mean(confidences):.2f}')

    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'confidence_histogram.png', dpi=150)
    plt.close()

    logger.info(f"Saved confidence histogram")


def create_confusion_matrix_chart(accuracy_data: Dict, output_path: Path):
    """Create confusion matrix heatmap chart."""
    cm_data = accuracy_data.get('confusion_matrix', {})
    labels = cm_data.get('labels', [])
    matrix = cm_data.get('matrix', [])

    if not labels or not matrix:
        logger.warning("No confusion matrix data to plot")
        return

    matrix_np = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))

    im = ax.imshow(matrix_np, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Shorten labels for display
    short_labels = [l.replace('__', '\n') if len(l) > 20 else l for l in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    # Add text annotations
    thresh = matrix_np.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(matrix_np[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        color='white' if matrix_np[i, j] > thresh else 'black',
                        fontsize=7)

    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=150)
    plt.close()

    logger.info("Saved confusion matrix chart")


def generate_html_report(data: Dict, output_path: Path, accuracy_data: Dict = None):
    """Generate HTML report."""
    predictions = data.get('predictions', [])
    summary = data.get('summary', {})
    critical_alerts = data.get('critical_alerts', [])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Crop Health Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #2e7d32; border-bottom: 2px solid #2e7d32; padding-bottom: 10px; }}
        h2 {{ color: #1565c0; }}
        .summary-box {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .alert-box {{ background: #ffebee; border-left: 4px solid #d32f2f; padding: 15px; margin: 10px 0; }}
        .healthy {{ color: #2e7d32; font-weight: bold; }}
        .diseased {{ color: #d32f2f; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #1565c0; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
        .chart {{ margin: 10px; text-align: center; }}
        .treatment-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .severity-critical {{ border-left: 4px solid #d32f2f; }}
        .severity-high {{ border-left: 4px solid #ff9800; }}
        .severity-moderate {{ border-left: 4px solid #ffc107; }}
        .severity-none {{ border-left: 4px solid #4caf50; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Crop Health Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary-box">
            <h2>Summary</h2>
            <p><strong>Total Images Analyzed:</strong> {summary.get('total', 0)}</p>
            <p><strong>Healthy:</strong> <span class="healthy">{summary.get('healthy', 0)}</span></p>
            <p><strong>Diseased:</strong> <span class="diseased">{summary.get('diseased', 0)}</span></p>
            <p><strong>Errors:</strong> {summary.get('errors', 0)}</p>
        </div>
"""

    # Critical alerts
    if critical_alerts:
        html += """
        <h2>Critical Alerts</h2>
"""
        for alert in critical_alerts:
            html += f"""
        <div class="alert-box">
            <strong>{alert['image']}</strong>: {alert['disease']}<br>
            <em>Action Required:</em> {alert['action']}
        </div>
"""

    # Disease breakdown
    disease_breakdown = summary.get('disease_breakdown', {})
    if disease_breakdown:
        html += """
        <h2>Disease Breakdown</h2>
        <table>
            <tr><th>Disease</th><th>Count</th></tr>
"""
        for disease, count in sorted(disease_breakdown.items(), key=lambda x: x[1], reverse=True):
            html += f"            <tr><td>{disease}</td><td>{count}</td></tr>\n"
        html += "        </table>\n"

    # Charts
    chart_images = [
        ("disease_distribution.png", "Disease Distribution"),
        ("severity_distribution.png", "Severity Levels"),
        ("crop_health_summary.png", "Crop Health"),
        ("confidence_histogram.png", "Confidence Scores"),
    ]
    if accuracy_data:
        chart_images.append(("confusion_matrix.png", "Confusion Matrix"))

    html += """
        <h2>Visualizations</h2>
        <div class="chart-container">
"""
    for img_file, caption in chart_images:
        html += f'            <div class="chart"><img src="{img_file}" width="400"><p>{caption}</p></div>\n'
    html += """        </div>
"""

    # Accuracy metrics section
    if accuracy_data:
        overall_acc = accuracy_data.get('overall_accuracy', 0)
        total_eval = accuracy_data.get('total_evaluated', 0)
        correct_count = accuracy_data.get('correct', 0)
        incorrect_count = accuracy_data.get('incorrect', 0)
        unmatched_count = accuracy_data.get('unmatched', 0)
        per_class = accuracy_data.get('per_class', {})

        html += f"""
        <h2>Accuracy Metrics</h2>
        <div class="summary-box">
            <p><strong>Overall Accuracy:</strong> {overall_acc:.2%}</p>
            <p><strong>Total Evaluated:</strong> {total_eval}</p>
            <p><strong>Correct:</strong> <span class="healthy">{correct_count}</span></p>
            <p><strong>Incorrect:</strong> <span class="diseased">{incorrect_count}</span></p>
"""
        if unmatched_count > 0:
            html += f"            <p><strong>Unmatched:</strong> {unmatched_count}</p>\n"
        html += "        </div>\n"

        if per_class:
            html += """
        <h3>Per-Class Metrics</h3>
        <table>
            <tr>
                <th>Class</th>
                <th>Total</th>
                <th>Correct</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
            </tr>
"""
            for cls_name in sorted(per_class.keys()):
                metrics = per_class[cls_name]
                html += f"""            <tr>
                <td>{cls_name}</td>
                <td>{metrics['total']}</td>
                <td>{metrics['correct']}</td>
                <td>{metrics['accuracy']:.2%}</td>
                <td>{metrics['precision']:.2%}</td>
                <td>{metrics['recall']:.2%}</td>
                <td>{metrics['f1']:.2%}</td>
            </tr>
"""
            html += "        </table>\n"

    # Treatment recommendations
    unique_treatments = {}
    for pred in predictions:
        if 'error' not in pred and not pred.get('is_healthy', False):
            disease = pred.get('disease', 'Unknown')
            if disease not in unique_treatments:
                unique_treatments[disease] = pred.get('treatment', {})

    if unique_treatments:
        html += """
        <h2>Treatment Recommendations</h2>
"""
        for disease, treatment in unique_treatments.items():
            severity = treatment.get('severity', 'unknown')
            html += f"""
        <div class="treatment-card severity-{severity}">
            <h3>{disease}</h3>
            <p><strong>Severity:</strong> {severity.upper()}</p>
            <p><strong>Action:</strong> {treatment.get('action', 'N/A')}</p>
            <p><strong>Prevention:</strong> {treatment.get('prevention', 'N/A')}</p>
            <p><strong>Fertilizer:</strong> {treatment.get('fertilizer', 'N/A')}</p>
        </div>
"""

    # Detailed results table
    html += """
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Crop</th>
                <th>Disease</th>
                <th>Confidence</th>
                <th>Severity</th>
            </tr>
"""
    for pred in predictions[:50]:  # Limit to first 50
        if 'error' in pred:
            html += f"""
            <tr>
                <td>{pred.get('filename', 'N/A')}</td>
                <td colspan="4">Error: {pred['error']}</td>
            </tr>
"""
        else:
            status_class = 'healthy' if pred.get('is_healthy', False) else 'diseased'
            html += f"""
            <tr>
                <td>{pred.get('filename', 'N/A')}</td>
                <td>{pred.get('crop', 'N/A')}</td>
                <td class="{status_class}">{pred.get('disease', 'N/A')}</td>
                <td>{pred.get('confidence', 0):.2%}</td>
                <td>{pred.get('treatment', {}).get('severity', 'N/A')}</td>
            </tr>
"""

    if len(predictions) > 50:
        html += f"""
            <tr><td colspan="5"><em>... and {len(predictions) - 50} more results</em></td></tr>
"""

    html += """
        </table>

        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <p>Generated by Crop Health Workflow | Built with Claude</p>
        </footer>
    </div>
</body>
</html>
"""

    with open(output_path / 'report.html', 'w') as f:
        f.write(html)

    logger.info(f"Saved HTML report")


def generate_report(output_dir: str, format: str = 'all',
                    accuracy_file: str = None):
    """Generate complete report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load predictions
    data = []
    predictions = []
    for f in glob.glob("*.json"):
        partial = load_predictions(f)
        data.append(partial)
        predictions.append(partial.get('predictions', []))
    pprint(predictions)
        
    if not predictions:
        logger.warning("No predictions to report")
        return

    logger.info(f"Generating report for {len(predictions)} predictions")

    # Load accuracy data if provided
    accuracy_data = None
    if accuracy_file:
        try:
            with open(accuracy_file, 'r') as f:
                accuracy_data = json.load(f)
            logger.info(f"Loaded accuracy data: {accuracy_data.get('overall_accuracy', 0):.2%} overall accuracy")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load accuracy file: {e}")

    # Generate visualizations
    if format in ['all', 'charts']:
        create_disease_distribution_chart(predictions, output_path)
        create_severity_chart(predictions, output_path)
        create_crop_health_summary(predictions, output_path)
        create_confidence_histogram(predictions, output_path)
        if accuracy_data:
            create_confusion_matrix_chart(accuracy_data, output_path)

    # Generate HTML report
    if format in ['all', 'html']:
        generate_html_report(data, output_path, accuracy_data=accuracy_data)

    # Generate JSON summary
    if format in ['all', 'json']:
        summary_data = {
            'generated_at': datetime.now().isoformat(),
            'summary': data.get('summary', {}),
            'critical_alerts': data.get('critical_alerts', []),
        }
        if accuracy_data:
            summary_data['accuracy'] = {
                'overall_accuracy': accuracy_data.get('overall_accuracy'),
                'total_evaluated': accuracy_data.get('total_evaluated'),
                'correct': accuracy_data.get('correct'),
                'incorrect': accuracy_data.get('incorrect'),
            }
        summary_file = output_path / 'report_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved JSON summary")

    logger.info(f"Report generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate crop health report"
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for report'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['all', 'html', 'charts', 'json'],
        default='all',
        help='Output format'
    )

    parser.add_argument(
        '--accuracy', '-a',
        type=str,
        default=None,
        help='Optional accuracy results JSON file'
    )

    args = parser.parse_args()

    generate_report(args.output_dir, args.format,
                    accuracy_file=args.accuracy)


if __name__ == "__main__":
    main()
