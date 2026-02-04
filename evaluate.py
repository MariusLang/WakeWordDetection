import os
import json
import argparse
from glob import glob
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from detection.pytorch.pytorch_inference import load_model_from_path, preprocess_audio
from utils.data_loader import load_config
from utils.get_device import get_device


def find_wav_files(directory: str) -> list:
    return glob(os.path.join(directory, '**/*.wav'), recursive=True)


def evaluate_file(model, audio_file: str, cfg: dict, device: torch.device) -> dict:
    """
    Evaluate a single audio file.

    Returns:
        dict with keys:
            - max_prob: Maximum wakeword probability across segments
            - mean_prob: Mean wakeword probability across segments
            - positive_ratio: Ratio of segments with prob > 0.5
            - num_segments: Number of segments evaluated
            - segment_probs: List of all segment probabilities
    """
    try:
        segments = preprocess_audio(audio_file, cfg)

        if segments is None or len(segments) == 0:
            return {
                'error': 'No segments extracted',
                'max_prob': 0.0,
                'mean_prob': 0.0,
                'positive_ratio': 0.0,
                'num_segments': 0,
                'segment_probs': []
            }

        segments = segments.to(device)

        with torch.no_grad():
            logits = model(segments)
            probs = torch.softmax(logits, dim=1)
            wakeword_probs = probs[:, 1].cpu().numpy()

        return {
            'max_prob': float(np.max(wakeword_probs)),
            'mean_prob': float(np.mean(wakeword_probs)),
            'positive_ratio': float(np.mean(wakeword_probs > 0.5)),
            'num_segments': len(wakeword_probs),
            'segment_probs': wakeword_probs.tolist()
        }

    except Exception as e:
        return {
            'error': str(e),
            'max_prob': 0.0,
            'mean_prob': 0.0,
            'positive_ratio': 0.0,
            'num_segments': 0,
            'segment_probs': []
        }


def compute_metrics_at_threshold(
    wakeword_results: list,
    non_wakeword_results: list,
    threshold: float,
    aggregation: str = 'max_prob'
) -> dict:
    """
    Compute evaluation metrics at a specific threshold.

    Args:
        wakeword_results: List of evaluation results for wakeword files
        non_wakeword_results: List of evaluation results for non-wakeword files
        threshold: Detection threshold (file is detected as wakeword if aggregated prob > threshold)
        aggregation: How to aggregate segment probabilities ('max_prob', 'mean_prob', 'positive_ratio')

    Returns:
        dict with metrics: TP, TN, FP, FN, FAR, FRR, accuracy, precision, recall, f1
    """
    # True Positives: wakeword files correctly detected as wakeword
    # False Negatives: wakeword files incorrectly rejected
    tp = sum(1 for r in wakeword_results if r.get(aggregation, 0) > threshold)
    fn = len(wakeword_results) - tp

    # True Negatives: non-wakeword files correctly rejected
    # False Positives: non-wakeword files incorrectly accepted as wakeword
    tn = sum(1 for r in non_wakeword_results if r.get(aggregation, 0) <= threshold)
    fp = len(non_wakeword_results) - tn

    total_wakeword = len(wakeword_results)
    total_non_wakeword = len(non_wakeword_results)
    total = total_wakeword + total_non_wakeword

    # Compute metrics
    metrics = {
        'threshold': threshold,
        'aggregation': aggregation,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'total_wakeword': total_wakeword,
        'total_non_wakeword': total_non_wakeword,
    }

    # False Acceptance Rate: FP / Total Non-Wakeword
    if total_non_wakeword > 0:
        metrics['FAR'] = fp / total_non_wakeword
    else:
        metrics['FAR'] = None

    # False Rejection Rate: FN / Total Wakeword
    if total_wakeword > 0:
        metrics['FRR'] = fn / total_wakeword
    else:
        metrics['FRR'] = None

    # Accuracy: (TP + TN) / Total
    if total > 0:
        metrics['accuracy'] = (tp + tn) / total
    else:
        metrics['accuracy'] = None

    # Precision: TP / (TP + FP)
    if (tp + fp) > 0:
        metrics['precision'] = tp / (tp + fp)
    else:
        metrics['precision'] = None

    # Recall (Sensitivity): TP / (TP + FN) = 1 - FRR
    if total_wakeword > 0:
        metrics['recall'] = tp / total_wakeword
    else:
        metrics['recall'] = None

    # Specificity: TN / (TN + FP) = 1 - FAR
    if total_non_wakeword > 0:
        metrics['specificity'] = tn / total_non_wakeword
    else:
        metrics['specificity'] = None

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    if metrics['precision'] is not None and metrics['recall'] is not None:
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
    else:
        metrics['f1_score'] = None

    return metrics


def compute_threshold_sweep(
    wakeword_results: list,
    non_wakeword_results: list,
    aggregation: str = 'max_prob',
    num_thresholds: int = 101
) -> list:
    """
    Compute metrics across a range of thresholds for ROC/DET curve analysis.
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    results = []

    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(
            wakeword_results, non_wakeword_results, thresh, aggregation
        )
        results.append({
            'threshold': float(thresh),
            'FAR': metrics['FAR'],
            'FRR': metrics['FRR'],
            'accuracy': metrics['accuracy']
        })

    return results


def find_equal_error_rate(threshold_sweep: list) -> dict:
    """
    Find the Equal Error Rate (EER) - the point where FAR ≈ FRR.
    """
    valid_points = [p for p in threshold_sweep if p['FAR'] is not None and p['FRR'] is not None]

    if not valid_points:
        return {'eer': None, 'eer_threshold': None}

    # Find point where |FAR - FRR| is minimized
    min_diff = float('inf')
    eer_point = None

    for point in valid_points:
        diff = abs(point['FAR'] - point['FRR'])
        if diff < min_diff:
            min_diff = diff
            eer_point = point

    if eer_point:
        return {
            'eer': (eer_point['FAR'] + eer_point['FRR']) / 2,
            'eer_threshold': eer_point['threshold']
        }

    return {'eer': None, 'eer_threshold': None}


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a wake word detection model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full evaluation with both wakeword and non-wakeword data
  python evaluate.py --model experiments/wakeword_model_20260116_170802 \\
      --wakeword-dir data/wakeword_test \\
      --non-wakeword-dir data/non_wakeword \\
      --output results.json

  # Evaluate only wakeword detection (FRR only)
  python evaluate.py --model experiments/wakeword_model_20260116_170802 \\
      --wakeword-dir data/wakeword_test

  # Test at multiple thresholds
  python evaluate.py --model experiments/wakeword_model_20260116_170802 \\
      --wakeword-dir data/wakeword_test \\
      --non-wakeword-dir data/non_wakeword \\
      --threshold 0.3 --output results.json
        '''
    )

    parser.add_argument('--model', required=True,
                        help='Path to experiment directory (e.g., experiments/wakeword_model_TIMESTAMP)')
    parser.add_argument('--wakeword-dir',
                        help='Directory containing wakeword test samples (recursively searches for .wav files)')
    parser.add_argument('--non-wakeword-dir',
                        help='Directory containing non-wakeword test samples (recursively searches for .wav files)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold for primary metrics (default: 0.5)')
    parser.add_argument('--aggregation', choices=['max_prob', 'mean_prob', 'positive_ratio'],
                        default='max_prob',
                        help='How to aggregate segment probabilities (default: max_prob)')
    parser.add_argument('--output', '-o',
                        help='Output JSON file path (default: prints to stdout)')
    parser.add_argument('--include-file-details', action='store_true',
                        help='Include per-file probability details in output')
    parser.add_argument('--threshold-sweep', action='store_true',
                        help='Include threshold sweep analysis for ROC/DET curves')

    args = parser.parse_args()

    # Validate inputs
    if not args.wakeword_dir and not args.non_wakeword_dir:
        parser.error('At least one of --wakeword-dir or --non-wakeword-dir must be specified')

    if not os.path.exists(args.model):
        parser.error(f'Model directory not found: {args.model}')

    # Load configuration
    cfg = load_config()
    device = get_device()

    print(f'Loading model from: {args.model}')
    print(f'Using device: {device}')

    # Load model
    model = load_model_from_path(args.model, device)
    model.eval()

    # Load experiment config for metadata
    config_path = os.path.join(args.model, 'config.json')
    experiment_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            experiment_config = json.load(f)

    # Collect and evaluate files
    wakeword_results = []
    non_wakeword_results = []

    if args.wakeword_dir:
        if not os.path.exists(args.wakeword_dir):
            parser.error(f'Wakeword directory not found: {args.wakeword_dir}')

        wakeword_files = find_wav_files(args.wakeword_dir)
        print(f'Found {len(wakeword_files)} wakeword files in {args.wakeword_dir}')

        for audio_file in tqdm(wakeword_files, desc='Evaluating wakeword files'):
            result = evaluate_file(model, audio_file, cfg, device)
            result['file'] = os.path.relpath(audio_file, args.wakeword_dir)
            result['ground_truth'] = 'wakeword'
            result['predicted'] = 'wakeword' if result.get(args.aggregation, 0) > args.threshold else 'non_wakeword'
            result['correct'] = result['predicted'] == 'wakeword'
            wakeword_results.append(result)

    if args.non_wakeword_dir:
        if not os.path.exists(args.non_wakeword_dir):
            parser.error(f'Non-wakeword directory not found: {args.non_wakeword_dir}')

        non_wakeword_files = find_wav_files(args.non_wakeword_dir)
        print(f'Found {len(non_wakeword_files)} non-wakeword files in {args.non_wakeword_dir}')

        for audio_file in tqdm(non_wakeword_files, desc='Evaluating non-wakeword files'):
            result = evaluate_file(model, audio_file, cfg, device)
            result['file'] = os.path.relpath(audio_file, args.non_wakeword_dir)
            result['ground_truth'] = 'non_wakeword'
            result['predicted'] = 'wakeword' if result.get(args.aggregation, 0) > args.threshold else 'non_wakeword'
            result['correct'] = result['predicted'] == 'non_wakeword'
            non_wakeword_results.append(result)

    # Compute metrics at specified threshold
    metrics = compute_metrics_at_threshold(
        wakeword_results, non_wakeword_results, args.threshold, args.aggregation
    )

    # Build output structure
    output = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_path': os.path.abspath(args.model),
        'model_architecture': experiment_config.get('model_architecture', 'unknown'),
        'model_accuracy_training': experiment_config.get('final_accuracy'),
        'evaluation_config': {
            'threshold': args.threshold,
            'aggregation': args.aggregation,
            'wakeword_dir': os.path.abspath(args.wakeword_dir) if args.wakeword_dir else None,
            'non_wakeword_dir': os.path.abspath(args.non_wakeword_dir) if args.non_wakeword_dir else None,
            'wakeword_files_count': len(wakeword_results),
            'non_wakeword_files_count': len(non_wakeword_results),
        },
        'metrics': {
            'FAR': metrics['FAR'],
            'FRR': metrics['FRR'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'specificity': metrics['specificity'],
            'f1_score': metrics['f1_score'],
            'true_positives': metrics['true_positives'],
            'true_negatives': metrics['true_negatives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
        }
    }

    # Add threshold sweep if requested
    if args.threshold_sweep and wakeword_results and non_wakeword_results:
        sweep = compute_threshold_sweep(wakeword_results, non_wakeword_results, args.aggregation)
        eer_info = find_equal_error_rate(sweep)
        output['threshold_analysis'] = {
            'equal_error_rate': eer_info['eer'],
            'eer_threshold': eer_info['eer_threshold'],
            'sweep': sweep
        }

    # Add per-file details if requested
    if args.include_file_details:
        # Remove segment_probs to reduce output size unless explicitly needed
        wakeword_summary = []
        for r in wakeword_results:
            summary = {k: v for k, v in r.items() if k != 'segment_probs'}
            wakeword_summary.append(summary)

        non_wakeword_summary = []
        for r in non_wakeword_results:
            summary = {k: v for k, v in r.items() if k != 'segment_probs'}
            non_wakeword_summary.append(summary)

        output['file_results'] = {
            'wakeword': wakeword_summary,
            'non_wakeword': non_wakeword_summary
        }

        # Add misclassified files for quick reference
        output['misclassified'] = {
            'false_rejections': [r['file'] for r in wakeword_results if not r['correct']],
            'false_acceptances': [r['file'] for r in non_wakeword_results if not r['correct']]
        }

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nResults saved to: {args.output}')
    else:
        print(json.dumps(output, indent=2))

    # Print summary to console
    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f"Model: {experiment_config.get('model_architecture', 'unknown')}")
    print(f"Threshold: {args.threshold} (aggregation: {args.aggregation})")
    print(f"Wakeword files: {len(wakeword_results)}")
    print(f"Non-wakeword files: {len(non_wakeword_results)}")
    print('-' * 60)

    if metrics['FAR'] is not None:
        print(f"False Acceptance Rate (FAR): {metrics['FAR']:.4f} ({metrics['false_positives']} / {metrics['total_non_wakeword']})")
    if metrics['FRR'] is not None:
        print(f"False Rejection Rate (FRR): {metrics['FRR']:.4f} ({metrics['false_negatives']} / {metrics['total_wakeword']})")
    if metrics['accuracy'] is not None:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    if metrics['precision'] is not None:
        print(f"Precision: {metrics['precision']:.4f}")
    if metrics['recall'] is not None:
        print(f"Recall: {metrics['recall']:.4f}")
    if metrics['f1_score'] is not None:
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    if 'threshold_analysis' in output:
        print('-' * 60)
        print(f"Equal Error Rate (EER): {output['threshold_analysis']['equal_error_rate']:.4f}")
        print(f"EER Threshold: {output['threshold_analysis']['eer_threshold']:.3f}")

    print('=' * 60)


if __name__ == '__main__':
    main()
