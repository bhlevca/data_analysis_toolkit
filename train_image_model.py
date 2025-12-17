"""
CLI to train image CNN using labels.csv produced by `src.data_toolkit.image_data`.

Example:
    python train_image_model.py --data test_data/digits --epochs 20 --batch 32 --model_out models/digits_cnn.h5
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure `src` is importable when running from repo root
ROOT = Path(__file__).parent.resolve()
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_toolkit.image_models import train_cnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to data folder containing labels.csv (and images subfolders)')
    p.add_argument('--labels_csv', default='labels.csv')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--batch', '--batch_size', type=int, default=32, help='Batch size (alias: --batch_size)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--model_out', default='models/digits_cnn.keras')
    p.add_argument('--save_format', choices=['h5', 'keras'], default='keras', help='Model save format (prefer native .keras)')
    p.add_argument('--transfer', action='store_true', help='Use transfer learning (MobileNetV2)')
    p.add_argument('--fine_tune', action='store_true', help='Fine-tune base model when using transfer learning')
    p.add_argument('--base_filters', type=int, default=32)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--dense', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.4)
    args = p.parse_args()

    print('Training with settings:', args)

    if args.transfer:
        # Use transfer learning trainer
        result = None
        try:
            result = __import__('src.data_toolkit.image_models', fromlist=['train_transfer_learning']).train_transfer_learning(
                args.data,
                labels_csv=args.labels_csv,
                image_size=args.image_size,
                batch_size=args.batch,
                epochs=args.epochs,
                model_out=args.model_out,
                base_trainable=args.fine_tune,
            )
        except Exception:
            # fallback import path
            from data_toolkit.image_models import train_transfer_learning
            result = train_transfer_learning(
                args.data,
                labels_csv=args.labels_csv,
                image_size=args.image_size,
                batch_size=args.batch,
                epochs=args.epochs,
                model_out=args.model_out,
                base_trainable=args.fine_tune,
            )
    else:
        # standard CNN training
        result = train_cnn(
            data_dir=args.data,
            labels_csv=args.labels_csv,
            image_size=args.image_size,
            batch_size=args.batch,
            epochs=args.epochs,
            model_out=args.model_out,
            base_filters=args.base_filters,
            depth=args.depth,
            dense_units=args.dense,
            dropout=args.dropout
        )

    # Save in requested format if necessary
    model = result.get('model')
    out_path = Path(args.model_out)
    if args.save_format == 'keras':
        keras_path = out_path.with_suffix('.keras')
        model.save(str(keras_path))
        print('Finished training. Model saved to', str(keras_path))
    else:
        # ensure model already saved by trainer; otherwise save as h5
        if not out_path.exists():
            model.save(str(out_path))
        print('Finished training. Model saved to', str(out_path))


if __name__ == '__main__':
    main()
