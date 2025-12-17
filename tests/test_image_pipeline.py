import sys
import pathlib
import csv
from pathlib import Path

import pytest

# Ensure src is on path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data_toolkit import image_data


def test_generate_digit_images(tmp_path):
    out = tmp_path / 'digits'
    out_str = str(out)
    # generate 20 small images
    image_data.generate_digit_images(output_dir=out_str, n_images=20, image_size=64, seed=1)

    labels = out / 'labels.csv'
    assert labels.exists(), 'labels.csv was not created'

    # read CSV and count rows
    with open(labels, newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 20

    images_dir = out / 'images'
    assert images_dir.exists(), 'images/ folder not created'
    img_files = list(images_dir.glob('*.png'))
    assert len(img_files) >= 1


def test_train_and_predict_smoke(tmp_path):
    # Skip if tensorflow not available
    pytest.importorskip('tensorflow')

    out = tmp_path / 'digits'
    out_str = str(out)
    image_data.generate_digit_images(output_dir=out_str, n_images=40, image_size=64, seed=2)

    from data_toolkit import image_models

    model_out = str(tmp_path / 'model.h5')
    res = image_models.train_cnn(
        data_dir=out_str,
        labels_csv='labels.csv',
        image_size=64,
        batch_size=8,
        epochs=1,
        model_out=model_out,
        base_filters=16,
        depth=2,
        dense_units=64,
        dropout=0.3,
    )

    assert 'model_path' in res
    assert Path(res['model_path']).exists()

    # pick one image to run prediction
    imgs = list((Path(out_str) / 'images').glob('*.png'))
    assert len(imgs) > 0
    sample = str(imgs[0])

    pred = image_models.predict_image(sample, res['model_path'], image_size=64, class_names=res.get('class_names'))
    assert 'predicted_label' in pred
    assert 'probabilities' in pred
