"""Download the trusted course model into a fresh folder and verify inference."""
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd

REVISION = '723762febcfffdaf114f2ae835a3f8d8912ea7d3'
BASE = f'https://huggingface.co/iaac-maai/day3-group3-mumbai-rental-profiles/resolve/{REVISION}/'
EXPECTED = '4bf93aaa563afbef19d59aeb86bc0fca5e2b212d03d2bdfff542ca11a46deb1b'

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    for name in ('model.joblib', 'cluster_names.json'):
        urlretrieve(BASE + name, folder / name)
    assert hashlib.sha256((folder / 'model.joblib').read_bytes()).hexdigest() == EXPECTED
    # Only deserialize this trusted, explicitly pinned course artifact.
    model = joblib.load(folder / 'model.joblib')
    names = json.loads((folder / 'cluster_names.json').read_text())
    example = pd.DataFrame([{'Area_num': 750.0, 'BHK_num': 2.0, 'Bathrooms_num': 2.0}])
    label = int(model.predict(example)[0])
    assert label == 0 and names[str(label)] == 'Two-bedroom profiles'
    print(f'Published model: cluster {label}, {names[str(label)]}')
    root = Path(__file__).resolve().parents[1]
    generated = root / 'dataset/mumbai_rental_profiles.csv'
    if generated.exists():
        table = pd.read_csv(generated)
        np.testing.assert_array_equal(
            model.predict(table[['Area_num', 'BHK_num', 'Bathrooms_num']]), table.cluster,
        )
        print(f'Published model matches all {len(table):,} regenerated cluster labels.')
