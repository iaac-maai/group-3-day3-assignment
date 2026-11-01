"""Predict one classroom example using our trusted local fitted pipeline."""
from pathlib import Path
import hashlib
import json
import joblib
import pandas as pd

folder = Path(__file__).resolve().parent
evaluation = json.loads((folder / "evaluation.json").read_text())
artifact = folder / "model.joblib"
if hashlib.sha256(artifact.read_bytes()).hexdigest() != evaluation["model_sha256"]:
    raise ValueError("Model bytes differ from the recorded artifact. Review before loading.")
# Joblib can execute code: a matching hash does not make an untrusted file safe.
model = joblib.load(artifact)
names = json.loads((folder / "cluster_names.json").read_text())
example = pd.DataFrame({"Area_num": [750.0], "BHK_num": [2.0], "Bathrooms_num": [2.0]})
label = int(model.predict(example)[0])
print(f"Cluster {label}: {names[str(label)]}")
assert label == 0
