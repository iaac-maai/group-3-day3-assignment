---
license: mit
library_name: sklearn
tags:
  - clustering
  - k-means
  - unsupervised-learning
  - housing
---
# Mumbai physical rental profiles - K-Means

Local model card; this artifact has not been uploaded. Developed by Ahmed Belahcen, Nandini Rathod, Sanjana Kiran and Akanksha Dhumal, Group 3, AI Methods I, IAAC MaAI 2026/27. See the existing MIT LICENSE for code/model terms; source data retains CC BY 4.0 attribution.

## Model and training

The full pipeline median-imputes numeric inputs, applies ln(1 + area), standardizes the three features and fits K-Means (k=5, n_init=20, random_state=42). Only Area_num, BHK_num and Bathrooms_num form clusters. Rent, furnishing, facing and property type describe results afterwards. Balconies are excluded because more than half are missing. All 18,846 listings are retained; 15 missing bathroom values are imputed to 2 inside the fitted pipeline.

PCA is fitted on the prepared three-feature space for visualization; its first two axes retain 96.3069915% of variance. t-SNE uses all rows, perplexity=40, init=pca, learning_rate=auto, max_iter=1000, random_state=42, Barnes-Hut, angle=0.5 and two output dimensions. Neither embedding forms the K-Means inputs. The fitted clustering pipeline can predict new inputs; the exported t-SNE map cannot transform new listings.
The pipeline is fitted on the full cleaned snapshot. No supervised target, test split or accuracy is defined. See evaluation below for subsampling checks. Runtime: Python 3.12.14, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, joblib 1.6.0; see ../requirements.txt for the complete analysis environment.

## Inputs and outputs

Pass a pandas DataFrame with exactly these numeric feature columns:

| Input | Meaning | Unit / rule |
|---|---|---|
| Area_num | Reported area | Source says square feet; positive; unit consistency unverified |
| BHK_num | Bedroom count | Positive count |
| Bathrooms_num | Bathroom count | Positive count; NaN accepted and median-imputed |

The classroom pipeline assumes valid numeric inputs; it does not enforce all eligibility rules for external callers. Validate positive areas and counts before prediction. Output is an integer cluster ID from 0 to 4. Look up its descriptive name in cluster_names.json. IDs have no ordinal meaning and may change after retraining. The model does not output a rent estimate, probability, furnishing label or t-SNE coordinate.

| ID | Typical profile | Listings | Median area | Median bedrooms / baths | Median rent INR/month |
|---|---|---:|---:|---:|---:|
| 0 | Two-bedroom profiles | 8,006 | 1,025 | 2 / 2 | 65,000 |
| 1 | Three-bedroom profiles | 4,870 | 1,575 | 3 / 3 | 130,000 |
| 2 | Compact, one-bath profiles | 2,272 | 565 | 1 / 1 | 31,000 |
| 3 | Large, multi-room profiles | 1,685 | 2,750 | 4 / 4 | 300,000 |
| 4 | Compact, two-bath profiles | 2,013 | 650 | 1 / 2 | 38,000 |

## Evaluation

K-Means k=2,...,8 is fitted on the full dataset. Silhouette is evaluated on the same fixed 3,000-row random sample (seed 42): k=5 scores 0.5792832075113192 versus k=4 at 0.5718932442523518. Full-data inertia at k=5 is 6962.908980584956. The elbow is around four to five; five is a working interpretation, not a verified ground truth.

A separate fair comparison refits preprocessing and both K-Means and Ward hierarchical clustering on that same 3,000-row reference subset. At k=5 their silhouettes are 0.5817217246340777 and 0.573165655223993; agreement ARI is 0.8839211748987167. The notebook includes the Ward dendrogram.

Five seeds (7, 21, 42, 84, 123) give mean full-data assignment ARI 0.9922043456759448 (population SD 0.004798840195798869). Five 80% subsample refits, including preprocessing refitted on each subset, give mean ARI on all listings 0.9910121873666575 (SD 0.0047200014888221535). Scores for excluded rows are also recorded. Equal weighting of 2,478 distinct raw physical tuples gives ARI 0.8414894188661575 against the reference assignments, demonstrating sensitivity to repeated profiles. ARI is agreement, not predictive accuracy. These are exploratory internal checks, not a supervised train/test evaluation. Exact records are in models/evaluation.json.

## Use and artifact security

From the project root, in the pinned environment:

```python
import json
from pathlib import Path
import joblib
import pandas as pd

# Only load this trusted local artifact; see predict_example.py for a hash check.
model = joblib.load("models/model.joblib")
names = json.loads(Path("models/cluster_names.json").read_text())
example = pd.DataFrame({"Area_num": [750.0], "BHK_num": [2.0], "Bathrooms_num": [2.0]})
label = int(model.predict(example)[0])
print(label, names[str(label)])
# 0 Two-bedroom profiles
```

Joblib/pickle deserialization can execute code. Load only trusted artifacts in compatible environments. A checksum detects changed bytes but does not establish authorship. The notebook writes its own artifact and reloads it; the standalone example verifies the model SHA256 below. Predictions from the reloaded pipeline reproduce every exported label.

Model SHA256: `4bf93aaa563afbef19d59aeb86bc0fca5e2b212d03d2bdfff542ca11a46deb1b`.

## Intended use and limitations

Listings are not verified unique dwellings. Repeated characteristics influence weighting, and cluster names describe medians rather than strict membership rules. The source reports area in square feet, but row-level unit consistency is unverified. Rent is advertised monthly INR, not a transaction price. Collection date is unknown. Location is only Mumbai, not a neighborhood. Bedrooms and bathrooms have equal standardized-feature weighting to log-area, which is a modeling choice. No demographic or causal conclusions are supported. Use for classroom descriptive exploration; do not use for tenant selection, credit, automated valuation or claims about today's market.

## Training data and attribution

Source: [cleaned Day 2 snapshot](https://github.com/iaac-maai/group-3-day2-assignment/blob/6b2a6d17af8c4d8950f12e60dc97f8b9a8d4dbc1/data/mumbai_cleaned.csv), derived from Harshal Honde's [Property Rental Listings Dataset](https://www.kaggle.com/datasets/harshalhonde/property-rental-listings-dataset). The dataset is attributed under the source's declared CC BY 4.0 license; the existing project code/model license is MIT (see ../LICENSE). MIT does not replace the source data's attribution/license. Original collection date and listing verification are unavailable.

Source snapshot SHA256: `354d28e07432137cdc97a2197a556269137e74135af01186420e2e2917a19ee9`.
