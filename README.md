# MammalTox 1.0

**ML-based mammalian acute toxicity prediction**

MammalTox 1.0 is a browser-based machine-learning platform for species-specific classification of mammalian intravenous acute toxicity. The application validates submitted chemical structures with RDKit, calculates the endpoint-specific Mordred molecular descriptors used during model development, applies the deployed classifier, and presents prediction and chemical-space context in a research-oriented interface.

## Available endpoints

| Endpoint | Deployed model |
|---|---|
| Cat IV LD50 | SVC |
| Dog IV LD50 | Decision Tree |
| Guinea Pig IV LD50 | MLP |
| Mouse IV LD50 | XGBoost |
| Rabbit IV LD50 | AdaBoost |
| Rat IV LD50 | MLP |

## Main functions

- Single-compound input using synchronized SMILES and chemical-structure drawing
- Batch prediction from pasted SMILES or uploaded CSV/XLSX, SMI/TXT, and SDF files
- Endpoint-specific Low toxicity / High toxicity classification
- Predicted-class probability where supported by the deployed estimator
- Endpoint-specific k-nearest-neighbor distance-based applicability domain
- Two-dimensional PCA chemical-space visualization
- Batch SHAP Summary and Individual Waterfall Plot
- Model information, final model-performance tables, and validation figures
- Downloadable CSV and Excel prediction results

## Class definition

The deployed binary encoding is:

- **Class 0 = Low toxicity**
- **Class 1 = High toxicity**

Predicted-class probability is the probability assigned by the model to the displayed predicted class. It is a model output, not experimental certainty.

## Applicability domain

Applicability-domain status is calculated separately for each endpoint in the model-selected Mordred descriptor space. The query compound's mean Euclidean distance to its five nearest training compounds is compared with a training-derived cutoff equal to the 95th percentile of training-set mean five-nearest-neighbor distances.

- **In AD:** the query distance is less than or equal to the endpoint cutoff.
- **Out of AD:** the query distance exceeds the endpoint cutoff and the prediction should be interpreted with additional caution.

## Installation

Python 3.10 is recommended because the deployed serialized estimators were verified with the versions pinned in `requirements.txt`.

```bash
git clone https://github.com/<organization-or-user>/<repository-name>.git
cd <repository-name>

python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run MammalTox locally from the repository root:

```bash
python3 -m streamlit run streamlit_app.py
```

The local application is normally available at `http://localhost:8501`.

## Streamlit Community Cloud deployment

1. Upload this release folder as the root of a GitHub repository.
2. Sign in to Streamlit Community Cloud and select **Create app**.
3. Select the GitHub repository and deployment branch.
4. Set the main file path to `streamlit_app.py`.
5. In advanced settings, select Python 3.10 when that option is available.
6. Deploy the app.
7. Review the build log if a pinned scientific package cannot be installed. Do not remove model-compatibility pins without revalidating all six serialized estimators.

No credentials or Streamlit secrets are required by the current application.

## Input formats

### Single compound

Enter one valid, parseable SMILES or draw/edit one structure in the embedded two-dimensional molecular editor. The application accepts canonical, non-canonical, isomeric, aromatic, and Kekule SMILES when RDKit can parse and standardize the submitted single compound.

### Batch input

- **CSV/XLSX:** requires a SMILES column. Case-insensitive names such as `SMILES`, `smiles`, `Smiles`, `Canonical_SMILES`, and `canonical_smiles` are recognized. An ID or compound-name column is optional.
- **SMI/TXT:** one SMILES per line, optionally followed by a compound identifier or name.
- **SDF:** one or more molecular records; available identifiers and names are retained when possible.

The example file in `example_files/example_batch_input.csv` can be used to test batch upload.

## Output fields

Downloaded prediction results include the following user-facing fields:

- **Compound ID:** submitted identifier or an application-generated identifier
- **Input SMILES:** structure string supplied by the user
- **Canonical SMILES:** RDKit-generated canonical isomeric SMILES
- **Species:** endpoint species
- **Route:** exposure route
- **Endpoint:** LD50 classification endpoint
- **Model:** deployed endpoint model
- **Prediction:** Low toxicity or High toxicity
- **Predicted-class probability:** probability assigned to the displayed prediction, or a clear non-calibrated status when probability output is unavailable
- **AD Status:** In AD, Out of AD, or Unavailable
- **Error Message:** `No error` for a successful row or a concise reason for a genuine processing or prediction failure

AD distance and threshold are also retained in downloaded results for transparent interpretation.

## Repository structure

```text
.
├── streamlit_app.py              # Streamlit Community Cloud entry point
├── requirements.txt              # Pinned Python dependencies
├── runtime.txt                   # Recommended Python runtime
├── README.md
├── DEPLOYMENT_MANIFEST.md
├── assets/                       # Header and browser-icon assets
├── models/                       # Six deployed endpoint estimators and required scalers
├── metadata/                     # Registry, descriptors, metrics, dataset and AD metadata
├── data/
│   ├── training_files/           # Selected training descriptor spaces for AD/PCA/SHAP
│   ├── test_files/               # Compact validation samples
│   └── model_information/        # Test, curated, and optimization data used by app figures
├── example_files/                # Example batch input
├── tests/                        # Release validation script
└── utils/                        # Prediction, parsing, AD, PCA, SHAP, plotting, and loading modules
```

See `DEPLOYMENT_MANIFEST.md` for the purpose of every included file category and the categories deliberately excluded from the public release.

## Reproducibility and limitations

- MammalTox outputs are computational estimates intended for research use.
- Predictions should be interpreted together with endpoint-specific applicability-domain status.
- The software does not replace experimental testing and should not be used as the sole basis for experimental, clinical, regulatory, or safety decisions.
- Model performance and chemical-space coverage are endpoint-specific and should not be generalized across species.
- Input structures that cannot be parsed or represented by the endpoint descriptor/preprocessing workflow may fail.
- Large batch SHAP calculations can require substantial time and memory on shared cloud infrastructure.

## Citation

A citation for the peer-reviewed MammalTox publication will be added after publication. Do not cite a placeholder DOI or unpublished bibliographic record.

## Authors and affiliation

- **Lihui Xin** — `lihuixin22@gmail.com`, `xinl@kean.edu`
- **Supratik Kar** — `skar@kean.edu`

Chemometrics & Molecular Modeling Laboratory, Department of Chemistry and Physics, Kean University, New Jersey, USA

## License

No project software license has been selected in this release package. The authors must choose and add an appropriate license before public release. Third-party packages and components remain subject to their respective licenses.
