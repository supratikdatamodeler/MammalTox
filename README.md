# MammalTox 1.0

**Machine-learning platform for species-specific mammalian acute toxicity prediction**

MammalTox 1.0 is a browser-based machine-learning application for predicting **intravenous acute toxicity (LD50 classification)** across six mammalian species. The platform integrates molecular-structure validation, endpoint-specific Mordred descriptor calculation, trained machine-learning classifiers, applicability-domain assessment, chemical-space visualization, and model interpretation in a single research-oriented interface.

## Supported endpoints

| Species | Endpoint | Model |
|---|---|---|
| Cat | IV LD50 | Support Vector Classifier |
| Dog | IV LD50 | Decision Tree |
| Guinea pig | IV LD50 | Multilayer Perceptron |
| Mouse | IV LD50 | XGBoost |
| Rabbit | IV LD50 | AdaBoost |
| Rat | IV LD50 | Multilayer Perceptron |

## Key features

- Single-compound prediction from **SMILES** or an embedded 2D chemical-structure editor
- Batch prediction from **CSV, XLSX, SMI, TXT, and SDF**
- Species-specific **Low toxicity / High toxicity** classification
- Predicted-class probability where supported by the deployed estimator
- Endpoint-specific **k-nearest-neighbor applicability domain**
- **PCA chemical-space visualization**
- Batch **SHAP summary plots**
- Individual **SHAP waterfall plots**
- Model-performance and validation information
- Downloadable prediction results in **CSV and Excel**
- Automatic SMILES parsing, canonicalization, and molecular-descriptor generation using RDKit and Mordred

## Toxicity class definition

MammalTox uses the following binary encoding:

| Class | Interpretation |
|---|---|
| **0** | Low toxicity |
| **1** | High toxicity |

The reported predicted-class probability is the probability assigned by the deployed model to the displayed class. It should be interpreted as a **model-derived confidence measure rather than experimental certainty**.

## Applicability domain

An endpoint-specific distance-based applicability domain is used to determine whether a submitted compound is adequately represented by the chemical space of the corresponding training set.

For each training compound, the compound itself is excluded and the mean Euclidean distance to its **five nearest remaining training compounds** is calculated in the model-selected Mordred descriptor space. The applicability threshold is defined as the **95th percentile** of these training-set mean distances.

For a submitted compound:

- **In AD** — mean distance to the five nearest training compounds is less than or equal to the endpoint-specific threshold.
- **Out of AD** — mean distance exceeds the threshold.

Predictions outside the applicability domain should be interpreted with additional caution.

## Web application

MammalTox is designed for deployment through **Streamlit Community Cloud**.

Once the public application is deployed, add the application URL here:

```text
https://<your-streamlit-app-url>
```

## Installation

Python **3.10** is recommended because the serialized models were validated using the package versions specified in `requirements.txt`.

Clone the repository:

```bash
git clone https://github.com/<username>/MammalTox.git
cd MammalTox
```

Create a virtual environment.

### Linux / macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run MammalTox:

```bash
python -m streamlit run streamlit_app.py
```

The local application will normally open at:

```text
http://localhost:8501
```

## Input formats

### Single-compound prediction

Users may:

- enter a valid SMILES string, or
- draw/edit a molecular structure using the embedded 2D molecular editor.

MammalTox accepts molecular structures that can be successfully parsed and standardized by RDKit, including canonical, non-canonical, isomeric, aromatic, and Kekulé SMILES.

### Batch prediction

#### CSV / XLSX

A SMILES column is required. Common case-insensitive column names are recognized, including:

```text
SMILES
smiles
Smiles
Canonical_SMILES
canonical_smiles
```

A compound ID or compound-name column is optional.

#### SMI / TXT

Use one compound per line:

```text
CCO Compound_1
CC(=O)O Compound_2
c1ccccc1 Compound_3
```

The identifier or name is optional.

#### SDF

Single- and multi-record SDF files are supported. Available compound identifiers and names are retained when possible.

An example batch file is provided at:

```text
example_files/example_batch_input.csv
```

## Prediction outputs

Downloaded results include:

| Field | Description |
|---|---|
| Compound ID | Submitted or automatically generated compound identifier |
| Input SMILES | Original submitted structure |
| Canonical SMILES | RDKit-generated canonical isomeric SMILES |
| Species | Selected mammalian species |
| Route | Exposure route |
| Endpoint | Toxicity endpoint |
| Model | Deployed machine-learning algorithm |
| Prediction | Low toxicity or High toxicity |
| Predicted-class probability | Probability assigned to the displayed prediction |
| AD Status | In AD, Out of AD, or Unavailable |
| AD Distance | Mean distance to the five nearest training compounds |
| AD Threshold | Endpoint-specific applicability-domain cutoff |
| Error Message | Processing or prediction error, when applicable |

## Repository structure

```text
MammalTox/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── assets/
│   └── Application images and interface assets
│
├── models/
│   └── Deployed endpoint-specific models and scalers
│
├── metadata/
│   └── Descriptor lists, model registry, performance metrics,
│       dataset metadata, and applicability-domain information
│
├── data/
│   ├── training_files/
│   ├── test_files/
│   └── model_information/
│
├── example_files/
│   └── Example batch-prediction files
│
├── utils/
│   └── Prediction, parsing, descriptor calculation, AD,
│       PCA, SHAP, plotting, and model-loading modules
│
└── tests/
    └── Release and deployment validation scripts
```

## Model interpretation

MammalTox provides complementary tools for understanding predictions.

### PCA chemical space

Principal component analysis is used to visualize the position of submitted compounds relative to compounds in the corresponding endpoint training set.

### SHAP analysis

SHAP-based interpretation is available for supported model workflows:

- **Summary plots** provide descriptor-level interpretation across batch predictions.
- **Waterfall plots** provide compound-specific interpretation of individual predictions.

SHAP results describe the behavior of the deployed statistical model and should not automatically be interpreted as mechanistic or causal toxicological relationships.

## Reproducibility

The public repository contains the files required to reproduce the deployed prediction workflow, including:

- endpoint-specific serialized models,
- model-selected descriptors,
- preprocessing information,
- training descriptor spaces required for applicability-domain calculations,
- model-performance metadata,
- dependency specifications,
- prediction and visualization utilities.

Package versions required for model compatibility are specified in:

```text
requirements.txt
```

Changes to package versions, descriptor implementations, preprocessing procedures, or serialized-model dependencies may alter predictions and should therefore be independently validated before use.

## Limitations

MammalTox is a computational prediction platform intended primarily for **research and screening applications**.

Important limitations include:

- Predictions are statistical estimates and do not replace experimental toxicity measurements.
- Predictive performance differs among species-specific endpoints.
- Chemical-space coverage is endpoint dependent.
- Predictions outside the applicability domain have greater extrapolation uncertainty.
- A high predicted probability does not establish toxicological certainty.
- SHAP values describe model behavior and do not establish biological causality.
- Input structures that cannot be parsed or represented by the required molecular descriptors cannot be evaluated.
- Large batch calculations, particularly SHAP analyses, may require substantial computational resources.
- MammalTox should not be used as the sole basis for clinical, regulatory, environmental, or safety-critical decisions.

## Citation

A formal citation to the peer-reviewed MammalTox publication will be added following publication.


Recommended software citation format:

```text
Xin, L., & Kar, S. MammalTox 1.0: Machine-learning platform for
species-specific mammalian acute toxicity prediction. Version 1.0.
Github: https://github.com/supratikdatamodeler/MammalTox
Streamlit: 
```


## Developers and affiliation

**Lihui Xin**  
Chemometrics & Molecular Modeling Laboratory  
Department of Chemistry and Physics  
Kean University, New Jersey, USA  
Email: `xinl@kean.edu`

**Supratik Kar, Ph.D.**  
Associate Professor
Chemometrics & Molecular Modeling Laboratory  
Department of Chemistry and Physics  
Kean University, New Jersey, USA  
Email: `skar@kean.edu`

## License

MammalTox is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license terms.

Third-party Python packages, molecular-informatics libraries, and other dependencies remain subject to their respective licenses.
