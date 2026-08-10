from __future__ import annotations

import ast
import base64
from functools import cmp_to_key
import html
from io import BytesIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from utils.descriptor_calculation import canonicalize_smiles, official_descriptor_descriptions
from utils.metadata_loading import (
    APP_ROOT,
    active_models,
    load_dataset_statistics,
    load_model_performance,
    load_registry,
    load_selected_descriptors,
    resolve_app_path,
)
from utils.model_performance import (
    final_model_development_figure,
    final_model_evaluation_figure,
    load_endpoint_source_map,
    prepare_model_performance_data,
)
from utils.plotting import class_distribution_figure, class_distribution_percentage_table
from utils.prediction import (
    PredictionBundle,
    display_endpoint,
    display_route,
    endpoint_display_label,
    main_results_table,
    predict_compounds,
    technical_output_table,
)
from utils.structure_input import (
    COMPOUND_COLUMNS,
    editor_molblock_to_smiles,
    example_input_bytes,
    example_preview_frame,
    parse_smi_text,
    parse_uploaded_file,
    smiles_to_editor_molblock,
    validate_single_smiles_input,
    validate_compound_inputs,
)
from utils.visualizations import batch_shap_figure, local_shap_figure, pca_chemical_space_figure


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_SOURCE_PATH = ASSETS_DIR / "MammalTox Logo.png"
HEADER_LOGO_PATH = ASSETS_DIR / "mammaltox_header_logo_v1_retina.png"
HEADER_EMBLEM_PATH = ASSETS_DIR / "mammaltox_header_emblem_retina.png"
HEADER_WORDMARK_PATH = ASSETS_DIR / "mammaltox_header_wordmark_retina.png"
ICON_PATH = ASSETS_DIR / "mammaltox_sidebar_icon_retina.png"
FAVICON_PATH = ASSETS_DIR / "mammaltox_favicon_512.png"
HEADER_EMBLEM_CSS_WIDTH = 238
HEADER_WORDMARK_CSS_WIDTH = 500
SIDEBAR_LOGO_CSS_WIDTH = 128
DEPLOYMENT_CACHE_VERSION = "mammaltox-1.0-github-release-2026-07-31"
LOGO_RENDER_AUDIT = {
    "source_filename": LOGO_SOURCE_PATH.name,
    "source_dimensions": (1536, 1024),
    "header_dimensions": (1408, 571),
    "header_emblem_dimensions": (520, 571),
    "header_wordmark_dimensions": (858, 160),
    "sidebar_dimensions": (768, 768),
    "favicon_dimensions": (512, 512),
    "header_emblem_css_width": HEADER_EMBLEM_CSS_WIDTH,
    "header_wordmark_css_width": HEADER_WORDMARK_CSS_WIDTH,
    "sidebar_css_width": SIDEBAR_LOGO_CSS_WIDTH,
}
PAGE_ICON = Image.open(FAVICON_PATH) if FAVICON_PATH.exists() else None

st.set_page_config(page_title="MammalTox 1.0", page_icon=PAGE_ICON, layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {display: none !important;}
    [data-testid="stAppViewContainer"] > .main {margin-left: 0 !important;}
    .block-container {
        margin-left: auto;
        margin-right: auto;
        max-width: 1280px;
        padding: 2rem 2rem 3rem;
    }
    h1, h2, h3 {color: #1f2933;}
    .stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
    .stTabs [data-baseweb="tab"] {background-color: #f7f9fb; border-radius: 6px 6px 0 0;}
    .mammaltox-subtitle {color: #52616b; font-size: 1.02rem; margin: 0.25rem 0 1.15rem 0;}
    .mammaltox-header-brand {align-items: center; display: flex; gap: 1.35rem; margin: 0 0 0.45rem; max-width: 900px; min-width: 0; width: 100%;}
    .mammaltox-header-emblem {display: block; flex: 0 0 auto; height: auto; object-fit: contain; width: 238px;}
    .mammaltox-header-wordmark-block {flex: 1 1 auto; margin-top: 2rem; min-width: 0;}
    .mammaltox-header-wordmark-row {align-items: flex-end; display: flex; min-width: 0;}
    .mammaltox-header-wordmark {display: block; flex: 0 1 auto; height: auto; max-width: calc(100% - 3.8rem); object-fit: contain; width: 500px;}
    .mammaltox-header-version {align-self: flex-end; color: #bb8b3d; flex: 0 0 auto; font-family: Didot, "Bodoni 72", Baskerville, Georgia, "Times New Roman", serif; font-size: 4.45rem; font-weight: 400; letter-spacing: 0; line-height: 0.9; margin: 0 0 -0.08rem 0.22rem; white-space: nowrap;}
    .mammaltox-header-tagline {color: #263746; font-size: 1.12rem; line-height: 1.25; margin: 0.32rem 0 0 0.08rem; white-space: nowrap;}
    .mammaltox-live-title-version {color: #bb8b3d; font-family: Didot, "Bodoni 72", Baskerville, Georgia, "Times New Roman", serif; font-size: 0.72em; font-weight: 400; vertical-align: baseline; white-space: nowrap;}
    .mammaltox-affiliation {color: #313b44; font-size: 0.94rem; margin: 0.1rem 0 1rem; overflow-x: auto; padding-bottom: 0.1rem; white-space: nowrap;}
    .descriptor-definition-table {border-collapse: collapse; table-layout: fixed; width: 100%;}
    .descriptor-definition-table th, .descriptor-definition-table td {border-bottom: 1px solid #e4e8eb; padding: 0.48rem 0.62rem; text-align: left; vertical-align: top; white-space: normal; overflow-wrap: anywhere;}
    .descriptor-definition-table th {background: #f7f8f9; color: #33414c; font-weight: 600;}
    .descriptor-definition-table th:first-child, .descriptor-definition-table td:first-child {width: 28%;}
    .descriptor-definition-table th:last-child, .descriptor-definition-table td:last-child {width: 72%;}
    .shap-view-title {color: #1f2933; font-size: 1.55rem; font-weight: 650; line-height: 1.2; margin: 1rem 0 0.35rem;}
    .pca-view-title {margin-bottom: -0.2rem;}
    .single-structure-panel-heading {
        color: #26333d;
        font-size: 1.35rem;
        font-weight: 650;
        line-height: 1.25;
        margin: 0 0 0.7rem;
    }
    .model-information-section-heading {
        color: #26333d;
        font-size: 1.08rem;
        font-weight: 650;
        line-height: 1.25;
        margin: 0 0 0.65rem;
    }
    .model-summary-metrics-marker {display: none;}
    div[data-testid="stElementContainer"]:has(.model-summary-metrics-marker) {
        display: none;
    }
    .structure-sync-indicator {
        align-items: center;
        box-sizing: border-box;
        color: #58716f;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        min-height: 74px;
        padding-top: 1.1rem;
        width: 68px;
    }
    div[data-testid="stHorizontalBlock"]:has(.single-structure-workspace-marker)
    > div[data-testid="stColumn"]:has(.single-structure-workspace-marker) {
        flex: 0 0 68px !important;
        min-width: 68px !important;
        width: 68px !important;
    }
    .structure-sync-icon {
        display: block;
        flex: 0 0 42px;
        height: 42px;
        width: 42px;
    }
    .structure-sync-label {
        color: #58716f;
        font-size: 0.68rem;
        font-weight: 600;
        line-height: 1;
        margin-top: 0.25rem;
        text-align: center;
        white-space: nowrap;
    }
    .structure-valid-status {align-items: center; color: #466452; display: flex; font-size: 0.84rem; gap: 0.3rem; margin: 0.2rem 0 0.45rem;}
    .structure-valid-dot {background: #6f9279; border-radius: 50%; display: inline-block; height: 0.48rem; width: 0.48rem;}
    .ad-information-card {
        align-items: center;
        background: #fbfcfc;
        border: 1px solid #dfe4e7;
        border-radius: 7px;
        box-sizing: border-box;
        display: flex;
        gap: 0.8rem;
        margin: 0.35rem 0 0.75rem;
        padding: 0.65rem 0.8rem;
        width: 100%;
    }
    .ad-information-icon-block {flex: 0 0 50px; text-align: center;}
    .ad-information-icon {display: block; height: 46px; margin: 0 auto; width: 46px;}
    .ad-information-icon-label {color: #526b75; font-size: 0.64rem; font-weight: 600; line-height: 1; margin-top: 0.12rem;}
    .ad-information-copy {color: #3f4d55; font-size: 0.86rem; line-height: 1.35; min-width: 0;}
    .ad-information-copy p {margin: 0.08rem 0;}
    .ad-information-title {align-items: center; color: #263842; display: flex; flex-wrap: wrap; font-size: 0.94rem; gap: 0.35rem; margin: 0 0 0.18rem;}
    .ad-method-badge {background: #edf3f2; border: 1px solid #cddbd8; border-radius: 4px; color: #456c68; font-size: 0.66rem; font-weight: 600; line-height: 1; padding: 0.2rem 0.32rem;}
    @media (max-width: 640px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        .mammaltox-header-brand {gap: 0.7rem;}
        .mammaltox-header-emblem {width: 150px;}
        .mammaltox-header-wordmark-block {margin-top: 1.15rem;}
        .mammaltox-header-wordmark {max-width: calc(100% - 2.8rem); width: 390px;}
        .mammaltox-header-version {font-size: 3.4rem; margin: 0 0 -0.05rem 0.18rem;}
        .mammaltox-header-tagline {font-size: 0.83rem; white-space: normal;}
        .ad-information-card {align-items: flex-start; gap: 0.65rem;}
        .ad-information-icon {flex-basis: 42px; height: 42px; width: 42px;}
    }
    @media (min-width: 641px) and (max-width: 900px) {
        .mammaltox-header-brand {gap: 1rem;}
        .mammaltox-header-emblem {width: 205px;}
        .mammaltox-header-wordmark-block {margin-top: 1.55rem;}
        .mammaltox-header-wordmark {width: 455px;}
        .mammaltox-header-version {font-size: 4rem; margin: 0 0 -0.06rem 0.2rem;}
        .mammaltox-header-tagline {font-size: 0.98rem;}
    }
    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"]:has(.model-summary-metrics-marker) {
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        div[data-testid="stHorizontalBlock"]:has(.model-summary-metrics-marker)
        > div[data-testid="stColumn"] {
            flex: 1 1 calc(50% - 0.75rem) !important;
            min-width: 220px !important;
            width: auto !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.model-information-tables-marker) {
            flex-direction: column;
            gap: 0.75rem;
        }
        div[data-testid="stHorizontalBlock"]:has(.model-information-tables-marker)
        > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 0 !important;
            width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.single-structure-workspace-marker) {
            flex-direction: column;
            gap: 0.45rem;
        }
        div[data-testid="stHorizontalBlock"]:has(.single-structure-workspace-marker) > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 0 !important;
            width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.single-structure-workspace-marker)
        > div[data-testid="stColumn"]:has(.single-structure-workspace-marker) {
            flex: 1 1 100% !important;
            min-width: 0 !important;
            width: 100% !important;
        }
        .structure-sync-indicator {
            margin: 0 auto;
            min-height: 50px;
            padding-top: 0.2rem;
            width: 68px;
        }
        div[data-testid="stHorizontalBlock"]:has(.structure-responsive-marker) {
            flex-direction: column;
            gap: 0.75rem;
        }
        div[data-testid="stHorizontalBlock"]:has(.structure-responsive-marker) > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 0 !important;
            width: 100% !important;
        }
    }
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"]:has(.model-summary-metrics-marker)
        > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_registry_resources(deployment_version: str):
    del deployment_version
    return load_registry(APP_ROOT), load_selected_descriptors(APP_ROOT)


@st.cache_data(show_spinner=False)
def cached_metadata_tables(deployment_version: str):
    del deployment_version
    return load_model_performance(APP_ROOT), load_dataset_statistics(APP_ROOT)


@st.cache_data(show_spinner=False)
def cached_endpoint_source_map(deployment_version: str):
    del deployment_version
    return pd.read_csv(APP_ROOT / "metadata" / "endpoint_source_files.csv").fillna("")


@st.cache_data(show_spinner=False)
def cached_test_set_ad_coverage(deployment_version: str) -> pd.DataFrame:
    """Load verified aggregate test-set AD counts and derive endpoint coverage."""
    del deployment_version
    source_path = APP_ROOT / "metadata" / "applicability_domain_summary.csv"
    columns = ["model_id", "test_in_ad", "test_out_of_ad", "test_in_ad_coverage"]
    if not source_path.exists():
        return pd.DataFrame(columns=columns)

    source = pd.read_csv(source_path)
    required = {"model_id", "test_in_ad", "test_out_of_ad"}
    if not required.issubset(source.columns):
        return pd.DataFrame(columns=columns)

    coverage = source.loc[:, ["model_id", "test_in_ad", "test_out_of_ad"]].copy()
    coverage["model_id"] = coverage["model_id"].astype(str)
    coverage["test_in_ad"] = pd.to_numeric(coverage["test_in_ad"], errors="coerce")
    coverage["test_out_of_ad"] = pd.to_numeric(
        coverage["test_out_of_ad"], errors="coerce"
    )
    denominator = coverage["test_in_ad"] + coverage["test_out_of_ad"]
    coverage["test_in_ad_coverage"] = (
        100.0 * coverage["test_in_ad"] / denominator.where(denominator.gt(0))
    )
    return coverage.dropna(subset=["test_in_ad_coverage"]).reset_index(drop=True)


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="MammalTox 1.0 Results")
    return output.getvalue()


def compact_user_table(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Remove fully empty display-only rows and columns without changing source data."""
    selected = [column for column in (columns or list(frame.columns)) if column in frame.columns]
    display = frame.loc[:, selected].copy()
    for column in display.columns:
        if not pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].where(pd.notna(display[column]), "")
    nonempty_columns = [
        column
        for column in display.columns
        if display[column].astype(str).str.strip().ne("").any()
    ]
    display = display.loc[:, nonempty_columns]
    if display.empty:
        return display
    nonempty_rows = display.astype(str).apply(lambda column: column.str.strip().ne(""))
    return display.loc[nonempty_rows.any(axis=1)].reset_index(drop=True)


def render_descriptor_definition_table(descriptor_names: list[str]) -> None:
    """Render exact-order descriptor names with installed Mordred definitions."""
    descriptions = official_descriptor_descriptions(descriptor_names)
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(name))}</td>"
        f"<td>{html.escape(str(description))}</td>"
        "</tr>"
        for name, description in zip(descriptor_names, descriptions)
    )
    st.markdown(
        '<table class="descriptor-definition-table">'
        "<thead><tr><th>Mordred descriptor</th><th>Official description</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )


def hyperparameter_table(raw_parameters) -> pd.DataFrame:
    """Parse saved final parameters into a compact two-column display table."""
    parsed = raw_parameters
    if isinstance(raw_parameters, str):
        try:
            parsed = ast.literal_eval(raw_parameters)
        except (SyntaxError, ValueError):
            parsed = None
    if not isinstance(parsed, dict) or not parsed:
        return pd.DataFrame({"Parameter": ["Not available"], "Value": ["Not available"]})

    def readable_value(value) -> str:
        if value is None:
            return "Not available"
        if isinstance(value, (list, tuple, dict)):
            normalized = list(value) if isinstance(value, tuple) else value
            return json.dumps(normalized, ensure_ascii=True, sort_keys=isinstance(normalized, dict))
        return str(value)

    return pd.DataFrame(
        {
            "Parameter": [str(parameter) for parameter in parsed],
            "Value": [readable_value(value) for value in parsed.values()],
        }
    )


def final_model_performance_table(metric_summary: pd.DataFrame) -> pd.DataFrame:
    """Transpose stored training/test metrics without changing their values."""
    metric_order = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC-AUC"]
    transposed = (
        metric_summary.set_index("Subset")
        .reindex(index=["Training set", "Test set"], columns=metric_order)
        .transpose()
        .rename_axis("Metric")
        .reset_index()
    )
    transposed["Metric"] = transposed["Metric"].replace(
        {"F1": "F1 score", "ROC-AUC": "ROC–AUC"}
    )
    return transposed


def prediction_label_table(frame: pd.DataFrame):
    """Apply restrained browser-only color cues to predicted toxicity labels."""
    label_columns = ["Prediction"] if "Prediction" in frame.columns else []
    if frame.empty or not label_columns:
        return frame

    def label_style(value) -> str:
        normalized = str(value).strip().lower()
        if normalized == "low toxicity":
            return "background-color: #E8F0EA; color: #466452; border: 1px solid #C4D5C8;"
        if normalized == "high toxicity":
            return "background-color: #F3E6E3; color: #874E48; border: 1px solid #D9BBB6;"
        return ""

    def ad_style(value) -> str:
        if value == "In AD":
            return "background-color: #E8F0EA; color: #466452; border: 1px solid #C4D5C8;"
        if value == "Out of AD":
            return "background-color: #F4EBDD; color: #765B32; border: 1px solid #DDC9A5;"
        return ""

    styled = frame.style.map(label_style, subset=label_columns)
    if "AD Status" in frame.columns:
        styled = styled.map(ad_style, subset=["AD Status"])
    return styled


def model_display_order(performance_table: pd.DataFrame) -> list[str]:
    """Derive the presentation order from final deployed test-set performance."""
    if performance_table.empty or "species" not in performance_table.columns:
        return []
    ranking = performance_table.copy()
    for column in ["Test MCC", "Test Accuracy", "ROC-AUC", "Test F1"]:
        if column not in ranking.columns:
            ranking[column] = pd.NA
        ranking[column] = pd.to_numeric(ranking[column], errors="coerce")

    if ranking["Test F1"].isna().any() and {"Test Precision", "Test Recall"}.issubset(ranking.columns):
        precision = pd.to_numeric(ranking["Test Precision"], errors="coerce")
        recall = pd.to_numeric(ranking["Test Recall"], errors="coerce")
        derived_f1 = 2 * precision * recall / (precision + recall)
        ranking["Test F1"] = ranking["Test F1"].fillna(derived_f1)

    records = ranking.to_dict("records")

    def compare(left: dict, right: dict) -> int:
        for metric in ["Test MCC", "Test Accuracy"]:
            left_value = left.get(metric)
            right_value = right.get(metric)
            left_value = float(left_value) if pd.notna(left_value) else float("-inf")
            right_value = float(right_value) if pd.notna(right_value) else float("-inf")
            if left_value != right_value:
                return -1 if left_value > right_value else 1
        for metric in ["ROC-AUC", "Test F1"]:
            left_value = left.get(metric)
            right_value = right.get(metric)
            if pd.notna(left_value) and pd.notna(right_value) and float(left_value) != float(right_value):
                return -1 if float(left_value) > float(right_value) else 1
        left_species = str(left.get("species", ""))
        right_species = str(right.get("species", ""))
        return (left_species > right_species) - (left_species < right_species)

    return [str(record["species"]) for record in sorted(records, key=cmp_to_key(compare))]


def sort_species_for_display(
    frame: pd.DataFrame,
    performance_table: pd.DataFrame,
    species_column: str = "species",
) -> pd.DataFrame:
    """Apply the final metric-derived species order without changing model mappings."""
    if frame.empty or species_column not in frame.columns:
        return frame.copy()
    order = model_display_order(performance_table)
    display_order = {species: index for index, species in enumerate(order)}
    display = frame.copy()
    display["_species_display_order"] = (
        display[species_column].astype(str).map(display_order).fillna(len(display_order))
    )
    return (
        display.sort_values("_species_display_order", kind="stable")
        .drop(columns="_species_display_order")
    )


def retina_image_html(path: Path, css_class: str, alt_text: str) -> str:
    """Embed native PNG bytes so Streamlit does not resize the Retina source."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="{css_class}" src="data:image/png;base64,{encoded}" alt="{alt_text}">'


def header_brand_html(emblem_path: Path, wordmark_path: Path) -> str:
    """Compose original high-resolution artwork with responsive live version text."""
    emblem = retina_image_html(
        emblem_path, "mammaltox-header-emblem", "MammalTox mammal emblem"
    )
    wordmark = retina_image_html(
        wordmark_path, "mammaltox-header-wordmark", "MammalTox"
    )
    return (
        '<div class="mammaltox-header-brand">'
        f"{emblem}"
        '<div class="mammaltox-header-wordmark-block">'
        '<div class="mammaltox-header-wordmark-row">'
        f'{wordmark}<span class="mammaltox-header-version">1.0</span>'
        "</div>"
        '<div class="mammaltox-header-tagline">ML-based mammalian acute toxicity prediction</div>'
        "</div>"
        "</div>"
    )


def render_ad_information_card() -> None:
    """Render a compact explanation of the displayed applicability-domain status."""
    st.markdown(
        """
        <section class="ad-information-card" aria-labelledby="ad-information-title">
          <div class="ad-information-icon-block">
            <svg class="ad-information-icon" viewBox="0 0 64 64" role="img"
                 aria-label="k-nearest-neighbor distance-based applicability domain">
              <title>k-nearest-neighbor distance-based applicability domain</title>
              <circle cx="32" cy="32" r="27" fill="none" stroke="#d8c5a1"
                      stroke-width="1.25" stroke-dasharray="3.5 3.5" />
              <g fill="none" stroke="#c7d1d7" stroke-width="1.35" stroke-linecap="round">
                <line x1="32" y1="32" x2="32" y2="9" />
                <line x1="32" y1="32" x2="52" y2="20" />
                <line x1="32" y1="32" x2="48" y2="49" />
                <line x1="32" y1="32" x2="17" y2="52" />
                <line x1="32" y1="32" x2="10" y2="24" />
              </g>
              <circle class="knn-neighbor" cx="32" cy="9" r="3.5" fill="#5f8584" />
              <circle class="knn-neighbor" cx="52" cy="20" r="3.5" fill="#c39a58" />
              <circle class="knn-neighbor" cx="48" cy="49" r="3.5" fill="#5f8584" />
              <circle class="knn-neighbor" cx="17" cy="52" r="3.5" fill="#c39a58" />
              <circle class="knn-neighbor" cx="10" cy="24" r="3.5" fill="#5f8584" />
              <circle cx="32" cy="32" r="6.5" fill="#263d53" stroke="#ffffff" stroke-width="1.5" />
            </svg>
            <div class="ad-information-icon-label">k-NN</div>
          </div>
          <div class="ad-information-copy">
            <div class="ad-information-title" id="ad-information-title"><span><strong>k-NN</strong> Distance-Based Applicability Domain</span><span class="ad-method-badge">k-NN AD</span></div>
            <p><strong>Method:</strong> The query compound is compared with its <strong>5 nearest training compounds</strong> in the model-selected Mordred descriptor space, using the training-set <strong>95th percentile</strong> as the distance cutoff.</p>
            <p><strong>Interpretation:</strong> <strong>In AD</strong> means the compound is represented within the model’s training chemical space; <strong>Out of AD</strong> means the prediction should be interpreted with caution.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_local_shap_summary(
    results: pd.DataFrame,
    model_id: str,
    shap_data: dict,
    query_index: int,
) -> None:
    query_id = str(shap_data["query_ids"][query_index])
    matching = results[
        results["_Model_ID"].astype(str).eq(model_id)
        & results["Compound_ID"].astype(str).eq(query_id)
    ]
    predicted_class = "Not available"
    query_outputs = shap_data.get("query_outputs", shap_data.get("query_probabilities", []))
    predicted_output = float(query_outputs[query_index])
    output_space = str(shap_data.get("output_space", "probability"))
    ad_status = "Not available"
    if not matching.empty:
        row = matching.iloc[0]
        predicted_class = str(int(row["Predicted_Class"])) if pd.notna(row["Predicted_Class"]) else "Not available"
        ad_status = str(row.get("AD_Status", "Not available"))
    base_value = float(shap_data["base_values"][query_index])
    summary_columns = st.columns(4)
    summary_columns[0].metric("Predicted class", predicted_class)
    if output_space == "probability":
        summary_columns[1].metric("High toxicity probability", f"{predicted_output:.4f}")
        summary_columns[2].metric(
            "Baseline High toxicity probability", f"{base_value:.4f}"
        )
    else:
        summary_columns[1].metric("Predicted model score", f"{predicted_output:.4f}")
        summary_columns[2].metric("Baseline model score", f"{base_value:.4f}")
    summary_columns[3].metric("AD status", ad_status)


def selected_model_records(registry: pd.DataFrame, performance_table: pd.DataFrame) -> pd.DataFrame:
    deployed = sort_species_for_display(active_models(registry), performance_table)
    if deployed.empty:
        st.warning("No deployed models are available in metadata/model_registry.csv.")
        return deployed

    selection_mode = st.segmented_control(
        "Prediction target",
        ["All deployed endpoints", "Select one endpoint"],
        default="Select one endpoint",
        key="prediction_target_mode",
    )
    if selection_mode == "All deployed endpoints":
        return deployed

    labels = {
        f"{endpoint_display_label(row)} | {row['model_name']}": index for index, row in deployed.iterrows()
    }
    selected_label = st.selectbox("Endpoint and model", list(labels), key="prediction_model")
    return deployed.loc[[labels[selected_label]]]


def store_defined_compounds(compounds: pd.DataFrame) -> pd.DataFrame:
    validated = validate_compound_inputs(compounds)
    st.session_state.defined_compounds = validated
    st.session_state.prediction_bundle = None
    return validated


def input_summary(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"total": 0, "valid": 0, "failed": 0, "duplicates": 0}
    valid = int(frame["Input_Error"].fillna("").eq("").sum())
    return {
        "total": int(len(frame)),
        "valid": valid,
        "failed": int(len(frame) - valid),
        "duplicates": int(frame["Duplicate_Of"].fillna("").ne("").sum()),
    }


def render_defined_structures(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No structures have been defined yet.")
        return
    summary = input_summary(frame)
    columns = st.columns(4)
    columns[0].metric("Input records", summary["total"])
    columns[1].metric("Valid structures", summary["valid"])
    columns[2].metric("Invalid structures", summary["failed"])
    columns[3].metric("Duplicate structures", summary["duplicates"])
    preview_columns = ["Compound_ID", "Input_SMILES", "Canonical_SMILES", "Input_Status"]
    st.dataframe(
        compact_user_table(frame, preview_columns),
        hide_index=True,
        width="stretch",
    )
    if summary["failed"]:
        st.error(
            "The submitted SMILES could not be processed. Please provide a valid "
            "single-compound SMILES."
        )
    if summary["duplicates"]:
        st.warning("Duplicate canonical structures were detected and retained for prediction.")


def synchronize_pasted_smiles() -> None:
    """Load one valid pasted molecule into the shared Ketcher state."""
    text = str(st.session_state.get("pasted_smiles_input", ""))
    st.session_state.pasted_smiles_draft = text
    canonical, status = validate_single_smiles_input(text)
    st.session_state.structure_sync_status = status
    if status == "empty":
        if st.session_state.get("synchronized_smiles", ""):
            st.session_state.structure_editor_generation += 1
        st.session_state.synchronized_smiles = ""
        st.session_state.synchronized_molblock = ""
        st.session_state.current_canonical_smiles = ""
        st.session_state.synchronization_source = "text"
        st.session_state.last_smiles_loaded_into_editor = ""
        st.session_state.last_editor_smiles = ""
        st.session_state.last_valid_user_smiles = ""
        st.session_state.last_valid_editor_smiles = ""
        st.session_state.editor_programmatic_update_pending = False
        return
    if status != "valid":
        return
    st.session_state.last_valid_user_smiles = canonical
    st.session_state.current_canonical_smiles = canonical
    if (
        canonical == st.session_state.get("synchronized_smiles", "")
        and st.session_state.get("synchronized_molblock", "")
    ):
        return
    molblock, molblock_error = smiles_to_editor_molblock(canonical)
    if molblock_error:
        st.session_state.structure_sync_status = molblock_error
        return
    st.session_state.synchronized_smiles = canonical
    st.session_state.synchronized_molblock = molblock
    st.session_state.synchronization_source = "text"
    st.session_state.editor_programmatic_update_pending = True
    st.session_state.structure_editor_generation += 1


def clear_synchronized_structure() -> None:
    """Clear only the active single-compound draft and its editor instance."""
    st.session_state.synchronized_smiles = ""
    st.session_state.synchronized_molblock = ""
    st.session_state.current_canonical_smiles = ""
    st.session_state.pasted_smiles_input = ""
    st.session_state.pasted_smiles_draft = ""
    st.session_state.synchronization_source = "editor"
    st.session_state.last_smiles_loaded_into_editor = ""
    st.session_state.last_editor_smiles = ""
    st.session_state.last_valid_user_smiles = ""
    st.session_state.last_valid_editor_smiles = ""
    st.session_state.structure_sync_status = "empty"
    st.session_state.editor_programmatic_update_pending = False
    st.session_state.structure_editor_generation += 1


def synchronize_before_structure_mode_switch() -> None:
    """Preserve the outgoing mode's draft without mixing single and batch state."""
    previous_mode = st.session_state.get("last_structure_input_mode", "Single compound")
    if previous_mode == "Single compound" and "pasted_smiles_input" in st.session_state:
        synchronize_pasted_smiles()
        st.session_state.single_compound_identifier_draft = str(
            st.session_state.get("single_compound_identifier", "")
        )
    elif previous_mode == "Batch input" and "batch_smiles_input" in st.session_state:
        st.session_state.batch_smiles_draft = str(
            st.session_state.get("batch_smiles_input", "")
        )
    st.session_state.last_structure_input_mode = st.session_state.get(
        "structure_input_mode", previous_mode
    )


def preserve_batch_smiles_draft() -> None:
    st.session_state.batch_smiles_draft = str(
        st.session_state.get("batch_smiles_input", "")
    )


def preserve_single_identifier() -> None:
    st.session_state.single_compound_identifier_draft = str(
        st.session_state.get("single_compound_identifier", "")
    )


def structure_sync_indicator_html(synchronized: bool) -> str:
    """Render a compact vector indicator without participating in synchronization state."""
    label = '<span class="structure-sync-label">Synced</span>' if synchronized else ""
    upper_arrow = "#b88a43" if synchronized else "#9aa3a6"
    lower_arrow = "#557d79" if synchronized else "#9aa3a6"
    return f"""
    <div class="single-structure-workspace-marker structure-sync-indicator"
         title="Bidirectional structure–SMILES synchronization">
      <svg class="structure-sync-icon" viewBox="0 0 48 48"
           preserveAspectRatio="xMidYMid meet" role="img"
           aria-label="Bidirectional structure–SMILES synchronization"
           xmlns="http://www.w3.org/2000/svg">
        <title>Bidirectional structure–SMILES synchronization</title>
        <circle cx="24" cy="24" r="21" fill="none" stroke="#e1e6e5"
                stroke-width="1.2"/>
        <path d="M12 18H34M29.5 13.5L34 18L29.5 22.5" fill="none"
              stroke="{upper_arrow}" stroke-width="2.25" stroke-linecap="round"
              stroke-linejoin="round"/>
        <path d="M36 30H14M18.5 25.5L14 30L18.5 34.5" fill="none"
              stroke="{lower_arrow}" stroke-width="2.25" stroke-linecap="round"
              stroke-linejoin="round"/>
      </svg>
      {label}
    </div>
    """


def render_structure_drawer() -> tuple[str, str | None]:
    try:
        from streamlit_ketcher import st_ketcher
    except Exception:
        st.error("The structure editor is unavailable because streamlit-ketcher is not installed.")
        return "", "The structure editor is unavailable"
    initial_smiles = str(st.session_state.get("synchronized_smiles", ""))
    initial_molblock = str(st.session_state.get("synchronized_molblock", ""))
    editor_generation = int(st.session_state.get("structure_editor_generation", 0))
    editor_key = f"mammaltox_structure_editor_{editor_generation}"
    try:
        drawn = st_ketcher(
            initial_molblock,
            height=500,
            molecule_format="MOLFILE",
            key=editor_key,
        )
    except Exception as exc:
        st.error(f"Structure drawing is unavailable in this environment: {exc}")
        return "", "Structure drawing is unavailable in this environment"

    editor_value = drawn if isinstance(drawn, str) else ""
    if not editor_value.strip():
        return "", None

    canonical, error = editor_molblock_to_smiles(editor_value)
    if not canonical:
        return "", error or "Drawn structure could not be converted to a valid molecule"

    st.session_state.last_smiles_loaded_into_editor = initial_smiles
    st.session_state.editor_programmatic_update_pending = False
    if canonical != st.session_state.get("last_editor_smiles", ""):
        st.session_state.last_editor_smiles = canonical
        st.session_state.last_valid_editor_smiles = canonical
    if canonical != st.session_state.get("synchronized_smiles", ""):
        st.session_state.synchronized_smiles = canonical
        st.session_state.synchronized_molblock = editor_value
        st.session_state.current_canonical_smiles = canonical
        st.session_state.pasted_smiles_input = canonical
        st.session_state.pasted_smiles_draft = canonical
        st.session_state.synchronization_source = "editor"
        st.session_state.structure_sync_status = "valid"
    return canonical, None


registry, selected_descriptors = cached_registry_resources(DEPLOYMENT_CACHE_VERSION)
performance, dataset_stats = cached_metadata_tables(DEPLOYMENT_CACHE_VERSION)
endpoint_sources = cached_endpoint_source_map(DEPLOYMENT_CACHE_VERSION)
test_set_ad_coverage = cached_test_set_ad_coverage(DEPLOYMENT_CACHE_VERSION)
st.session_state.setdefault("defined_compounds", pd.DataFrame(columns=COMPOUND_COLUMNS))
st.session_state.setdefault("prediction_bundle", None)
st.session_state.setdefault("pasted_smiles_input", "")
st.session_state.setdefault("pasted_smiles_draft", st.session_state.pasted_smiles_input)
st.session_state.setdefault("synchronized_smiles", "")
st.session_state.setdefault("synchronized_molblock", "")
st.session_state.setdefault("current_canonical_smiles", st.session_state.synchronized_smiles)
st.session_state.setdefault("synchronization_source", "")
st.session_state.setdefault("last_smiles_loaded_into_editor", "")
st.session_state.setdefault("last_editor_smiles", "")
st.session_state.setdefault("last_valid_user_smiles", "")
st.session_state.setdefault("last_valid_editor_smiles", "")
st.session_state.setdefault("structure_editor_generation", 0)
st.session_state.setdefault("structure_sync_status", "empty")
st.session_state.setdefault("editor_programmatic_update_pending", False)
st.session_state.setdefault("structure_input_mode", "Single compound")
st.session_state.setdefault("last_structure_input_mode", "Single compound")
st.session_state.setdefault("single_compound_identifier", "")
st.session_state.setdefault("single_compound_identifier_draft", "")
st.session_state.setdefault("batch_smiles_input", "")
st.session_state.setdefault("batch_smiles_draft", "")

if HEADER_EMBLEM_PATH.exists() and HEADER_WORDMARK_PATH.exists():
    st.markdown(
        header_brand_html(HEADER_EMBLEM_PATH, HEADER_WORDMARK_PATH),
        unsafe_allow_html=True,
    )
elif HEADER_LOGO_PATH.exists():
    st.image(HEADER_LOGO_PATH, width=700)
else:
    st.markdown(
        '<h1>MammalTox <span class="mammaltox-live-title-version">1.0</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mammaltox-subtitle">ML-based mammalian acute toxicity prediction</div>',
        unsafe_allow_html=True,
    )

tabs = st.tabs(
    [
        "Define Structure",
        "Prediction",
        "Results",
        "Model Information",
        "Data Statistics",
        "About / Help / Contact",
    ]
)

with tabs[0]:
    st.subheader("Define Structure")
    input_mode = st.segmented_control(
        "Structure input mode",
        ["Single compound", "Batch input"],
        key="structure_input_mode",
        on_change=synchronize_before_structure_mode_switch,
    )

    if input_mode == "Single compound":
        st.session_state.pasted_smiles_input = str(
            st.session_state.get("pasted_smiles_draft", "")
        )
        st.session_state.single_compound_identifier = str(
            st.session_state.get("single_compound_identifier_draft", "")
        )
        drawing_panel, sync_panel, smiles_panel = st.columns(
            [64, 6, 30], gap="small", vertical_alignment="top"
        )
        with drawing_panel:
            with st.container(border=True):
                st.markdown(
                    '<div class="single-structure-panel-heading">Draw or edit structure</div>',
                    unsafe_allow_html=True,
                )
                drawn_smiles, drawing_error = render_structure_drawer()
                st.caption(
                    "Draw or edit one compound, then select Apply in the editor to update SMILES."
                )
                st.button(
                    "Clear structure",
                    icon=":material/delete_sweep:",
                    on_click=clear_synchronized_structure,
                    key="clear_single_structure",
                )
                if drawing_error:
                    st.error(
                        "The drawn structure could not be processed. Complete or clear the structure and try again."
                    )

        with sync_panel:
            synchronized = bool(
                st.session_state.get("structure_sync_status") == "valid"
                and st.session_state.get("synchronized_smiles")
                and st.session_state.get("current_canonical_smiles")
                == st.session_state.get("synchronized_smiles")
            )
            st.markdown(
                structure_sync_indicator_html(synchronized),
                unsafe_allow_html=True,
            )

        with smiles_panel:
            with st.container(border=True):
                st.markdown(
                    '<div class="single-structure-panel-heading">SMILES</div>',
                    unsafe_allow_html=True,
                )
                single_smiles = st.text_input(
                    "SMILES",
                    placeholder="CCO",
                    help=(
                        "Enter any valid, parseable SMILES, including canonical, non-canonical, "
                        "isomeric, aromatic, or Kekulé SMILES. Canonical SMILES is not required. "
                        "For best compatibility, submit a single organic compound; salts, mixtures, "
                        "metals, and disconnected fragments may be rejected during preprocessing."
                    ),
                    key="pasted_smiles_input",
                    on_change=synchronize_pasted_smiles,
                )
                st.caption("Enter one valid SMILES to load the corresponding structure.")
                compound_identifier = st.text_input(
                    "Compound ID or name (optional)",
                    placeholder="Compound_001",
                    key="single_compound_identifier",
                    on_change=preserve_single_identifier,
                )

                canonical_smiles, synchronization_status = validate_single_smiles_input(
                    single_smiles
                )
                if synchronization_status == "valid":
                    st.markdown(
                        '<div class="structure-valid-status"><span class="structure-valid-dot"></span>'
                        "Valid single compound</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Canonical SMILES**")
                    st.code(canonical_smiles, language="text")
                elif synchronization_status == "multiple":
                    st.error("Enter exactly one SMILES in Single compound mode.")
                elif synchronization_status != "empty":
                    st.error(
                        "The submitted SMILES could not be processed. Please provide a valid "
                        "single-compound SMILES."
                    )

                if st.button(
                    "Use this compound",
                    type="primary",
                    disabled=synchronization_status != "valid",
                    key="confirm_single_compound",
                ):
                    synchronize_pasted_smiles()
                    identifier = str(compound_identifier or "").strip() or "Compound_001"
                    store_defined_compounds(
                        pd.DataFrame(
                            [
                                {
                                    "Compound_ID": identifier,
                                    "Compound_Name": str(compound_identifier or "").strip(),
                                    "Input_SMILES": str(single_smiles).strip(),
                                    "Source_Error": "",
                                }
                            ]
                        )
                    )

    else:
        st.session_state.batch_smiles_input = str(
            st.session_state.get("batch_smiles_draft", "")
        )
        paste_panel, upload_panel = st.columns(
            2, gap="medium", vertical_alignment="top"
        )
        with paste_panel:
            st.markdown(
                '<span class="structure-responsive-marker" aria-hidden="true"></span>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown("#### Paste multiple SMILES")
                batch_text = st.text_area(
                    "Batch SMILES",
                    placeholder="CCO Compound_001\nCC(=O)O Compound_002",
                    help="Enter one SMILES per line. An optional compound ID or name may follow each SMILES.",
                    height=180,
                    key="batch_smiles_input",
                    on_change=preserve_batch_smiles_draft,
                )
                st.caption("Enter one SMILES per line with an optional compound ID or name.")
                if str(batch_text).strip():
                    batch_preview = validate_compound_inputs(
                        parse_smi_text(batch_text, "Compound")
                    )
                    batch_summary = input_summary(batch_preview)
                    st.caption(
                        f"{batch_summary['total']} records · {batch_summary['valid']} valid · "
                        f"{batch_summary['failed']} invalid · {batch_summary['duplicates']} duplicates"
                    )
                    st.dataframe(
                        compact_user_table(
                            batch_preview,
                            ["Compound_ID", "Input_SMILES", "Canonical_SMILES", "Input_Status"],
                        ),
                        hide_index=True,
                        width="stretch",
                        height=min(260, 38 + 35 * len(batch_preview)),
                    )
                if st.button(
                    "Use pasted batch",
                    type="primary",
                    disabled=not str(batch_text).strip(),
                    key="confirm_pasted_batch",
                ):
                    preserve_batch_smiles_draft()
                    store_defined_compounds(parse_smi_text(batch_text, "Compound"))

        with upload_panel:
            with st.container(border=True):
                st.markdown("#### Upload Input File")
                st.caption(
                    "CSV/XLSX files require a SMILES column. SMI/TXT files use one SMILES per line; "
                    "SDF files may contain multiple molecular records."
                )
                example = example_preview_frame()
                st.markdown("**Example input preview**")
                st.dataframe(example, hide_index=True, width="stretch")
                st.download_button(
                    "Download Example Input File",
                    data=example_input_bytes(),
                    file_name="MammalTox_1.0_example_input.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                uploaded = st.file_uploader(
                    "Upload Input File",
                    type=["csv", "xlsx", "smi", "txt", "sdf"],
                    key="structure_upload",
                )
                if uploaded is not None:
                    signature = (uploaded.name, uploaded.size)
                    if st.session_state.get("last_upload_signature") != signature:
                        store_defined_compounds(
                            parse_uploaded_file(uploaded.name, uploaded.getvalue())
                        )
                        st.session_state.last_upload_signature = signature

    st.markdown("### Current defined structures")
    render_defined_structures(st.session_state.defined_compounds)
    if not st.session_state.defined_compounds.empty and st.button(
        "Clear defined structures", icon=":material/delete_sweep:"
    ):
        st.session_state.defined_compounds = pd.DataFrame(columns=COMPOUND_COLUMNS)
        st.session_state.prediction_bundle = None
        st.session_state.pop("last_upload_signature", None)
        st.rerun()

with tabs[1]:
    st.subheader("Prediction")
    compounds = st.session_state.defined_compounds
    summary = input_summary(compounds)
    if summary["total"] == 0:
        st.info("Define or upload structures in the Define Structure tab before prediction.")
    else:
        mode = "Single prediction" if summary["valid"] == 1 else "Batch prediction"
        st.caption(
            f"Detected workflow: {mode}. {summary['total']} input record(s), "
            f"{summary['valid']} valid structure(s), and {summary['failed']} invalid structure(s)."
        )
        selected_models = selected_model_records(registry, performance)
        st.caption(
            "MammalTox 1.0 validates and standardizes input structures with RDKit, calculates the "
            "endpoint-specific Mordred descriptors used during training, aligns the required features, "
            "and applies the saved model."
        )
        if st.button("Run prediction", type="primary", disabled=summary["valid"] == 0 or selected_models.empty):
            with st.status("Running MammalTox 1.0 prediction", expanded=True) as status:
                st.write("Canonicalizing structures and calculating Mordred descriptors...")
                bundle = predict_compounds(
                    compounds,
                    None,
                    selected_models,
                    selected_descriptors,
                    APP_ROOT,
                    include_explanations=True,
                )
                st.write("Calculating applicability domain, PCA projection, and SHAP explanations...")
                st.session_state.prediction_bundle = bundle
                status.update(label="Prediction complete", state="complete", expanded=False)
            st.success("Prediction completed. Open the Results tab to review outputs and explanations.")

with tabs[2]:
    st.subheader("Results")
    bundle: PredictionBundle | None = st.session_state.prediction_bundle
    if bundle is None:
        st.info("Run a prediction to display results.")
    else:
        sorted_results = sort_species_for_display(bundle.results, performance, "Species")
        technical_output = technical_output_table(sorted_results)
        browser_output = main_results_table(sorted_results)
        summary = bundle.input_summary
        metrics = st.columns(4)
        metrics[0].metric("Uploaded records", summary.get("uploaded", 0))
        metrics[1].metric("Valid structures", summary.get("valid", 0))
        metrics[2].metric("Predicted compounds", summary.get("predicted", 0))
        metrics[3].metric("Failed compounds", summary.get("failed", 0))
        successful_rows = pd.to_numeric(
            sorted_results["Predicted_Class"], errors="coerce"
        ).notna()
        genuine_failures = browser_output.loc[
            ~successful_rows,
            ["Compound ID", "Species", "Model", "Error Message"],
        ]
        if genuine_failures.empty:
            st.success("All selected endpoint predictions completed successfully.")
        else:
            st.warning(
                f"{len(genuine_failures)} endpoint prediction(s) could not be completed. "
                "Successful results remain available below."
            )
        st.caption(
            "Predicted-class probability is the probability assigned by the model to the displayed "
            "predicted class; it is a model output, not experimental certainty."
        )
        st.dataframe(
            prediction_label_table(browser_output),
            hide_index=True,
            width="stretch",
            column_config={
                "Compound ID": st.column_config.TextColumn(width="medium"),
                "Input SMILES": st.column_config.TextColumn(width="large"),
                "Species": st.column_config.TextColumn(width="small"),
                "Route": st.column_config.TextColumn(width="medium"),
                "Endpoint": st.column_config.TextColumn(width="small"),
                "Model": st.column_config.TextColumn(width="medium"),
                "Prediction": st.column_config.TextColumn(width="small"),
                "Predicted-class probability": st.column_config.TextColumn(width="large"),
                "AD Status": st.column_config.TextColumn(
                    width="medium",
                    help=(
                        "Applicability-domain status indicates whether the query lies within the "
                        "chemical-space coverage of the endpoint model's training data."
                    ),
                ),
                "Error Message": st.column_config.TextColumn(width="large"),
            },
        )
        render_ad_information_card()

        if not genuine_failures.empty:
            st.markdown("**Prediction failures**")
            st.dataframe(
                genuine_failures,
                hide_index=True,
                width="stretch",
                height=min(36 * (len(genuine_failures) + 1) + 3, 260),
            )

        downloads = st.columns(2)
        with downloads[0]:
            st.download_button(
                "Download Prediction Results",
                data=technical_output.to_csv(index=False).encode("utf-8"),
                file_name="MammalTox_1.0_prediction_results.csv",
                mime="text/csv",
                icon=":material/download:",
            )
        with downloads[1]:
            st.download_button(
                "Download Excel results",
                data=to_excel_bytes(technical_output),
                file_name="MammalTox_1.0_prediction_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:",
            )

        deployed = sort_species_for_display(active_models(registry), performance).set_index("model_id", drop=False)
        artifact_model_ids = [
            model_id for model_id in deployed.index if model_id in bundle.pca_artifacts
        ]
        if artifact_model_ids:
            artifact_labels = {
                f"{endpoint_display_label(deployed.loc[model_id])} | {deployed.loc[model_id, 'model_name']}": model_id
                for model_id in artifact_model_ids
            }
            selected_artifact_label = st.selectbox(
                "Visualization endpoint", list(artifact_labels), key="visualization_endpoint"
            )
            selected_model_id = artifact_labels[selected_artifact_label]
            model_row = deployed.loc[selected_model_id]

            st.markdown(
                '<div class="shap-view-title pca-view-title">'
                "PCA Chemical-Space Visualization</div>",
                unsafe_allow_html=True,
            )
            pca_data = bundle.pca_artifacts.get(selected_model_id, {})
            if pca_data.get("error"):
                st.error(pca_data["error"])
            else:
                pca_figure = pca_chemical_space_figure(pca_data, endpoint_display_label(model_row))
                if pca_figure is None:
                    st.error("PCA projection failed because the visualization artifact is incomplete")
                else:
                    st.plotly_chart(
                        pca_figure,
                        width="stretch",
                        config={"displayModeBar": False, "responsive": True},
                    )

            shap_data = bundle.shap_artifacts.get(selected_model_id, {})
            if shap_data.get("error"):
                st.error(shap_data["error"])
            else:
                if shap_data.get("saturation_warning"):
                    st.warning(shap_data["saturation_warning"])
                if len(shap_data["query_ids"]) == 1:
                    st.markdown(
                        '<div class="shap-view-title">Individual Waterfall Plot</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Positive SHAP contributions increase the High toxicity "
                        "prediction; negative contributions decrease it."
                    )
                    render_local_shap_summary(
                        bundle.results, selected_model_id, shap_data, 0
                    )
                    shap_figure = local_shap_figure(shap_data)
                    st.pyplot(
                        shap_figure,
                        clear_figure=True,
                        width="stretch",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(shap_figure)
                    if "TreeExplainer was incompatible" in str(shap_data.get("fallback_reason", "")):
                        st.caption(
                            "TreeExplainer is unavailable for this saved model in the current library "
                            "environment; PermutationExplainer is used."
                        )
                else:
                    st.markdown(
                        '<div class="shap-view-title">Batch SHAP Summary</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Positive SHAP contributions increase the High toxicity "
                        "prediction; negative contributions decrease it."
                    )
                    st.caption(
                        f"Compounds explained: "
                        f"{shap_data.get('explained_query_count', len(shap_data['query_ids']))}"
                    )
                    shap_figure = batch_shap_figure(shap_data)
                    st.pyplot(
                        shap_figure,
                        clear_figure=True,
                        width="stretch",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(shap_figure)
                    if "TreeExplainer was incompatible" in str(shap_data.get("fallback_reason", "")):
                        st.caption(
                            "TreeExplainer is unavailable for this saved model in the current library "
                            "environment; PermutationExplainer is used."
                        )
                    st.markdown(
                        '<div class="shap-view-title">Individual Waterfall Plot</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Positive SHAP contributions increase the High toxicity "
                        "prediction; negative contributions decrease it."
                    )
                    selected_query = st.selectbox(
                        "Individual compound explanation",
                        shap_data["query_ids"],
                        key=f"local_shap_{selected_model_id}",
                    )
                    selected_index = shap_data["query_ids"].index(selected_query)
                    render_local_shap_summary(
                        bundle.results, selected_model_id, shap_data, selected_index
                    )
                    local_figure = local_shap_figure(shap_data, selected_index)
                    st.pyplot(
                        local_figure,
                        clear_figure=True,
                        width="stretch",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(local_figure)

with tabs[3]:
    st.subheader("Model information")
    deployed_registry = sort_species_for_display(active_models(registry), performance)
    if deployed_registry.empty:
        st.info("No deployed model records are available.")
    else:
        endpoint_options = {
            f"{endpoint_display_label(row)} | {row['model_name']}": index
            for index, row in deployed_registry.iterrows()
        }
        selected_endpoint = st.selectbox(
            "Deployed endpoint",
            list(endpoint_options),
            key="model_information_endpoint",
        )
        model_row = deployed_registry.loc[endpoint_options[selected_endpoint]]
        model_id = str(model_row["model_id"])
        performance_rows = performance[performance["model_id"].astype(str).eq(model_id)]
        statistics_rows = dataset_stats[
            dataset_stats["species"].astype(str).eq(str(model_row["species"]))
        ]
        if performance_rows.empty:
            st.error("Final model-performance metadata are unavailable for this endpoint.")
        elif statistics_rows.empty:
            st.error("Endpoint dataset statistics are unavailable for this endpoint.")
        else:
            performance_row = performance_rows.iloc[0]
            descriptor_count = len(selected_descriptors.get(model_id, []))
            source_rows = endpoint_sources[
                endpoint_sources["model_id"].astype(str).eq(model_id)
            ]
            if source_rows.empty:
                st.error("Endpoint source-file mapping is unavailable for this deployed model.")
                performance_data = {
                    "error": "Endpoint source-file mapping is unavailable",
                    "availability": pd.DataFrame(
                        [
                            {
                                "Visualization": "Model performance visualizations",
                                "Status": "Unavailable",
                                "Notes": "Performance visualization data are unavailable; prediction remains available.",
                            }
                        ]
                    ),
                }
            else:
                source_row = source_rows.iloc[0]
                performance_data = prepare_model_performance_data(
                    model_id,
                    str(model_row["model_path"]),
                    "" if pd.isna(model_row.get("preprocessor_path", "")) else str(model_row.get("preprocessor_path", "")),
                    str(resolve_app_path(str(source_row["selected_train_file"]), APP_ROOT)),
                    str(resolve_app_path(str(source_row["full_test_file"]), APP_ROOT)),
                    str(resolve_app_path(str(source_row["full_dataset_file"]), APP_ROOT)),
                    str(resolve_app_path(str(source_row["tuning_results_file"]), APP_ROOT)),
                    str(resolve_app_path(str(source_row["performance_results_file"]), APP_ROOT)),
                    str(resolve_app_path(str(source_row["fiore_output_file"]), APP_ROOT)),
                    tuple(selected_descriptors.get(model_id, [])),
                    str(APP_ROOT),
                    DEPLOYMENT_CACHE_VERSION,
                )

            if performance_data.get("error"):
                st.error(performance_data["error"])
                training_count = int(performance_row["training_compound_count"])
                test_count = int(performance_row["test_compound_count"])
                metric_summary = pd.DataFrame(
                    {
                        "Subset": ["Training set", "Test set"],
                        "Accuracy": [performance_row["Training Accuracy"], performance_row["Test Accuracy"]],
                        "Precision": [performance_row["Training Precision"], performance_row["Test Precision"]],
                        "Recall": [performance_row["Training Recall"], performance_row["Test Recall"]],
                        "F1": [performance_row.get("Training F1", pd.NA), performance_row.get("Test F1", pd.NA)],
                        "MCC": [performance_row["Training MCC"], performance_row["Test MCC"]],
                        "ROC-AUC": [pd.NA, performance_row.get("ROC-AUC", pd.NA)],
                    }
                )
            else:
                training_count = int(performance_data["train_count"])
                test_count = int(performance_data["test_count"])
                metrics = performance_data["metric_data"].set_index("Metric")
                if "F1" in metrics.index:
                    training_f1 = metrics.loc["F1", "Training"]
                    test_f1 = metrics.loc["F1", "Test"]
                else:
                    training_precision = float(metrics.loc["Precision", "Training"])
                    training_recall = float(metrics.loc["Recall", "Training"])
                    test_precision = float(metrics.loc["Precision", "Test"])
                    test_recall = float(metrics.loc["Recall", "Test"])
                    training_f1 = 2 * training_precision * training_recall / (
                        training_precision + training_recall
                    )
                    test_f1 = 2 * test_precision * test_recall / (
                        test_precision + test_recall
                    )
                metric_summary = pd.DataFrame(
                    {
                        "Subset": ["Training set", "Test set"],
                        "Accuracy": [metrics.loc["Accuracy", "Training"], metrics.loc["Accuracy", "Test"]],
                        "Precision": [metrics.loc["Precision", "Training"], metrics.loc["Precision", "Test"]],
                        "Recall": [metrics.loc["Recall", "Training"], metrics.loc["Recall", "Test"]],
                        "F1": [training_f1, test_f1],
                        "MCC": [metrics.loc["MCC", "Training"], metrics.loc["MCC", "Test"]],
                        "ROC-AUC": [performance_data.get("train_roc_auc", pd.NA), performance_data["roc"]["auc"]],
                    }
                )

            coverage_rows = test_set_ad_coverage[
                test_set_ad_coverage["model_id"].astype(str).eq(model_id)
            ]
            coverage_display = (
                f"{float(coverage_rows.iloc[0]['test_in_ad_coverage']):.2f}%"
                if len(coverage_rows) == 1
                else "Not available"
            )
            summary_columns = st.columns(4)
            with summary_columns[0]:
                st.markdown(
                    '<span class="model-summary-metrics-marker"></span>',
                    unsafe_allow_html=True,
                )
                st.metric("Training compounds", training_count)
            summary_columns[1].metric("Test compounds", test_count)
            summary_columns[2].metric("Selected Mordred descriptors", descriptor_count)
            summary_columns[3].metric("Test-set In-AD coverage", coverage_display)

            preprocessor_path = "" if pd.isna(model_row.get("preprocessor_path", "")) else str(model_row.get("preprocessor_path", "")).strip()
            preprocessing_method = performance_data.get(
                "preprocessing_method",
                (
                    "Training-derived standardization"
                    if preprocessor_path
                    else "No separate preprocessing step"
                ),
            )
            trial_count = (
                len(performance_data.get("tuning", []))
                if not performance_data.get("error")
                else "Not available"
            )
            prediction_output = performance_data.get("prediction_output", "Not available")
            model_summary = pd.DataFrame(
                {
                    "Field": [
                        "Species",
                        "Route",
                        "Endpoint",
                        "Model",
                        "Prediction output",
                        "Preprocessing method",
                        "Optuna trials",
                    ],
                    "Value": [
                        str(model_row["species"]),
                        display_route(model_row["route"]),
                        display_endpoint(model_row["endpoint"]),
                        str(model_row["model_name"]),
                        prediction_output,
                        preprocessing_method,
                        trial_count,
                    ],
                }
            )
            model_summary["Value"] = model_summary["Value"].astype(str)
            performance_display = final_model_performance_table(metric_summary)
            details_left, details_right = st.columns(
                [44, 56], gap="medium", vertical_alignment="top"
            )
            with details_left:
                with st.container(border=True):
                    st.markdown(
                        '<div class="model-information-section-heading '
                        'model-information-tables-marker">Model summary</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        model_summary,
                        hide_index=True,
                        width="stretch",
                        height=36 * (len(model_summary) + 1) + 3,
                        column_config={
                            "Field": st.column_config.TextColumn(width=165),
                            "Value": st.column_config.TextColumn(width=300),
                        },
                    )
            with details_right:
                with st.container(border=True):
                    st.markdown(
                        '<div class="model-information-section-heading">'
                        "Final model performance</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        performance_display,
                        hide_index=True,
                        width="stretch",
                        height=36 * (len(performance_display) + 1) + 3,
                        column_config={
                            "Metric": st.column_config.TextColumn(width="medium"),
                            "Training set": st.column_config.NumberColumn(
                                format="%.4f", width="medium"
                            ),
                            "Test set": st.column_config.NumberColumn(
                                format="%.4f", width="medium"
                            ),
                        },
                    )
                    st.caption(
                        "Training- and test-set performance is reported using Accuracy, Precision, "
                        "Recall, F1 score, MCC, and ROC–AUC."
                    )

            with st.expander("Selected Mordred descriptors and final hyperparameters"):
                descriptor_names = selected_descriptors.get(model_id, [])
                render_descriptor_definition_table(descriptor_names)
                st.caption(
                    "Descriptor definitions are obtained from the installed Mordred molecular "
                    "descriptor package."
                )
                st.markdown("**Final hyperparameters**")
                parameters = hyperparameter_table(
                    performance_row.get("best_parameters", "Not available")
                )
                st.dataframe(
                    parameters,
                    hide_index=True,
                    width="stretch",
                    height=min(36 * (len(parameters) + 1) + 3, 320),
                    column_config={
                        "Parameter": st.column_config.TextColumn(width="medium"),
                        "Value": st.column_config.TextColumn(width="large"),
                    },
                )

            st.caption(
                "Applicability domain method: Endpoint-specific 5-nearest-neighbor distance in the "
                "model-selected Mordred descriptor space."
            )

            st.markdown("### Final model performance visualizations")
            if not performance_data.get("error"):
                evaluation_figure = final_model_evaluation_figure(
                    performance_data,
                    endpoint_display_label(model_row),
                    str(model_row["model_name"]),
                )
                st.plotly_chart(
                    evaluation_figure,
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )
                st.caption("Rows show the true classes, and columns show the predicted classes.")
                development_figure = final_model_development_figure(
                    performance_data,
                    endpoint_display_label(model_row),
                    str(model_row["model_name"]),
                )
                st.plotly_chart(
                    development_figure,
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )

                st.markdown("#### Data source summary")
                st.dataframe(
                    performance_data["source_summary"],
                    hide_index=True,
                    width="stretch",
                )

            st.markdown("#### Visualization availability")
            st.dataframe(
                performance_data["availability"],
                hide_index=True,
                width="stretch",
                column_config={
                    "Visualization": st.column_config.TextColumn(width="medium"),
                    "Status": st.column_config.TextColumn(width="small"),
                    "Notes": st.column_config.TextColumn(width="large"),
                },
            )

with tabs[4]:
    st.subheader("Full curated endpoint dataset statistics")
    st.caption(
        "Counts represent each endpoint's final clear-class curated dataset used for model development, "
        "not the current prediction batch or a training-only subset."
    )
    if dataset_stats.empty:
        st.info("Dataset statistics metadata is not available.")
    else:
        metric_columns = st.columns(3)
        metric_columns[0].metric("Species endpoints", dataset_stats["species"].nunique())
        metric_columns[1].metric("Curated compounds", int(dataset_stats["total_compound_count"].sum()))
        metric_columns[2].metric("Deployed models", int(active_models(registry).shape[0]))
        display_dataset_stats = sort_species_for_display(dataset_stats, performance)
        percentage_table = class_distribution_percentage_table(display_dataset_stats)
        statistics_display_table = percentage_table.rename(
            columns={
                "species": "Species",
                "total_count": "Total compounds",
                "non_toxic_count": "Low toxicity count",
                "toxic_count": "High toxicity count",
                "non_toxic_percent": "Low toxicity (%)",
                "toxic_percent": "High toxicity (%)",
            }
        )
        st.dataframe(statistics_display_table, hide_index=True, width="stretch")
        figure = class_distribution_figure(display_dataset_stats)
        if figure is not None:
            st.plotly_chart(
                figure,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
        else:
            st.info("Class distribution visualization is unavailable.")

with tabs[5]:
    st.subheader("About MammalTox 1.0")
    st.write(
        "MammalTox 1.0 is an ML-based web application for species-specific prediction of mammalian acute intravenous toxicity."
    )
    st.write(
        "MammalTox 1.0 validates and standardizes input structures with RDKit, calculates the "
        "endpoint-specific Mordred descriptors used during training, aligns the required features, "
        "and applies the saved model."
    )
    st.write(
        "Prediction classes are presented as Low toxicity and High toxicity."
    )
    st.markdown(
        """
        **Contributors and contact**

        **Lihui Xin**  
        Research email: [lihuixin22@gmail.com](mailto:lihuixin22@gmail.com)  
        Institutional email: [xinl@kean.edu](mailto:xinl@kean.edu)

        **Dr. Supratik Kar**  
        Email: [skar@kean.edu](mailto:skar@kean.edu)

        **Affiliation**
        """
    )
    st.markdown(
        '<div class="mammaltox-affiliation">Chemometrics &amp; Molecular Modeling Laboratory, Department of Chemistry and Physics, Kean University, New Jersey, USA</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Applicability domain**  
        Endpoint-specific AD uses the exact selected Mordred descriptors with training-derived preprocessing. The query's mean distance to its five nearest training neighbors is compared with the 95th percentile of training mean-neighbor distances. Results are reported as `In AD` or `Out of AD`.
        """
    )
    st.warning(
        "Predictions are computational estimates and should not be used as the sole basis for experimental, clinical, regulatory, or safety decisions."
    )
