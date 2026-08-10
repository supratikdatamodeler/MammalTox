"""Shared Mordred descriptor and prediction engine for MammalTox."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd

from .applicability_domain import applicability_domain_details
from .chemical_space import load_training_space, pca_artifact
from .descriptor_calculation import (
    DescriptorSelectionError,
    calculate_mordred_descriptors,
    select_required_descriptors,
)
from .explanations import SHAP_BACKGROUND_SIZE, calculate_shap_values
from .metadata_loading import resolve_app_path
from .model_loading import (
    ModelLoadError,
    load_model_from_path,
    load_preprocessor_from_path,
    prepare_model_input,
    resolve_required_descriptors,
)


CLASS_LABELS = {0: "Low toxicity", 1: "High toxicity"}
LOGGER = logging.getLogger("mammaltox.prediction")


class PredictionProbabilityError(RuntimeError):
    """Raised when an advertised model probability API fails."""

RESULT_COLUMNS = [
    "Compound_ID",
    "Compound_Name",
    "Input_SMILES",
    "Canonical_SMILES",
    "Species",
    "Route",
    "Endpoint",
    "Model",
    "Model_Classes",
    "Predicted_Class",
    "Predicted_Label",
    "Predicted_Class_Probability",
    "Toxicity_Probability",
    "Raw_Decision_Margin",
    "Probability_Availability",
    "Probability_Calibration_Status",
    "AD_Status",
    "AD_Distance",
    "AD_Threshold",
    "Error_Message",
]


@dataclass
class PredictionBundle:
    results: pd.DataFrame
    pca_artifacts: dict[str, dict]
    shap_artifacts: dict[str, dict]
    input_summary: dict[str, int | str]


def class_to_label(value) -> str:
    try:
        key = int(value)
    except Exception:
        return str(value)
    return CLASS_LABELS.get(key, str(value))


def display_route(value) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"intravenous", "iv", "i.v.", "i.v"}:
        return "IV"
    if normalized == "oral":
        return "Oral"
    return str(value or "Not available").strip()


def result_route(value) -> str:
    """Use an unabbreviated route in prediction results and downloads."""
    return "Intravenous" if display_route(value) == "IV" else display_route(value)


def display_endpoint(value) -> str:
    return "LD50" if "ld50" in str(value or "").lower() else str(value or "Not available").strip()


def endpoint_display_label(model_row: pd.Series) -> str:
    return f"{str(model_row.get('species', 'Unknown')).strip()} {display_route(model_row.get('route'))} {display_endpoint(model_row.get('endpoint'))}"


def _predict_with_probability(model, descriptor_frame: pd.DataFrame, preprocessor=None):
    """Predict classes and retain probability and decision outputs supported by the model."""
    inputs = prepare_model_input(model, descriptor_frame, preprocessor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = np.asarray(model.predict(inputs))

    toxic_probabilities: list[float | None] = [None] * len(predictions)
    confidences: list[float | None] = [None] * len(predictions)
    model_scores: list[float | None] = [None] * len(predictions)
    if hasattr(model, "predict_proba"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probabilities = np.asarray(model.predict_proba(inputs), dtype=float)
            classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
            if 1 in classes:
                positive_index = classes.index(1)
                toxic_probabilities = [float(value) for value in probabilities[:, positive_index]]
            confidences = [
                float(probabilities[row_index, classes.index(predicted_class)])
                for row_index, predicted_class in enumerate(predictions)
            ]
        except Exception as exc:
            LOGGER.exception("Prediction probability calculation failed")
            raise PredictionProbabilityError from exc
    if hasattr(model, "decision_function"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_scores = np.asarray(model.decision_function(inputs), dtype=float)
            if raw_scores.ndim == 1:
                classes = list(getattr(model, "classes_", [0, 1]))
                positive_scores = raw_scores if classes[-1] == 1 else -raw_scores
            else:
                classes = list(getattr(model, "classes_", range(raw_scores.shape[1])))
                positive_index = classes.index(1) if 1 in classes else raw_scores.shape[1] - 1
                positive_scores = raw_scores[:, positive_index]
            model_scores = [float(value) for value in np.asarray(positive_scores).reshape(-1)]
        except Exception:
            LOGGER.exception("Prediction score calculation failed")
    return predictions, toxic_probabilities, confidences, model_scores


def _optional_text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _model_classes_text(model) -> str:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return "Not available"
    return ", ".join(str(value) for value in np.asarray(classes).tolist())


def _clean_input_error(value: str) -> str:
    text = str(value or "").strip()
    if text in {"Invalid SMILES", "Missing SMILES value"}:
        return (
            "The submitted SMILES could not be processed. Please provide a valid "
            "single-compound SMILES."
        )
    if text.startswith(("Mordred descriptor calculation failed", "RDKit/Mordred import failed")):
        return "Mordred descriptor calculation failed."
    return text


def _normalized_inputs(compounds: pd.DataFrame) -> pd.DataFrame:
    working = compounds.copy().reset_index(drop=True)
    rename_map = {}
    if "ID" in working.columns and "Compound_ID" not in working.columns:
        rename_map["ID"] = "Compound_ID"
    if "SMILES" in working.columns and "Input_SMILES" not in working.columns:
        rename_map["SMILES"] = "Input_SMILES"
    working = working.rename(columns=rename_map)
    for column in ["Compound_ID", "Compound_Name", "Input_SMILES", "Source_Error"]:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].fillna("").astype(str)
    for index in working.index:
        if not working.loc[index, "Compound_ID"].strip():
            working.loc[index, "Compound_ID"] = f"Compound_{index + 1:03d}"
    working["_Input_Row"] = working.index.astype(int)
    return working


def _empty_prediction_row(input_row: pd.Series, model_row: pd.Series, error: str) -> dict:
    return {
        "Compound_ID": input_row.get("Compound_ID", ""),
        "Compound_Name": input_row.get("Compound_Name", ""),
        "Input_SMILES": input_row.get("Input_SMILES", ""),
        "Canonical_SMILES": input_row.get("Canonical_SMILES", ""),
        "Species": str(model_row.get("species", "Not available")).strip(),
        "Route": result_route(model_row.get("route")),
        "Endpoint": display_endpoint(model_row.get("endpoint")),
        "Model": str(model_row.get("model_name", "Not available")).strip(),
        "Model_Classes": "Not available",
        "Predicted_Class": pd.NA,
        "Predicted_Label": "",
        "Predicted_Class_Probability": np.nan,
        "Toxicity_Probability": np.nan,
        "Raw_Decision_Margin": np.nan,
        "Probability_Availability": "Not available",
        "Probability_Calibration_Status": "Not available",
        "AD_Status": "Unavailable",
        "AD_Distance": np.nan,
        "AD_Threshold": np.nan,
        "Error_Message": error or "Prediction failed",
        "_Input_Row": int(input_row.get("_Input_Row", 0)),
        "_Model_ID": str(model_row.get("model_id", "")),
    }


def predict_compounds(
    compounds: pd.DataFrame,
    endpoint: str | None,
    model_config: pd.DataFrame,
    selected_descriptors_by_model: dict[str, Sequence[str]],
    app_root: Path,
    include_explanations: bool = True,
) -> PredictionBundle:
    """Run one shared RDKit -> Mordred -> model workflow for single or batch input."""
    if compounds.empty or model_config.empty:
        return PredictionBundle(pd.DataFrame(columns=RESULT_COLUMNS), {}, {}, {})

    selected_models = model_config.copy()
    if endpoint and endpoint != "All":
        selected_models = selected_models[selected_models["model_id"].astype(str).eq(str(endpoint))]
    working = _normalized_inputs(compounds)
    descriptor_df, status_df = calculate_mordred_descriptors(working["Input_SMILES"].tolist())
    working["Canonical_SMILES"] = status_df["Canonical SMILES"].fillna("").astype(str).tolist()

    source_errors = working["Source_Error"].fillna("").astype(str).str.strip()
    validation_errors = status_df["descriptor_error"].fillna("").astype(str).str.strip()
    input_errors = source_errors.where(source_errors.ne(""), validation_errors)
    input_errors = input_errors.map(_clean_input_error)
    valid_input_mask = input_errors.eq("")

    results: list[dict] = []
    pca_artifacts: dict[str, dict] = {}
    shap_artifacts: dict[str, dict] = {}

    for _, model_row in selected_models.iterrows():
        model_id = str(model_row.get("model_id", ""))
        row_errors = {int(index): str(input_errors.iloc[index]) for index in working.index if input_errors.iloc[index]}

        try:
            model = load_model_from_path(str(model_row["model_path"]), str(app_root))
            preprocessor_path = _optional_text(model_row.get("preprocessor_path", ""))
            preprocessor = load_preprocessor_from_path(preprocessor_path, str(app_root))
            required = resolve_required_descriptors(model, model_id, selected_descriptors_by_model)
            if not required:
                raise DescriptorSelectionError("No selected Mordred descriptor list is available for this model")
            selected_x, descriptor_errors = select_required_descriptors(descriptor_df, required)
            for index, error in descriptor_errors.items():
                LOGGER.warning("Descriptor alignment failed for %s row %s: %s", model_id, index, error)
                row_errors.setdefault(
                    index, "Prediction failed because required descriptors could not be aligned."
                )
        except ModelLoadError:
            LOGGER.exception("Model loading failed for %s", model_id)
            for _, input_row in working.iterrows():
                results.append(
                    _empty_prediction_row(
                        input_row, model_row, "Prediction failed because the model could not be loaded."
                    )
                )
            pca_artifacts[model_id] = {"error": "PCA projection is unavailable because the model could not be loaded"}
            shap_artifacts[model_id] = {"error": "SHAP calculation is unavailable because the model could not be loaded"}
            continue
        except (DescriptorSelectionError, KeyError):
            LOGGER.exception("Descriptor alignment failed for %s", model_id)
            for _, input_row in working.iterrows():
                results.append(
                    _empty_prediction_row(
                        input_row,
                        model_row,
                        "Prediction failed because required descriptors could not be aligned.",
                    )
                )
            pca_artifacts[model_id] = {"error": "PCA projection is unavailable because required descriptors could not be aligned"}
            shap_artifacts[model_id] = {"error": "SHAP calculation is unavailable because required descriptors could not be aligned"}
            continue

        ok_indices = [int(index) for index in working.index if int(index) not in row_errors]
        predictions: dict[int, object] = {}
        toxic_probabilities: dict[int, float | None] = {}
        confidences: dict[int, float | None] = {}
        model_scores: dict[int, float | None] = {}
        ad_statuses: dict[int, str] = {}
        ad_distances: dict[int, float] = {}
        ad_thresholds: dict[int, float] = {}

        if ok_indices:
            query_frame = selected_x.loc[ok_indices, required]
            try:
                predicted, toxic, confidence, score = _predict_with_probability(model, query_frame, preprocessor)
                predictions = dict(zip(ok_indices, predicted))
                toxic_probabilities = dict(zip(ok_indices, toxic))
                confidences = dict(zip(ok_indices, confidence))
                model_scores = dict(zip(ok_indices, score))
            except PredictionProbabilityError:
                LOGGER.exception("Probability output failed for %s", model_id)
                for index in ok_indices:
                    row_errors[index] = "Probability calculation failed for this model output."
                ok_indices = []
            except Exception:
                LOGGER.exception("Model prediction or preprocessing failed for %s", model_id)
                for index in ok_indices:
                    row_errors[index] = "Prediction failed during model preprocessing or inference."
                ok_indices = []

        if ok_indices:
            training_path = resolve_app_path(str(model_row.get("selected_descriptor_source", "")), app_root)
            try:
                space = load_training_space(
                    model_id,
                    str(model_row["model_path"]),
                    preprocessor_path,
                    str(training_path),
                    tuple(required),
                    str(app_root),
                )
                query_frame = selected_x.loc[ok_indices, required]
                for index in ok_indices:
                    ad_status, ad_distance, ad_threshold = applicability_domain_details(
                        space, query_frame.loc[index]
                    )
                    ad_statuses[index] = ad_status
                    ad_distances[index] = ad_distance
                    ad_thresholds[index] = ad_threshold
                query_ids = working.loc[ok_indices, "Compound_ID"].astype(str).tolist()
                pca_artifacts[model_id] = pca_artifact(space, query_frame, query_ids)
            except Exception:
                LOGGER.exception("Applicability-domain preparation failed for %s", model_id)
                for index in ok_indices:
                    row_errors[index] = (
                        "Applicability-domain calculation failed because the required training space "
                        "was unavailable."
                    )
                pca_artifacts[model_id] = {
                    "error": "PCA projection failed because the required training space was unavailable"
                }
                ok_indices = []

        if include_explanations and ok_indices:
            shap_indices = ok_indices
            shap_query = selected_x.loc[shap_indices, required]
            shap_artifact = calculate_shap_values(
                model_id,
                str(model_row["model_path"]),
                preprocessor_path,
                str(resolve_app_path(str(model_row.get("selected_descriptor_source", "")), app_root)),
                tuple(required),
                str(app_root),
                tuple(tuple(float(value) for value in row) for row in shap_query.to_numpy()),
                tuple(working.loc[shap_indices, "Compound_ID"].astype(str).tolist()),
                SHAP_BACKGROUND_SIZE,
            )
            shap_artifact.update(
                {
                    "successful_query_count": len(ok_indices),
                    "explained_query_count": len(shap_indices),
                    "sample_limit": "Unlimited",
                    "sampling_method": "All successful compounds",
                    "app_model_outputs": [
                        (
                            float(toxic_probabilities[index])
                            if toxic_probabilities[index] is not None
                            else (
                                float(model_scores[index])
                                if model_scores[index] is not None
                                else float("nan")
                            )
                        )
                        for index in shap_indices
                    ],
                }
            )
            if model_id == "rat_mlp" and not shap_artifact.get("error"):
                wrapper_probabilities = np.asarray(
                    shap_artifact.get("query_probabilities", []), dtype=float
                )
                app_probabilities = np.asarray(
                    shap_artifact.get("app_model_outputs", []), dtype=float
                )
                if len(wrapper_probabilities) == len(app_probabilities):
                    probability_delta = np.abs(wrapper_probabilities - app_probabilities)
                    max_delta = float(probability_delta.max()) if probability_delta.size else 0.0
                    shap_artifact.setdefault("diagnostics", {})[
                        "app_wrapper_probability_max_delta"
                    ] = max_delta
                    LOGGER.info(
                        "Rat IV MLP app/SHAP probability comparison: app=%s wrapper=%s max_delta=%.3g",
                        app_probabilities.tolist(),
                        wrapper_probabilities.tolist(),
                        max_delta,
                    )
            shap_artifacts[model_id] = shap_artifact
        elif include_explanations:
            shap_artifacts[model_id] = {"error": "SHAP calculation failed: no successful query records are available"}

        for index, input_row in working.iterrows():
            index = int(index)
            if index in row_errors:
                results.append(_empty_prediction_row(input_row, model_row, row_errors[index]))
                continue
            predicted_class = predictions[index]
            probability = confidences[index]
            decision_margin = model_scores[index]
            if probability is not None:
                probability_availability = "Available"
                probability_calibration = "Model probability output; calibration not separately assessed"
            else:
                probability_availability = "Not available"
                probability_calibration = (
                    "Not probability-calibrated"
                    if decision_margin is not None
                    else "Not available"
                )
            results.append(
                {
                    "Compound_ID": input_row["Compound_ID"],
                    "Compound_Name": input_row["Compound_Name"],
                    "Input_SMILES": input_row["Input_SMILES"],
                    "Canonical_SMILES": input_row["Canonical_SMILES"],
                    "Species": str(model_row.get("species", "Not available")).strip(),
                    "Route": result_route(model_row.get("route")),
                    "Endpoint": display_endpoint(model_row.get("endpoint")),
                    "Model": str(model_row.get("model_name", "Not available")).strip(),
                    "Model_Classes": _model_classes_text(model),
                    "Predicted_Class": int(predicted_class),
                    "Predicted_Label": class_to_label(predicted_class),
                    "Predicted_Class_Probability": probability,
                    "Toxicity_Probability": toxic_probabilities[index],
                    "Raw_Decision_Margin": decision_margin,
                    "Probability_Availability": probability_availability,
                    "Probability_Calibration_Status": probability_calibration,
                    "AD_Status": ad_statuses[index],
                    "AD_Distance": ad_distances[index],
                    "AD_Threshold": ad_thresholds[index],
                    "Error_Message": "No error",
                    "_Input_Row": index,
                    "_Model_ID": model_id,
                }
            )

    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        result_frame = pd.DataFrame(columns=RESULT_COLUMNS)
    successful_rows = (
        pd.to_numeric(result_frame.get("Predicted_Class"), errors="coerce").notna()
        if "Predicted_Class" in result_frame
        else pd.Series(dtype=bool)
    )
    predicted_inputs = int(result_frame.loc[successful_rows, "_Input_Row"].nunique()) if successful_rows.any() else 0
    summary = {
        "uploaded": int(len(working)),
        "valid": int(valid_input_mask.sum()),
        "predicted": predicted_inputs,
        "failed": int(len(working) - predicted_inputs),
        "mode": "Single prediction" if int(valid_input_mask.sum()) == 1 else "Batch prediction",
    }
    return PredictionBundle(result_frame, pca_artifacts, shap_artifacts, summary)


def run_predictions(
    input_df: pd.DataFrame,
    model_records: pd.DataFrame,
    selected_descriptors_by_model: dict[str, Sequence[str]],
    app_root: Path,
) -> pd.DataFrame:
    """Compatibility wrapper using the shared prediction engine without visual work."""
    return predict_compounds(
        input_df,
        None,
        model_records,
        selected_descriptors_by_model,
        app_root,
        include_explanations=False,
    ).results


def prediction_output_table(results: pd.DataFrame) -> pd.DataFrame:
    output = results[[column for column in RESULT_COLUMNS if column in results.columns]].copy()
    if "Predicted_Class" in output.columns:
        predicted_class = pd.to_numeric(output["Predicted_Class"], errors="coerce").round()
        if "Predicted_Label" in output.columns:
            normalized_labels = predicted_class.map(CLASS_LABELS)
            output["Predicted_Label"] = normalized_labels.where(
                predicted_class.notna(), output["Predicted_Label"]
            )
        output["Predicted_Class"] = predicted_class.map(
            lambda value: str(int(value)) if pd.notna(value) else "Not available"
        )
    for column in [
        "Predicted_Class_Probability",
        "Toxicity_Probability",
        "Raw_Decision_Margin",
        "AD_Distance",
        "AD_Threshold",
    ]:
        if column in output.columns:
            numeric = pd.to_numeric(output[column], errors="coerce").round(6)
            output[column] = numeric.map(
                lambda value: f"{value:.6f}" if pd.notna(value) else "Not available"
            )
    for column in ["Compound_Name"]:
        if column in output.columns and output[column].fillna("").astype(str).str.strip().eq("").all():
            output = output.drop(columns=column)
    for column in output.columns:
        if column == "Error_Message":
            continue
        if not pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].where(pd.notna(output[column]), "Not available")
            output[column] = output[column].map(
                lambda value: "Not available" if str(value).strip() in {"", "None", "NaN", "nan", "null"} else value
            )
    if "Error_Message" in output.columns:
        output["Error_Message"] = output["Error_Message"].fillna("Prediction failed").replace("", "Prediction failed")
    return output


TECHNICAL_COLUMN_LABELS = {
    "Compound_ID": "Compound ID",
    "Input_SMILES": "Input SMILES",
    "Canonical_SMILES": "Canonical SMILES",
    "Species": "Species",
    "Route": "Route",
    "Endpoint": "Endpoint",
    "Model": "Model",
    "Predicted_Label": "Prediction",
    "Predicted_Class_Probability": "Predicted-class probability",
    "AD_Status": "AD status",
    "AD_Distance": "AD distance",
    "AD_Threshold": "AD threshold",
    "Error_Message": "Error message",
}


def technical_output_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return the streamlined user-facing prediction download table."""
    output = prediction_output_table(results)
    columns = [column for column in TECHNICAL_COLUMN_LABELS if column in output.columns]
    return output.loc[:, columns].rename(columns=TECHNICAL_COLUMN_LABELS)


def main_results_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return the concise browser table with only user-actionable status details."""
    output = prediction_output_table(results)
    successful = pd.to_numeric(output.get("Predicted_Class"), errors="coerce").notna()

    predicted_class = pd.to_numeric(output.get("Predicted_Class"), errors="coerce")
    prediction = predicted_class.map(
        lambda value: (
            "Low toxicity"
            if pd.notna(value) and int(value) == 0
            else "High toxicity"
            if pd.notna(value) and int(value) == 1
            else "Failed"
        )
    )
    probabilities = pd.to_numeric(
        output.get("Predicted_Class_Probability"), errors="coerce"
    )
    displayed_probability = probabilities.map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "Not probability-calibrated"
    )
    displayed_probability = displayed_probability.where(successful, "Not available")

    ad_status = output.get("AD_Status", pd.Series(index=output.index, dtype=object)).map(
        lambda value: value if value in {"In AD", "Out of AD"} else "Unavailable"
    )

    def displayed_error(index) -> str:
        backend_error = str(output.loc[index, "Error_Message"])
        if not bool(successful.loc[index]):
            if "valid single-compound SMILES" in backend_error or "Invalid SMILES" in backend_error:
                return "Structure could not be processed because the SMILES is invalid."
            if "required descriptors" in backend_error:
                return "Prediction failed because required descriptors could not be calculated."
            if "Applicability-domain calculation failed" in backend_error:
                return "AD unavailable because the applicability-domain calculation failed."
            return backend_error
        if ad_status.loc[index] == "Unavailable":
            return "AD unavailable because the applicability-domain calculation failed."
        if displayed_probability.loc[index] == "Not probability-calibrated":
            return (
                "Probability unavailable because this model does not provide calibrated "
                "probability output."
            )
        return "No error"

    error_messages = pd.Series(
        [displayed_error(index) for index in output.index],
        index=output.index,
        dtype=object,
    )

    return pd.DataFrame(
        {
            "Compound ID": output.get("Compound_ID", "Not available"),
            "Input SMILES": output.get("Input_SMILES", "Not available"),
            "Species": output.get("Species", "Not available"),
            "Route": output.get("Route", "Not available"),
            "Endpoint": output.get("Endpoint", "Not available"),
            "Model": output.get("Model", "Not available"),
            "Prediction": prediction,
            "Predicted-class probability": displayed_probability,
            "AD Status": ad_status,
            "Error Message": error_messages,
        }
    )


def columns_for_single_prediction(results: pd.DataFrame) -> pd.DataFrame:
    return prediction_output_table(results)


def columns_for_batch_prediction(results: pd.DataFrame) -> pd.DataFrame:
    return prediction_output_table(results)
