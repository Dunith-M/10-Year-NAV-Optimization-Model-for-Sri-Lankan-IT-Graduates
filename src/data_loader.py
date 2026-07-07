import json
from pathlib import Path
from typing import Any, Dict, List

from src.dataset_validator import (
    DatasetValidationError,
    validate_dataset
)

from src.currency_utils import (
    get_country_currency,
    get_currency_symbol,
    get_exchange_rate_key,
    get_exchange_rate_path,
    get_exchange_rate_to_lkr,
    get_nested_value
)


VALUE_METADATA_KEYS = {
    "value",
    "currency",
    "unit",
    "year",
    "source",
    "source_url",
    "url",
    "notes"
}


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON dataset from the given path.
    Returns the dataset as a Python dictionary.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if path.suffix.lower() != ".json":
        raise DatasetValidationError("Dataset file must be a .json file.")

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"Invalid JSON format. Error: {error}"
        )

    if not isinstance(data, dict):
        raise DatasetValidationError("Dataset root must be a JSON object.")

    return data


def load_dataset(file_path: str) -> Dict[str, Any]:
    """
    Main function used by the Streamlit app.
    Loads and validates the selected country dataset.
    """

    data = load_json_file(file_path)
    validate_dataset(data)

    return data


def get_dataset_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key values for display in the app.

    This version is country-safe.
    It does not expose AUD-specific exchange-rate aliases.
    """

    currency = get_country_currency(data)
    currency_symbol = get_currency_symbol(currency)
    exchange_rate_key = get_exchange_rate_key(data)
    exchange_rate_path = get_exchange_rate_path(data)
    exchange_rate_value = get_exchange_rate_to_lkr(data)

    summary = {
        "dataset_name": get_nested_value(data, "metadata.dataset_name"),
        "country": get_nested_value(data, "metadata.country"),
        "currency": currency,
        "currency_symbol": currency_symbol,
        "base_year": get_nested_value(data, "metadata.base_year"),
        "time_horizon_years": get_nested_value(data, "metadata.time_horizon_years"),

        "student_visa_fee": get_nested_value(
            data, "visa.student_visa.application_fee.value"
        ),
        "graduate_visa_fee": get_nested_value(
            data, "visa.graduate_visa.application_fee.value"
        ),
        "pr_application_fee": get_nested_value(
            data, "visa.permanent_residency.application_fee.value"
        ),

        "annual_tuition_fee": get_nested_value(
            data, "education.masters_or_mba.annual_tuition_fee.value"
        ),

        "graduate_salary": get_nested_value(
            data, "income.it_software_salary.graduate_annual_salary.value"
        ),
        "mid_level_salary": get_nested_value(
            data, "income.it_software_salary.mid_level_annual_salary.value"
        ),
        "senior_salary": get_nested_value(
            data, "income.it_software_salary.senior_annual_salary.value"
        ),

        "single_monthly_rent": get_nested_value(
            data, "expenses.rent.single_monthly.value"
        ),
        "family_monthly_rent": get_nested_value(
            data, "expenses.rent.family_monthly.value"
        ),

        "inflation_rate": get_nested_value(
            data, "investment_and_economy.inflation_rate.value"
        ),
        "investment_return_rate": get_nested_value(
            data, "investment_and_economy.investment_return_rate.value"
        ),

        "exchange_rate_key": exchange_rate_key,
        "exchange_rate_path": exchange_rate_path,
        "exchange_rate_to_lkr": exchange_rate_value,

        exchange_rate_key: exchange_rate_value
    }

    return summary


# ---------------------------------------------------------------------
# Assumption/source explorer helpers
# ---------------------------------------------------------------------

def prettify_key(key: str) -> str:
    """
    Convert dataset keys into readable labels.
    Example:
        annual_tuition_fee -> Annual tuition fee
    """

    if not isinstance(key, str):
        return str(key)

    return key.replace("_", " ").strip().capitalize()


def prettify_path(path_parts: List[str]) -> str:
    """
    Convert a nested JSON path into a readable variable name.
    """

    if not path_parts:
        return "Dataset"

    return " → ".join(prettify_key(part) for part in path_parts)


def stringify_dataset_value(value: Any) -> Any:
    """
    Keep simple values as they are.
    Convert lists/dicts into readable JSON strings for table display.
    """

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return value


def is_value_record(node: Any) -> bool:
    """
    Detect a dataset object that stores a real value with metadata.
    """

    return isinstance(node, dict) and "value" in node


def extract_unit_or_currency(node: Dict[str, Any]) -> str:
    """
    Extract currency/unit display value from a value record.
    """

    currency = node.get("currency", "")
    unit = node.get("unit", "")

    if currency and unit:
        return f"{currency} / {unit}"

    if currency:
        return str(currency)

    if unit:
        return str(unit)

    return ""


def extract_source(node: Dict[str, Any]) -> str:
    """
    Extract source field from common source keys.
    """

    return str(
        node.get("source")
        or node.get("source_url")
        or node.get("url")
        or ""
    )


def make_flattened_row(
    path_parts: List[str],
    value: Any,
    currency_or_unit: str = "",
    year: Any = "",
    source: str = "",
    notes: str = "",
    is_structured_value_record: bool = False
) -> Dict[str, Any]:
    """
    Create one flattened dataset row.
    """

    section = prettify_key(path_parts[0]) if path_parts else "Dataset"
    variable_parts = path_parts[1:] if len(path_parts) > 1 else path_parts

    return {
        "path": ".".join(path_parts),
        "section": section,
        "variable": prettify_path(variable_parts),
        "value": stringify_dataset_value(value),
        "currency_or_unit": currency_or_unit,
        "year": year if year is not None else "",
        "source": source,
        "notes": notes if notes is not None else "",
        "is_value_record": is_structured_value_record
    }


def flatten_dataset_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten the dataset into assumption/source records.

    This does not change the dataset.
    It only converts nested JSON into table-friendly rows.
    """

    rows = []

    def walk(current_node: Any, current_path: List[str]) -> None:
        if is_value_record(current_node):
            rows.append(
                make_flattened_row(
                    path_parts=current_path,
                    value=current_node.get("value"),
                    currency_or_unit=extract_unit_or_currency(current_node),
                    year=current_node.get("year", ""),
                    source=extract_source(current_node),
                    notes=current_node.get("notes", ""),
                    is_structured_value_record=True
                )
            )
            return

        if isinstance(current_node, dict):
            for key, value in current_node.items():
                walk(value, current_path + [str(key)])
            return

        if isinstance(current_node, list):
            rows.append(
                make_flattened_row(
                    path_parts=current_path,
                    value=current_node,
                    is_structured_value_record=False
                )
            )
            return

        rows.append(
            make_flattened_row(
                path_parts=current_path,
                value=current_node,
                is_structured_value_record=False
            )
        )

    walk(data, [])

    return rows


def get_dataset_assumption_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Public helper for assumption explorer.
    """

    return flatten_dataset_records(data)


def get_dataset_source_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Public helper for source explorer.

    Only structured value records are returned because primitive scenario fields
    usually do not have source metadata.
    """

    rows = flatten_dataset_records(data)

    return [
        row for row in rows
        if row.get("is_value_record") is True
    ]