import json
from pathlib import Path
from typing import Any, Dict, List

from src.dataset_validator import DatasetValidationError


COUNTRY_REGISTRY_PATH = "data/country_registry.json"


def load_country_registry(
    registry_path: str = COUNTRY_REGISTRY_PATH
) -> Dict[str, Any]:
    """
    Load the country registry JSON file.

    The registry is the single source of truth for:
        - available countries
        - country codes
        - currencies
        - dataset file paths

    Do not hardcode country names inside app.py.
    """

    path = Path(registry_path)

    if not path.exists():
        raise FileNotFoundError(f"Country registry file not found: {registry_path}")

    if path.suffix.lower() != ".json":
        raise DatasetValidationError("Country registry must be a .json file.")

    try:
        with open(path, "r", encoding="utf-8") as file:
            registry = json.load(file)

    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"Invalid country registry JSON format. Error: {error}"
        )

    validate_country_registry(registry)

    return registry


def validate_country_registry(registry: Dict[str, Any]) -> None:
    """
    Validate the country registry structure.
    """

    if not isinstance(registry, dict):
        raise DatasetValidationError("Country registry root must be a JSON object.")

    if "countries" not in registry:
        raise DatasetValidationError("Country registry must contain a 'countries' list.")

    if not isinstance(registry["countries"], list):
        raise DatasetValidationError("'countries' must be a list.")

    if not registry["countries"]:
        raise DatasetValidationError("Country registry has no countries.")

    required_country_fields = [
        "name",
        "code",
        "currency",
        "dataset_path",
        "enabled"
    ]

    country_names = []

    for country in registry["countries"]:
        if not isinstance(country, dict):
            raise DatasetValidationError(
                "Each country registry item must be a JSON object."
            )

        missing_fields = [
            field for field in required_country_fields
            if field not in country
        ]

        if missing_fields:
            raise DatasetValidationError(
                "Country registry item is missing fields: "
                + ", ".join(missing_fields)
            )

        country_name = str(country.get("name", "")).strip()

        if not country_name:
            raise DatasetValidationError("Country name cannot be empty.")

        if country_name.lower() in country_names:
            raise DatasetValidationError(
                f"Duplicate country in registry: {country_name}"
            )

        country_names.append(country_name.lower())

        dataset_path = Path(country["dataset_path"])

        if country.get("enabled") is True and not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file for {country_name} was not found: {dataset_path}"
            )


def get_country_records(
    registry_path: str = COUNTRY_REGISTRY_PATH,
    enabled_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Return country registry records.
    """

    registry = load_country_registry(registry_path)
    countries = registry.get("countries", [])

    if enabled_only:
        countries = [
            country for country in countries
            if country.get("enabled") is True
        ]

    countries = sorted(
        countries,
        key=lambda item: item.get("sort_order", 999)
    )

    return countries


def get_available_countries(
    registry_path: str = COUNTRY_REGISTRY_PATH
) -> List[str]:
    """
    Return the list of available country names for Streamlit selectbox.
    """

    countries = get_country_records(
        registry_path=registry_path,
        enabled_only=True
    )

    return [
        country["name"]
        for country in countries
    ]


def get_default_country(
    registry_path: str = COUNTRY_REGISTRY_PATH
) -> str:
    """
    Return the default country from the registry.

    If the configured default country is not enabled or not found,
    the first enabled country is returned.
    """

    registry = load_country_registry(registry_path)

    default_country = registry.get("default_country")
    available_countries = get_available_countries(registry_path)

    if default_country in available_countries:
        return default_country

    if available_countries:
        return available_countries[0]

    raise DatasetValidationError("No enabled countries found in country registry.")


def get_country_config(
    country_name: str,
    registry_path: str = COUNTRY_REGISTRY_PATH
) -> Dict[str, Any]:
    """
    Return one country registry record by country name.
    """

    countries = get_country_records(
        registry_path=registry_path,
        enabled_only=True
    )

    selected_name = str(country_name).strip().lower()

    for country in countries:
        if country["name"].strip().lower() == selected_name:
            return country

    raise DatasetValidationError(
        f"Selected country was not found in registry: {country_name}"
    )


def get_country_dataset_path(
    country_name: str,
    registry_path: str = COUNTRY_REGISTRY_PATH
) -> str:
    """
    Return the dataset path for the selected country.
    """

    country_config = get_country_config(
        country_name=country_name,
        registry_path=registry_path
    )

    return country_config["dataset_path"]


def get_country_metadata(
    country_name: str,
    registry_path: str = COUNTRY_REGISTRY_PATH
) -> Dict[str, Any]:
    """
    Read only the metadata section of the selected country's dataset.

    This is useful when you need country, currency, base year, or horizon
    without running the full app simulation.
    """

    dataset_path = get_country_dataset_path(
        country_name=country_name,
        registry_path=registry_path
    )

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    try:
        with open(path, "r", encoding="utf-8") as file:
            dataset = json.load(file)

    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"Invalid dataset JSON format for {country_name}. Error: {error}"
        )

    metadata = dataset.get("metadata")

    if not isinstance(metadata, dict):
        raise DatasetValidationError(
            f"Dataset metadata missing or invalid for {country_name}."
        )

    return {
        "country": metadata.get("country", country_name),
        "currency": metadata.get("currency", country_config_currency(country_name)),
        "base_year": metadata.get("base_year"),
        "time_horizon_years": metadata.get("time_horizon_years"),
        "dataset_name": metadata.get("dataset_name"),
        "dataset_path": dataset_path
    }


def country_config_currency(
    country_name: str,
    registry_path: str = COUNTRY_REGISTRY_PATH
) -> str:
    """
    Return currency from registry for the selected country.
    """

    country_config = get_country_config(
        country_name=country_name,
        registry_path=registry_path
    )

    return country_config.get("currency", "")