from typing import Any, Dict, List


def get_value(dataset: Dict[str, Any], path: str) -> Any:
    """
    Read a nested value from the dataset using dot notation.
    """

    current_value = dataset

    for key in path.split("."):
        current_value = current_value[key]

    return current_value


def get_scenario_overrides(scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely read scenario overrides.

    This keeps older scenario_config objects backward-compatible.
    """

    return scenario_config.get("scenario_overrides", {})


def get_default_study_years(scenario_config: Dict[str, Any]) -> List[int]:
    """
    Dataset-default education logic.

    Student visa path:
        Uses dataset study years.

    Working visa path:
        No study by default.
    """

    migration_path_key = scenario_config["selected_keys"]["migration_path"]
    migration_path_defaults = scenario_config["migration_path_defaults"]

    if migration_path_key != "student_visa_path":
        return []

    return [
        int(year)
        for year in migration_path_defaults.get("study_years", [])
    ]


def get_default_tuition_load(scenario_config: Dict[str, Any]) -> float:
    """
    Dataset-default tuition load.

    Student path pays full tuition during study years.
    Working path pays no tuition unless education override is selected.
    """

    migration_path_key = scenario_config["selected_keys"]["migration_path"]

    if migration_path_key == "student_visa_path":
        return 1.0

    return 0.0


def get_study_years(scenario_config: Dict[str, Any]) -> List[int]:
    """
    Return the study years selected by the scenario.

    Priority:
        1. scenario_config["scenario_overrides"]["education_study_years"]
        2. dataset-default study years for the selected migration path
    """

    overrides = get_scenario_overrides(scenario_config)

    if overrides.get("education_override_enabled", False):
        return [
            int(year)
            for year in overrides.get("education_study_years", [])
        ]

    return get_default_study_years(scenario_config)


def get_tuition_load(scenario_config: Dict[str, Any]) -> float:
    """
    Return the tuition load selected by the scenario.

    Examples:
        No further study       -> 0.0
        Full-time study        -> 1.0
        Part-time study        -> 0.5
    """

    overrides = get_scenario_overrides(scenario_config)

    if overrides.get("education_override_enabled", False):
        return float(overrides.get("tuition_load") or 0.0)

    return get_default_tuition_load(scenario_config)


def get_education_status_for_year(
    scenario_config: Dict[str, Any],
    year: int
) -> str:
    """
    Human-readable education status for one simulation year.
    """

    study_years = get_study_years(scenario_config)
    tuition_load = get_tuition_load(scenario_config)

    if year not in study_years or tuition_load <= 0:
        return "Not studying"

    if tuition_load >= 0.75:
        return "Full-time study"

    return "Part-time study"


def calculate_tuition_expense(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> float:
    """
    Calculate tuition expense for a given year.

    The dataset provides the base tuition value.
    The scenario selection decides whether tuition applies.
    """

    study_years = get_study_years(scenario_config)
    tuition_load = get_tuition_load(scenario_config)

    if year not in study_years or tuition_load <= 0:
        return 0.0

    tuition_multiplier = float(
        scenario_config["adjustable_inputs"]["tuition_multiplier"]
    )

    annual_tuition = float(
        get_value(dataset, "education.masters_or_mba.annual_tuition_fee.value")
    )

    return annual_tuition * tuition_load * inflation_factor * tuition_multiplier
