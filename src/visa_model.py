from typing import Any, Dict, Optional


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


def get_legacy_default_pr_application_year(
    scenario_config: Dict[str, Any]
) -> Optional[int]:
    """
    Backward-compatible fallback only.

    New scenario configs should always carry:
        scenario_config["scenario_overrides"]["pr_application_year"]

    This fallback prevents old saved configs from crashing.
    """

    migration_path_key = scenario_config["selected_keys"]["migration_path"]

    if migration_path_key == "student_visa_path":
        return 6

    if migration_path_key == "working_visa_path":
        return 4

    return None


def get_pr_application_year(
    scenario_config: Dict[str, Any]
) -> Optional[int]:
    """
    Return the PR application year from scenario overrides.

    If the user selects "No PR within 10 years", this returns None.
    """

    overrides = get_scenario_overrides(scenario_config)

    if "pr_application_year" in overrides:
        pr_application_year = overrides.get("pr_application_year")

        if pr_application_year is None:
            return None

        return int(pr_application_year)

    return get_legacy_default_pr_application_year(scenario_config)


def get_visa_status_for_year(
    scenario_config: Dict[str, Any],
    year: int
) -> str:
    """
    Human-readable visa / migration status for one simulation year.
    """

    migration_path_key = scenario_config["selected_keys"]["migration_path"]
    migration_path_defaults = scenario_config["migration_path_defaults"]

    pr_application_year = get_pr_application_year(scenario_config)

    if pr_application_year is not None and year >= pr_application_year:
        return "PR application / PR pathway"

    if migration_path_key == "student_visa_path":
        graduate_visa_start_year = int(
            migration_path_defaults.get("graduate_visa_start_year", 3)
        )

        if year < graduate_visa_start_year:
            return "Student visa"

        return "Graduate visa"

    if migration_path_key == "working_visa_path":
        return "Skilled work visa"

    return "Unknown visa status"


def calculate_visa_fee_expense(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> float:
    """
    Calculate visa-related fees for a given year.

    Fixed starting visa costs:
        Student path pays student visa fee in Year 1.
        Student path pays graduate visa fee at graduate visa start year.
        Working path pays skilled work visa fee in Year 1.

    Flexible PR cost:
        PR application fee is controlled only by:
            scenario_config["scenario_overrides"]["pr_application_year"]

        If that value is None, PR fee is zero within the 10-year horizon.
    """

    migration_path_key = scenario_config["selected_keys"]["migration_path"]
    migration_path_defaults = scenario_config["migration_path_defaults"]

    student_visa_fee = float(
        get_value(dataset, "visa.student_visa.application_fee.value")
    )
    graduate_visa_fee = float(
        get_value(dataset, "visa.graduate_visa.application_fee.value")
    )
    skilled_work_visa_fee = float(
        get_value(dataset, "visa.skilled_work_visa.application_fee.value")
    )
    pr_application_fee = float(
        get_value(dataset, "visa.permanent_residency.application_fee.value")
    )

    visa_fee = 0.0
    pr_application_year = get_pr_application_year(scenario_config)

    if migration_path_key == "student_visa_path":
        graduate_visa_start_year = int(
            migration_path_defaults.get("graduate_visa_start_year", 3)
        )

        if year == 1:
            visa_fee += student_visa_fee

        if year == graduate_visa_start_year:
            visa_fee += graduate_visa_fee

    elif migration_path_key == "working_visa_path":
        if year == 1:
            visa_fee += skilled_work_visa_fee

    if pr_application_year is not None and year == pr_application_year:
        visa_fee += pr_application_fee

    return visa_fee * inflation_factor
