from typing import Any, Dict, List, Tuple


class DatasetValidationError(Exception):
    """
    Custom error for dataset validation problems.
    """
    pass


REQUIRED_SECTIONS = [
    "metadata",
    "visa",
    "education",
    "income",
    "tax_and_retirement",
    "expenses",
    "car",
    "investment_and_economy",
    "loans_and_debt",
    "scenario_defaults",
    "sensitivity_analysis"
]


REQUIRED_VALUE_PATHS = [
    "metadata.dataset_name",
    "metadata.country",
    "metadata.currency",
    "metadata.base_year",
    "metadata.time_horizon_years",

    "visa.student_visa.application_fee.value",
    "visa.student_visa.legal_work_hours_per_week.value",
    "visa.graduate_visa.application_fee.value",
    "visa.graduate_visa.duration_years.value",
    "visa.skilled_work_visa.application_fee.value",
    "visa.permanent_residency.application_fee.value",

    "education.masters_or_mba.annual_tuition_fee.value",
    "education.masters_or_mba.full_time_duration_years.value",
    "education.masters_or_mba.part_time_duration_years.value",

    "income.it_software_salary.graduate_annual_salary.value",
    "income.it_software_salary.mid_level_annual_salary.value",
    "income.it_software_salary.senior_annual_salary.value",
    "income.it_software_salary.annual_salary_growth_rate.value",
    "income.student_part_time_work.hourly_wage.value",
    "income.student_part_time_work.working_weeks_per_year.value",
    "income.spouse_income.annual_salary.value",

    "tax_and_retirement.effective_income_tax_rate.value",
    "tax_and_retirement.employer_superannuation_rate.value",

    "expenses.general_living_excluding_rent.single_monthly.value",
    "expenses.general_living_excluding_rent.couple_monthly.value",
    "expenses.general_living_excluding_rent.family_with_one_child_monthly.value",
    "expenses.rent.single_monthly.value",
    "expenses.rent.family_monthly.value",
    "expenses.childcare.monthly_cost_per_child.value",
    "expenses.childcare.applies_until_child_age.value",
    "expenses.transport.public_transport_monthly.value",
    "expenses.healthcare_or_insurance.monthly_cost.value",

    "car.used_car_purchase_price.value",
    "car.annual_insurance.value",
    "car.annual_fuel.value",
    "car.annual_maintenance.value",
    "car.annual_depreciation_rate.value",
    "car.car_loan_interest_rate.value",

    "investment_and_economy.inflation_rate.value",
    "investment_and_economy.savings_interest_rate.value",
    "investment_and_economy.investment_return_rate.value",

    "loans_and_debt.education_loan_interest_rate.value",
    "loans_and_debt.migration_loan_interest_rate.value",
    "loans_and_debt.borrowing_allowed",
    "loans_and_debt.negative_cash_allowed",

    "scenario_defaults.student_visa_path.study_years",
    "scenario_defaults.student_visa_path.full_time_work_start_year",
    "scenario_defaults.student_visa_path.graduate_visa_start_year",
    "scenario_defaults.working_visa_path.full_time_work_start_year",
    "scenario_defaults.life_scenarios.single",
    "scenario_defaults.life_scenarios.married_no_child",
    "scenario_defaults.life_scenarios.married_one_child",
    "scenario_defaults.life_scenarios.married_two_children"
]


SENSITIVITY_REQUIRED_KEYWORDS = {
    "salary growth": [
        "salary_growth",
        "salary growth",
        "salary_growth_rate",
        "annual_salary_growth_rate"
    ],
    "rent": [
        "rent",
        "rent_multiplier"
    ],
    "tuition": [
        "tuition",
        "tuition_multiplier"
    ],
    "childcare": [
        "childcare",
        "childcare_multiplier"
    ],
    "investment return": [
        "investment_return",
        "investment return",
        "investment_return_rate"
    ],
    "exchange rate": [
        "exchange_rate",
        "exchange rate",
        "_to_lkr"
    ],
    "spouse income": [
        "spouse_income",
        "spouse income"
    ]
}


def get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """
    Read a nested value using dot notation.

    Example:
        income.it_software_salary.graduate_annual_salary.value
    """

    current_value = data

    for key in path.split("."):
        if not isinstance(current_value, dict):
            raise DatasetValidationError(
                f"Invalid structure at '{path}'. "
                f"Expected dictionary before key '{key}'."
            )

        if key not in current_value:
            raise DatasetValidationError(f"Missing required field: {path}")

        current_value = current_value[key]

    return current_value


def validate_required_sections(data: Dict[str, Any]) -> None:
    """
    Check whether all main dataset sections exist.
    """

    missing_sections = []

    for section in REQUIRED_SECTIONS:
        if section not in data:
            missing_sections.append(section)

    if missing_sections:
        raise DatasetValidationError(
            "Missing required sections: " + ", ".join(missing_sections)
        )


def validate_required_values(data: Dict[str, Any]) -> None:
    """
    Check whether important nested values exist and are not empty.
    """

    missing_or_empty_fields = []

    for path in REQUIRED_VALUE_PATHS:
        try:
            value = get_nested_value(data, path)

            if value is None or value == "":
                missing_or_empty_fields.append(path)

        except DatasetValidationError:
            missing_or_empty_fields.append(path)

    if missing_or_empty_fields:
        raise DatasetValidationError(
            "Missing or empty required values: "
            + ", ".join(missing_or_empty_fields)
        )


def get_dataset_currency(data: Dict[str, Any]) -> str:
    """
    Return metadata currency as uppercase currency code.
    """

    currency = get_nested_value(data, "metadata.currency")

    if currency is None:
        raise DatasetValidationError("metadata.currency is missing.")

    currency = str(currency).strip().upper()

    if not currency:
        raise DatasetValidationError("metadata.currency cannot be empty.")

    return currency


def get_expected_exchange_rate_key(data: Dict[str, Any]) -> str:
    """
    Build expected exchange-rate field name dynamically.

    Examples:
        AUD -> aud_to_lkr_exchange_rate
        EUR -> eur_to_lkr_exchange_rate
        JPY -> jpy_to_lkr_exchange_rate
        LKR -> lkr_to_lkr_exchange_rate
    """

    currency = get_dataset_currency(data)

    return f"{currency.lower()}_to_lkr_exchange_rate"


def get_expected_exchange_rate_path(data: Dict[str, Any]) -> str:
    """
    Return full dot path for selected currency exchange-rate value.
    """

    exchange_rate_key = get_expected_exchange_rate_key(data)

    return f"investment_and_economy.{exchange_rate_key}.value"


def validate_exchange_rate_field(data: Dict[str, Any]) -> None:
    """
    Check whether the selected country's exchange-rate field exists.
    """

    exchange_rate_path = get_expected_exchange_rate_path(data)

    try:
        exchange_rate = get_nested_value(data, exchange_rate_path)

    except DatasetValidationError:
        raise DatasetValidationError(
            f"Missing exchange-rate field for metadata.currency. "
            f"Expected: {exchange_rate_path}"
        )

    try:
        exchange_rate_number = float(exchange_rate)

    except (TypeError, ValueError):
        raise DatasetValidationError(
            f"Exchange-rate value must be numeric: {exchange_rate_path}"
        )

    if exchange_rate_number <= 0:
        raise DatasetValidationError(
            f"Exchange-rate value must be greater than zero: {exchange_rate_path}"
        )


def is_value_record(node: Any) -> bool:
    """
    Detect objects that store a value with metadata.

    Example:
        {
            "value": 5000,
            "currency": "AUD",
            "year": 2026,
            "source": "..."
        }
    """

    return isinstance(node, dict) and "value" in node


def collect_value_records(
    data: Dict[str, Any]
) -> List[Tuple[List[str], Dict[str, Any]]]:
    """
    Collect all structured value records from the dataset.
    """

    records = []

    def walk(current_node: Any, current_path: List[str]) -> None:
        if is_value_record(current_node):
            records.append((current_path, current_node))
            return

        if isinstance(current_node, dict):
            for key, value in current_node.items():
                walk(value, current_path + [str(key)])

    walk(data, [])

    return records


def looks_like_currency_code(value: Any) -> bool:
    """
    Return True only for simple 3-letter currency codes.

    Examples:
        AUD, EUR, JPY, LKR -> True
        percentage, years, LKR per AUD -> False
    """

    if value is None:
        return False

    value = str(value).strip().upper()

    return len(value) == 3 and value.isalpha()


def is_exchange_rate_path(path_parts: List[str]) -> bool:
    """
    Detect exchange-rate records so they can be handled separately.

    Exchange-rate records may be stored as LKR per local currency,
    so their currency field should not be treated like normal local-currency values.
    """

    path_text = ".".join(path_parts).lower()

    return (
        "exchange_rate" in path_text
        or "_to_lkr" in path_text
        or "to_lkr" in path_text
    )


def validate_currency_fields(data: Dict[str, Any]) -> None:
    """
    Check that structured monetary records match metadata.currency.

    This catches dangerous mistakes like:
        Japan dataset metadata.currency = JPY
        but spouse income currency = EUR

    Exchange-rate records are excluded because they are conversion values.
    """

    expected_currency = get_dataset_currency(data)
    records = collect_value_records(data)

    currency_errors = []

    for path_parts, node in records:
        if is_exchange_rate_path(path_parts):
            continue

        record_currency = node.get("currency")

        if not record_currency:
            continue

        if not looks_like_currency_code(record_currency):
            continue

        record_currency = str(record_currency).strip().upper()

        if record_currency != expected_currency:
            currency_errors.append(
                f"{'.'.join(path_parts)} has currency {record_currency}, "
                f"expected {expected_currency}"
            )

    if currency_errors:
        raise DatasetValidationError(
            "Currency mismatch detected. "
            "This dataset is unsafe to simulate until fixed. "
            + " | ".join(currency_errors[:20])
        )


def validate_sensitivity_analysis(data: Dict[str, Any]) -> None:
    """
    Check that sensitivity_analysis exists and contains the key variables.

    This is intentionally flexible because different datasets may store
    sensitivity variables as lists, dictionaries, or nested objects.
    """

    sensitivity_data = data.get("sensitivity_analysis")

    if sensitivity_data is None:
        raise DatasetValidationError("Missing required section: sensitivity_analysis")

    if sensitivity_data == {} or sensitivity_data == []:
        raise DatasetValidationError("sensitivity_analysis cannot be empty.")

    sensitivity_text = str(sensitivity_data).lower()

    missing_variables = []

    for variable_name, keywords in SENSITIVITY_REQUIRED_KEYWORDS.items():
        found = any(
            keyword.lower() in sensitivity_text
            for keyword in keywords
        )

        if not found:
            missing_variables.append(variable_name)

    if missing_variables:
        raise DatasetValidationError(
            "Missing sensitivity variables: "
            + ", ".join(missing_variables)
        )


def validate_dataset(data: Dict[str, Any]) -> None:
    """
    Run all dataset validation checks.
    """

    validate_required_sections(data)
    validate_required_values(data)
    validate_exchange_rate_field(data)
    validate_currency_fields(data)
    validate_sensitivity_analysis(data)