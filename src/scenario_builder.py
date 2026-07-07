from typing import Any, Dict, Optional

from src.scenario_options import (
    resolve_education_mode,
    resolve_pr_application_year,
    resolve_car_purchase_timing,
    resolve_child_timing,
    resolve_investment_split
)

from src.family_model import (
    get_marriage_year,
    get_first_child_year,
    get_second_child_year
)


MIGRATION_PATH_OPTIONS = {
    "Student visa path": "student_visa_path",
    "Working visa path": "working_visa_path"
}


LIFE_SCENARIO_OPTIONS = {
    "Single": "single",
    "Married no child": "married_no_child",
    "Married one child": "married_one_child",
    "Married two children": "married_two_children"
}


CAR_OPTIONS = {
    "No car": False,
    "Buy car": True
}


INVESTMENT_OPTIONS = {
    "Save only": "save_only",
    "Invest positive cash flow": "invest_positive_cash_flow"
}


SPOUSE_INCOME_OPTIONS = {
    "Conservative": "conservative",
    "Moderate": "moderate",
    "Optimistic": "optimistic"
}


def get_value(dataset: Dict[str, Any], path: str) -> Any:
    """
    Get a nested value from the dataset using dot notation.

    Example:
    get_value(dataset, "income.it_software_salary.annual_salary_growth_rate.value")
    """

    current_value = dataset

    for key in path.split("."):
        current_value = current_value[key]

    return current_value


def get_default_model_inputs(dataset: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract default adjustable inputs from the JSON dataset.
    These become default values in the Streamlit sidebar.
    """

    return {
        "salary_growth_rate": get_value(
            dataset,
            "income.it_software_salary.annual_salary_growth_rate.value"
        ),
        "inflation_rate": get_value(
            dataset,
            "investment_and_economy.inflation_rate.value"
        ),
        "investment_return_rate": get_value(
            dataset,
            "investment_and_economy.investment_return_rate.value"
        ),
        "rent_multiplier": 1.0,
        "tuition_multiplier": 1.0,
        "childcare_multiplier": 1.0
    }


def get_default_pr_application_year(migration_path_key: str) -> Optional[int]:
    """
    Default PR timing used when the UI does not pass an explicit PR timing label.

    This default is assigned inside scenario_config["scenario_overrides"].
    Expense logic should read the override, not hardcode PR years.
    """

    if migration_path_key == "student_visa_path":
        return 6

    if migration_path_key == "working_visa_path":
        return 4

    return None


def get_default_education_settings(
    migration_path_key: str,
    migration_path_defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Dataset-default education settings.

    Student path:
        Study years come from the dataset and tuition load is full.

    Working path:
        No tuition by default unless the advanced education mode overrides it.
    """

    if migration_path_key == "student_visa_path":
        return {
            "education_override_enabled": False,
            "education_study_years": migration_path_defaults.get("study_years", []),
            "tuition_load": 1.0
        }

    return {
        "education_override_enabled": False,
        "education_study_years": [],
        "tuition_load": 0.0
    }


def normalize_scenario_overrides(
    scenario_overrides: Dict[str, Any],
    migration_path_key: str,
    migration_path_defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make sure the model files always receive the keys they need.

    Education, visa, car, investment, and family logic should read this control
    layer instead of hardcoding years inside model functions.
    """

    default_education_settings = get_default_education_settings(
        migration_path_key=migration_path_key,
        migration_path_defaults=migration_path_defaults
    )

    normalized = {
        **default_education_settings,

        "pr_override_enabled": True,
        "pr_application_year": get_default_pr_application_year(migration_path_key),

        "car_timing_override_enabled": False,
        "buy_car": None,
        "car_purchase_year": None,
        "car_purchase_after_positive_cash_flow": False,

        "first_child_override_enabled": False,
        "first_child_year_override": None,
        "second_child_override_enabled": False,
        "second_child_year_override": None,

        # Model-only assumption.
        # Dataset has family_with_one_child_monthly only.
        # Two-child family living cost is estimated as one-child cost * 1.20.
        "family_with_two_children_living_multiplier": 1.20,

        "investment_split": "Dataset default",
        "investment_percentage": 0.0,

        **scenario_overrides
    }

    if normalized.get("education_study_years") is None:
        normalized["education_study_years"] = []

    if normalized.get("tuition_load") is None:
        normalized["tuition_load"] = 0.0

    if normalized.get("family_with_two_children_living_multiplier") is None:
        normalized["family_with_two_children_living_multiplier"] = 1.20

    return normalized


def build_scenario_config(
    dataset: Dict[str, Any],
    migration_path_label: str,
    life_scenario_label: str,
    car_option_label: str,
    investment_option_label: str,
    spouse_income_case_label: str,
    salary_growth_rate: float,
    inflation_rate: float,
    investment_return_rate: float,
    rent_multiplier: float,
    tuition_multiplier: float,
    childcare_multiplier: float,
    education_mode_label: Optional[str] = None,
    pr_timing_label: Optional[str] = None,
    custom_pr_year: Optional[int] = None,
    car_purchase_timing_label: Optional[str] = None,
    first_child_timing_label: Optional[str] = None,
    second_child_timing_label: Optional[str] = None,
    investment_split_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build one clean scenario configuration object.

    The JSON dataset remains the base truth.
    Advanced UI options are stored only in:

        scenario_config["scenario_overrides"]

    This keeps the dataset unchanged and makes the simulation flexible.
    """

    migration_path_key = MIGRATION_PATH_OPTIONS[migration_path_label]
    life_scenario_key = LIFE_SCENARIO_OPTIONS[life_scenario_label]
    spouse_income_case_key = SPOUSE_INCOME_OPTIONS[spouse_income_case_label]

    old_investment_method = INVESTMENT_OPTIONS[investment_option_label]

    migration_path_defaults = dataset["scenario_defaults"][migration_path_key]
    life_scenario_defaults = dataset["scenario_defaults"]["life_scenarios"][
        life_scenario_key
    ]

    spouse_income_percentage = dataset["income"]["spouse_income"]["income_cases"][
        spouse_income_case_key
    ]

    education_overrides = resolve_education_mode(
        education_mode_label=education_mode_label
    )

    pr_overrides = resolve_pr_application_year(
        pr_timing_label=pr_timing_label,
        migration_path_key=migration_path_key,
        custom_pr_year=custom_pr_year
    )

    car_timing_overrides = resolve_car_purchase_timing(
        car_purchase_timing_label=car_purchase_timing_label
    )

    child_timing_overrides = resolve_child_timing(
        first_child_timing_label=first_child_timing_label,
        second_child_timing_label=second_child_timing_label
    )

    investment_split_overrides = resolve_investment_split(
        investment_split_label=investment_split_label,
        old_investment_method=old_investment_method
    )

    raw_scenario_overrides = {
        **education_overrides,
        **pr_overrides,
        **car_timing_overrides,
        **child_timing_overrides,
        **investment_split_overrides
    }

    scenario_overrides = normalize_scenario_overrides(
        scenario_overrides=raw_scenario_overrides,
        migration_path_key=migration_path_key,
        migration_path_defaults=migration_path_defaults
    )

    if scenario_overrides["car_timing_override_enabled"]:
        buy_car = bool(scenario_overrides["buy_car"])
    else:
        buy_car = CAR_OPTIONS[car_option_label]

    scenario_config = {
        "selected_labels": {
            "migration_path": migration_path_label,
            "life_scenario": life_scenario_label,
            "car_option": car_option_label,
            "investment_option": investment_option_label,
            "spouse_income_case": spouse_income_case_label,
            "education_mode": education_mode_label or "Dataset default",
            "pr_timing": pr_timing_label or "Dataset default",
            "car_purchase_timing": car_purchase_timing_label or "Dataset default",
            "first_child_timing": first_child_timing_label or "Dataset default",
            "second_child_timing": second_child_timing_label or "Dataset default",
            "investment_split": scenario_overrides["investment_split"]
        },

        "selected_keys": {
            "migration_path": migration_path_key,
            "life_scenario": life_scenario_key,
            "investment_option": old_investment_method,
            "spouse_income_case": spouse_income_case_key
        },

        "migration_path_defaults": migration_path_defaults,

        "life_scenario_defaults": life_scenario_defaults,

        "car_settings": {
            "buy_car": buy_car
        },

        "investment_settings": {
            "method": old_investment_method,
            "investment_percentage": scenario_overrides["investment_percentage"]
        },

        "spouse_income_settings": {
            "case": spouse_income_case_key,
            "income_percentage": spouse_income_percentage
        },

        "adjustable_inputs": {
            "salary_growth_rate": salary_growth_rate,
            "inflation_rate": inflation_rate,
            "investment_return_rate": investment_return_rate,
            "rent_multiplier": rent_multiplier,
            "tuition_multiplier": tuition_multiplier,
            "childcare_multiplier": childcare_multiplier
        },

        "scenario_overrides": scenario_overrides
    }

    return scenario_config


def create_scenario_summary(scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a simple flat summary for displaying in Streamlit.
    """

    labels = scenario_config["selected_labels"]
    adjustable_inputs = scenario_config["adjustable_inputs"]
    spouse_income_settings = scenario_config["spouse_income_settings"]
    car_settings = scenario_config["car_settings"]
    investment_settings = scenario_config["investment_settings"]
    scenario_overrides = scenario_config["scenario_overrides"]

    return {
        "Migration Path": labels["migration_path"],
        "Life Scenario": labels["life_scenario"],

        "Education Mode": labels["education_mode"],
        "Education Study Years": scenario_overrides["education_study_years"],
        "Tuition Load": scenario_overrides["tuition_load"],

        "PR Timing": labels["pr_timing"],
        "PR Application Year": scenario_overrides["pr_application_year"],

        "Car Option": labels["car_option"],
        "Car Purchase Timing": labels["car_purchase_timing"],
        "Buy Car": car_settings["buy_car"],
        "Car Purchase Year": scenario_overrides["car_purchase_year"],
        "Car After Positive Cash Flow": scenario_overrides[
            "car_purchase_after_positive_cash_flow"
        ],

        "First Child Timing": labels["first_child_timing"],
        "Second Child Timing": labels["second_child_timing"],
        "First Child Year Override": scenario_overrides["first_child_year_override"],
        "Second Child Year Override": scenario_overrides["second_child_year_override"],

        "Effective Marriage Year": get_marriage_year(scenario_config),
        "Effective First Child Year": get_first_child_year(scenario_config),
        "Effective Second Child Year": get_second_child_year(scenario_config),
        "Two-Child Living Cost Multiplier": scenario_overrides[
            "family_with_two_children_living_multiplier"
        ],

        "Investment Option": labels["investment_option"],
        "Investment Split": labels["investment_split"],
        "Investment Percentage": investment_settings["investment_percentage"],

        "Spouse Income Case": labels["spouse_income_case"],
        "Spouse Income Percentage": spouse_income_settings["income_percentage"],

        "Salary Growth Rate": adjustable_inputs["salary_growth_rate"],
        "Inflation Rate": adjustable_inputs["inflation_rate"],
        "Investment Return Rate": adjustable_inputs["investment_return_rate"],
        "Rent Multiplier": adjustable_inputs["rent_multiplier"],
        "Tuition Multiplier": adjustable_inputs["tuition_multiplier"],
        "Childcare Multiplier": adjustable_inputs["childcare_multiplier"]
    }