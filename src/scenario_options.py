from typing import Any, Dict, Optional


EDUCATION_MODE_OPTIONS = [
    "No further study",
    "Master’s full-time",
    "Master’s part-time",
    "MBA full-time",
    "MBA part-time"
]


PR_TIMING_OPTIONS = [
    "No PR within 10 years",
    "Early PR",
    "Normal PR",
    "Late PR",
    "Custom PR year"
]


CAR_PURCHASE_TIMING_OPTIONS = [
    "No car",
    "Buy car in Year 2",
    "Buy car in Year 3",
    "Buy car in Year 5",
    "Buy car after positive cash flow"
]


FIRST_CHILD_TIMING_OPTIONS = [
    "Dataset default",
    "First child Year 5",
    "First child Year 7",
    "First child Year 9",
    "No child"
]


SECOND_CHILD_TIMING_OPTIONS = [
    "Dataset default",
    "Second child Year 9",
    "No second child"
]


INVESTMENT_SPLIT_OPTIONS = [
    "Save only",
    "Invest 25% of positive cash flow",
    "Invest 50%",
    "Invest 75%",
    "Invest 100%"
]


def resolve_education_mode(
    education_mode_label: Optional[str]
) -> Dict[str, Any]:
    """
    Convert education UI label into model-friendly override values.

    If education_mode_label is None, the model keeps the old dataset/default logic.
    """

    if education_mode_label is None:
        return {
            "education_override_enabled": False,
            "education_mode": "Dataset default",
            "education_program": None,
            "education_study_mode": None,
            "education_study_years": None,
            "tuition_load": None
        }

    if education_mode_label == "No further study":
        return {
            "education_override_enabled": True,
            "education_mode": education_mode_label,
            "education_program": None,
            "education_study_mode": "none",
            "education_study_years": [],
            "tuition_load": 0.0
        }

    if education_mode_label == "Master’s full-time":
        return {
            "education_override_enabled": True,
            "education_mode": education_mode_label,
            "education_program": "masters",
            "education_study_mode": "full_time",
            "education_study_years": [1, 2],
            "tuition_load": 1.0
        }

    if education_mode_label == "Master’s part-time":
        return {
            "education_override_enabled": True,
            "education_mode": education_mode_label,
            "education_program": "masters",
            "education_study_mode": "part_time",
            "education_study_years": [1, 2, 3, 4],
            "tuition_load": 0.5
        }

    if education_mode_label == "MBA full-time":
        return {
            "education_override_enabled": True,
            "education_mode": education_mode_label,
            "education_program": "mba",
            "education_study_mode": "full_time",
            "education_study_years": [1, 2],
            "tuition_load": 1.0
        }

    if education_mode_label == "MBA part-time":
        return {
            "education_override_enabled": True,
            "education_mode": education_mode_label,
            "education_program": "mba",
            "education_study_mode": "part_time",
            "education_study_years": [1, 2, 3, 4],
            "tuition_load": 0.5
        }

    raise ValueError(f"Unknown education mode: {education_mode_label}")


def resolve_pr_application_year(
    pr_timing_label: Optional[str],
    migration_path_key: str,
    custom_pr_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convert PR timing UI label into a PR application year.

    If pr_timing_label is None, the model keeps the old default PR timing.
    """

    if pr_timing_label is None:
        return {
            "pr_override_enabled": False,
            "pr_timing": "Dataset default",
            "pr_application_year": None
        }

    if pr_timing_label == "No PR within 10 years":
        return {
            "pr_override_enabled": True,
            "pr_timing": pr_timing_label,
            "pr_application_year": None
        }

    if pr_timing_label == "Early PR":
        pr_year = 5 if migration_path_key == "student_visa_path" else 3

    elif pr_timing_label == "Normal PR":
        pr_year = 6 if migration_path_key == "student_visa_path" else 4

    elif pr_timing_label == "Late PR":
        pr_year = 8 if migration_path_key == "student_visa_path" else 6

    elif pr_timing_label == "Custom PR year":
        if custom_pr_year is None:
            raise ValueError("Custom PR year was selected, but no custom_pr_year was provided.")

        pr_year = int(custom_pr_year)

        if pr_year < 1 or pr_year > 10:
            raise ValueError("Custom PR year must be between Year 1 and Year 10.")

    else:
        raise ValueError(f"Unknown PR timing option: {pr_timing_label}")

    return {
        "pr_override_enabled": True,
        "pr_timing": pr_timing_label,
        "pr_application_year": pr_year
    }


def resolve_car_purchase_timing(
    car_purchase_timing_label: Optional[str]
) -> Dict[str, Any]:
    """
    Convert car timing UI label into model-friendly override values.

    If car_purchase_timing_label is None, the model keeps old car timing logic.
    """

    if car_purchase_timing_label is None:
        return {
            "car_timing_override_enabled": False,
            "buy_car": None,
            "car_purchase_year": None,
            "car_purchase_after_positive_cash_flow": False
        }

    if car_purchase_timing_label == "No car":
        return {
            "car_timing_override_enabled": True,
            "buy_car": False,
            "car_purchase_year": None,
            "car_purchase_after_positive_cash_flow": False
        }

    if car_purchase_timing_label == "Buy car in Year 2":
        return {
            "car_timing_override_enabled": True,
            "buy_car": True,
            "car_purchase_year": 2,
            "car_purchase_after_positive_cash_flow": False
        }

    if car_purchase_timing_label == "Buy car in Year 3":
        return {
            "car_timing_override_enabled": True,
            "buy_car": True,
            "car_purchase_year": 3,
            "car_purchase_after_positive_cash_flow": False
        }

    if car_purchase_timing_label == "Buy car in Year 5":
        return {
            "car_timing_override_enabled": True,
            "buy_car": True,
            "car_purchase_year": 5,
            "car_purchase_after_positive_cash_flow": False
        }

    if car_purchase_timing_label == "Buy car after positive cash flow":
        return {
            "car_timing_override_enabled": True,
            "buy_car": True,
            "car_purchase_year": None,
            "car_purchase_after_positive_cash_flow": True
        }

    raise ValueError(f"Unknown car purchase timing: {car_purchase_timing_label}")


def resolve_child_timing(
    first_child_timing_label: Optional[str],
    second_child_timing_label: Optional[str]
) -> Dict[str, Any]:
    """
    Convert child timing UI labels into override values.

    None or Dataset default keeps the dataset/default life scenario timing.
    """

    first_override_enabled = (
        first_child_timing_label is not None
        and first_child_timing_label != "Dataset default"
    )

    second_override_enabled = (
        second_child_timing_label is not None
        and second_child_timing_label != "Dataset default"
    )

    first_child_year_override = None
    second_child_year_override = None

    if first_child_timing_label == "First child Year 5":
        first_child_year_override = 5
    elif first_child_timing_label == "First child Year 7":
        first_child_year_override = 7
    elif first_child_timing_label == "First child Year 9":
        first_child_year_override = 9
    elif first_child_timing_label == "No child":
        first_child_year_override = None
        second_override_enabled = True
        second_child_year_override = None
    elif first_child_timing_label in [None, "Dataset default"]:
        pass
    else:
        raise ValueError(f"Unknown first child timing: {first_child_timing_label}")

    if first_child_timing_label != "No child":
        if second_child_timing_label == "Second child Year 9":
            second_child_year_override = 9
        elif second_child_timing_label == "No second child":
            second_child_year_override = None
        elif second_child_timing_label in [None, "Dataset default"]:
            pass
        else:
            raise ValueError(f"Unknown second child timing: {second_child_timing_label}")

    return {
        "first_child_override_enabled": first_override_enabled,
        "first_child_year_override": first_child_year_override,
        "second_child_override_enabled": second_override_enabled,
        "second_child_year_override": second_child_year_override
    }


def resolve_investment_split(
    investment_split_label: Optional[str],
    old_investment_method: str
) -> Dict[str, Any]:
    """
    Convert investment split UI label into investment percentage.

    If investment_split_label is None, keep old behavior:
    - save_only = 0%
    - invest_positive_cash_flow = 100%
    """

    if investment_split_label is None:
        if old_investment_method == "invest_positive_cash_flow":
            investment_percentage = 1.0
            investment_split = "Invest 100%"
        else:
            investment_percentage = 0.0
            investment_split = "Save only"

        return {
            "investment_split": investment_split,
            "investment_percentage": investment_percentage
        }

    mapping = {
        "Save only": 0.0,
        "Invest 25% of positive cash flow": 0.25,
        "Invest 50%": 0.50,
        "Invest 75%": 0.75,
        "Invest 100%": 1.0
    }

    if investment_split_label not in mapping:
        raise ValueError(f"Unknown investment split: {investment_split_label}")

    return {
        "investment_split": investment_split_label,
        "investment_percentage": mapping[investment_split_label]
    }