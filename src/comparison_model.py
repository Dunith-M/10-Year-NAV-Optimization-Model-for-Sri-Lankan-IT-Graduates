from typing import Any, Dict, List, Optional

import pandas as pd

from src.scenario_builder import (
    build_scenario_config,
    MIGRATION_PATH_OPTIONS,
    LIFE_SCENARIO_OPTIONS
)

from src.income_model import calculate_yearly_income
from src.expense_model import calculate_yearly_expenses
from src.nav_model import calculate_nav_simulation, get_nav_summary


def get_value(dataset: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely read a nested value from the dataset.
    """

    current_value = dataset

    try:
        for key in path.split("."):
            current_value = current_value[key]
        return current_value
    except (KeyError, TypeError):
        return default


def get_dataset_metadata(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return dataset.get("metadata", {})


def get_dataset_country(dataset: Dict[str, Any]) -> str:
    metadata = get_dataset_metadata(dataset)

    return str(
        metadata.get("country")
        or metadata.get("country_name")
        or dataset.get("country")
        or "Selected Country"
    )


def get_dataset_currency(dataset: Dict[str, Any]) -> str:
    metadata = get_dataset_metadata(dataset)

    return str(
        metadata.get("currency")
        or dataset.get("currency")
        or "LOCAL"
    ).upper()


def _extract_numeric_value(value: Any) -> Optional[float]:
    """
    Extract numeric value from raw value or {'value': x}.
    """

    if isinstance(value, dict):
        value = value.get("value")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _search_exchange_rate_recursively(data: Any) -> Optional[float]:
    """
    Search the dataset for a field that looks like local-currency-to-LKR exchange rate.
    """

    if isinstance(data, dict):
        for key, value in data.items():
            key_text = str(key).lower()

            if "to_lkr" in key_text and "exchange" in key_text:
                numeric_value = _extract_numeric_value(value)
                if numeric_value is not None:
                    return numeric_value

            if "lkr_exchange_rate" in key_text:
                numeric_value = _extract_numeric_value(value)
                if numeric_value is not None:
                    return numeric_value

            nested_value = _search_exchange_rate_recursively(value)
            if nested_value is not None:
                return nested_value

    elif isinstance(data, list):
        for item in data:
            nested_value = _search_exchange_rate_recursively(item)
            if nested_value is not None:
                return nested_value

    return None


def get_exchange_rate_from_dataset(dataset: Dict[str, Any]) -> float:
    """
    Read selected-country local currency to LKR exchange rate from the dataset.
    Falls back to 1.0 so comparison never crashes.
    """

    currency = get_dataset_currency(dataset).lower()

    direct_paths = [
        f"investment_and_economy.{currency}_to_lkr_exchange_rate.value",
        f"investment_and_economy.{currency}_to_lkr.value",
        "investment_and_economy.exchange_rate_to_lkr.value",
        "investment_and_economy.local_to_lkr_exchange_rate.value",
        "metadata.exchange_rate_to_lkr"
    ]

    for path in direct_paths:
        exchange_rate = get_value(dataset, path, None)
        numeric_value = _extract_numeric_value(exchange_rate)

        if numeric_value is not None:
            return numeric_value

    recursive_value = _search_exchange_rate_recursively(dataset)

    if recursive_value is not None:
        return recursive_value

    return 1.0


def looks_like_internal_key(value: Any) -> bool:
    """
    Detect option keys like:
    - student_visa_path
    - working_visa_path
    - married_two_children

    These should not be shown/passed as selectbox labels.
    """

    text = str(value).strip()

    if not text:
        return False

    has_underscore = "_" in text
    is_lowercase = text == text.lower()
    has_no_spaces = " " not in text

    return is_lowercase and has_no_spaces and has_underscore


def humanize_option_key(key: str) -> str:
    known_labels = {
        "student_visa_path": "Student visa path",
        "working_visa_path": "Working visa path",
        "single": "Single",
        "married_no_child": "Married no child",
        "married_one_child": "Married one child",
        "married_two_children": "Married two children"
    }

    if key in known_labels:
        return known_labels[key]

    return key.replace("_", " ").strip().capitalize()


def get_option_labels(options: Any) -> List[str]:
    """
    Convert scenario option constants into display labels.
    """

    if isinstance(options, dict):
        labels = []

        for key, value in options.items():
            key_text = str(key)

            if isinstance(value, dict):
                label = (
                    value.get("label")
                    or value.get("name")
                    or value.get("title")
                )

                if label:
                    labels.append(str(label))
                elif looks_like_internal_key(key_text):
                    labels.append(humanize_option_key(key_text))
                else:
                    labels.append(key_text)

            else:
                value_text = str(value)

                key_is_internal = looks_like_internal_key(key_text)
                value_is_internal = looks_like_internal_key(value_text)

                if value_is_internal and not key_is_internal:
                    labels.append(key_text)

                elif key_is_internal and not value_is_internal:
                    labels.append(value_text)

                elif key_is_internal and value_is_internal:
                    labels.append(humanize_option_key(key_text))

                else:
                    labels.append(key_text)

        return labels

    return [
        humanize_option_key(str(option))
        if looks_like_internal_key(option)
        else str(option)
        for option in options
    ]


def run_single_nav_simulation(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run income, expense, and NAV models for one scenario.
    """

    income_df = calculate_yearly_income(
        dataset=dataset,
        scenario_config=scenario_config
    )

    expense_df = calculate_yearly_expenses(
        dataset=dataset,
        scenario_config=scenario_config
    )

    nav_df = calculate_nav_simulation(
        dataset=dataset,
        scenario_config=scenario_config,
        income_df=income_df,
        expense_df=expense_df
    )

    nav_summary = get_nav_summary(nav_df)

    return {
        "income_df": income_df,
        "expense_df": expense_df,
        "nav_df": nav_df,
        "nav_summary": nav_summary
    }


def get_final_nav_from_summary(nav_summary: Dict[str, Any]) -> float:
    """
    Read final NAV from nav_summary using several likely key names.
    """

    possible_keys = [
        "year_10_nav",
        "final_nav",
        "final_year_10_nav",
        "Final NAV",
        "Year-10 NAV",
        "Year-10 NAV Local",
        "Final NAV Local"
    ]

    for key in possible_keys:
        if key in nav_summary:
            try:
                return float(nav_summary[key])
            except (TypeError, ValueError):
                return 0.0

    return 0.0


def _safe_last_value(
    df: pd.DataFrame,
    column_name: str,
    default: float = 0.0
) -> float:
    if column_name not in df.columns or df.empty:
        return default

    try:
        return float(df.iloc[-1][column_name])
    except (TypeError, ValueError):
        return default


def _safe_last_value_from_candidates(
    df: pd.DataFrame,
    candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or getattr(df, "empty", True):
        return default

    for column_name in candidates:
        if column_name in df.columns:
            return _safe_last_value(df, column_name, default)

    return default


def _safe_sum(df: pd.DataFrame, column_name: str) -> float:
    if column_name not in df.columns:
        return 0.0

    try:
        return float(df[column_name].sum())
    except (TypeError, ValueError):
        return 0.0


def _safe_sum_from_candidates(
    df: pd.DataFrame,
    candidates: List[str]
) -> float:
    if df is None or getattr(df, "empty", True):
        return 0.0

    for column_name in candidates:
        if column_name in df.columns:
            return _safe_sum(df, column_name)

    return 0.0


def _resolve_child_timing_for_life_scenario(
    life_scenario_label: str,
    first_child_timing_label: Optional[str],
    second_child_timing_label: Optional[str]
) -> Dict[str, Optional[str]]:
    """
    Prevent advanced child timing from accidentally turning child-free scenarios
    into child scenarios during comparison.
    """

    normalized_label = life_scenario_label.lower()

    has_one_child = "one child" in normalized_label or "1 child" in normalized_label
    has_two_children = "two children" in normalized_label or "2 children" in normalized_label

    if not has_one_child and not has_two_children:
        return {
            "first_child_timing_label": "No child",
            "second_child_timing_label": "No second child"
        }

    if has_one_child and not has_two_children:
        return {
            "first_child_timing_label": first_child_timing_label,
            "second_child_timing_label": "No second child"
        }

    return {
        "first_child_timing_label": first_child_timing_label,
        "second_child_timing_label": second_child_timing_label
    }


def build_scenario_comparison(
    dataset: Dict[str, Any],
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
) -> pd.DataFrame:
    """
    Compare all migration path x life scenario combinations inside one country.

    This remains single-country scenario comparison:
        Australia student path vs Australia working path
        Germany single vs Germany married
        Japan no car vs car decision is held constant here
    """

    country = get_dataset_country(dataset)
    currency = get_dataset_currency(dataset)
    migration_path_labels = get_option_labels(MIGRATION_PATH_OPTIONS)
    life_scenario_labels = get_option_labels(LIFE_SCENARIO_OPTIONS)
    exchange_rate = get_exchange_rate_from_dataset(dataset)

    records: List[Dict[str, Any]] = []

    for migration_path_label in migration_path_labels:
        for life_scenario_label in life_scenario_labels:
            child_timing = _resolve_child_timing_for_life_scenario(
                life_scenario_label=life_scenario_label,
                first_child_timing_label=first_child_timing_label,
                second_child_timing_label=second_child_timing_label
            )

            scenario_kwargs = {
                "dataset": dataset,
                "migration_path_label": migration_path_label,
                "life_scenario_label": life_scenario_label,
                "car_option_label": car_option_label,
                "investment_option_label": investment_option_label,
                "spouse_income_case_label": spouse_income_case_label,
                "salary_growth_rate": salary_growth_rate,
                "inflation_rate": inflation_rate,
                "investment_return_rate": investment_return_rate,
                "rent_multiplier": rent_multiplier,
                "tuition_multiplier": tuition_multiplier,
                "childcare_multiplier": childcare_multiplier
            }

            optional_kwargs = {
                "education_mode_label": education_mode_label,
                "pr_timing_label": pr_timing_label,
                "custom_pr_year": custom_pr_year,
                "car_purchase_timing_label": car_purchase_timing_label,
                "first_child_timing_label": child_timing["first_child_timing_label"],
                "second_child_timing_label": child_timing["second_child_timing_label"],
                "investment_split_label": investment_split_label
            }

            for key, value in optional_kwargs.items():
                if value is not None:
                    scenario_kwargs[key] = value

            scenario_config = build_scenario_config(**scenario_kwargs)

            result = run_single_nav_simulation(
                dataset=dataset,
                scenario_config=scenario_config
            )

            income_df = result["income_df"]
            expense_df = result["expense_df"]
            nav_df = result["nav_df"]
            nav_summary = result["nav_summary"]

            final_nav_local = get_final_nav_from_summary(nav_summary)
            final_nav_lkr = final_nav_local * exchange_rate
            scenario_name = f"{migration_path_label} + {life_scenario_label}"

            records.append(
                {
                    "Country": country,
                    "Currency": currency,
                    "Scenario": scenario_name,
                    "Migration Path": migration_path_label,
                    "Life Scenario": life_scenario_label,
                    "Year-10 NAV Local": round(final_nav_local, 2),
                    f"Year-10 NAV {currency}": round(final_nav_local, 2),
                    "Final NAV Local": round(final_nav_local, 2),
                    f"Final NAV {currency}": round(final_nav_local, 2),
                    "Final NAV": round(final_nav_local, 2),
                    "Year-10 NAV LKR": round(final_nav_lkr, 2),
                    "Final NAV LKR": round(final_nav_lkr, 2),
                    "Final Cash Local": round(
                        _safe_last_value_from_candidates(
                            nav_df,
                            ["Cash Balance", "Cash Savings", "Cash"]
                        ),
                        2
                    ),
                    "Final Investment Local": round(
                        _safe_last_value_from_candidates(
                            nav_df,
                            ["Investment Balance", "Investments"]
                        ),
                        2
                    ),
                    "Final Retirement Local": round(
                        _safe_last_value_from_candidates(
                            nav_df,
                            ["Superannuation Balance", "Superannuation", "Retirement Balance"]
                        ),
                        2
                    ),
                    "Final Total Assets Local": round(
                        _safe_last_value_from_candidates(
                            nav_df,
                            ["Total Assets", "Local Currency Assets"]
                        ),
                        2
                    ),
                    "Final Total Liabilities Local": round(
                        _safe_last_value_from_candidates(
                            nav_df,
                            ["Total Liabilities", "Local Currency Liabilities", "Total Debt"]
                        ),
                        2
                    ),
                    "Total Income Local": round(
                        _safe_sum_from_candidates(
                            income_df,
                            ["Net Income", "Total Net Income"]
                        ),
                        2
                    ),
                    "Total Expenses Local": round(
                        _safe_sum_from_candidates(
                            expense_df,
                            ["Total Expenses", "Total Expense"]
                        ),
                        2
                    )
                }
            )

    comparison_df = pd.DataFrame(records)

    if comparison_df.empty:
        return comparison_df

    comparison_df = rank_scenarios_by_nav(comparison_df)
    return comparison_df


def _get_nav_column(comparison_df: pd.DataFrame) -> str:
    possible_columns = [
        "Year-10 NAV Local",
        "Final NAV Local",
        "Final NAV",
        "Year-10 NAV",
        "NAV"
    ]

    for column_name in possible_columns:
        if column_name in comparison_df.columns:
            return column_name

    raise ValueError("No NAV column found in comparison_df.")


def rank_scenarios_by_nav(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank scenarios by final Year-10 local NAV.
    Highest NAV gets Rank 1.
    """

    if comparison_df.empty:
        return comparison_df.copy()

    nav_column = _get_nav_column(comparison_df)
    ranked_df = comparison_df.copy()
    ranked_df = ranked_df.sort_values(
        by=nav_column,
        ascending=False
    ).reset_index(drop=True)
    ranked_df["Rank"] = ranked_df.index + 1
    ranked_df["Scenario Rank"] = ranked_df["Rank"]

    return ranked_df


def get_best_scenario(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return the highest-NAV scenario as a dictionary.
    """

    if comparison_df.empty:
        return {}

    ranked_df = rank_scenarios_by_nav(comparison_df)
    return ranked_df.iloc[0].to_dict()


def get_worst_scenario(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return the lowest-NAV scenario as a dictionary.
    """

    if comparison_df.empty:
        return {}

    ranked_df = rank_scenarios_by_nav(comparison_df)
    return ranked_df.iloc[-1].to_dict()


def calculate_nav_gap(
    selected_scenario: Dict[str, Any],
    best_scenario: Dict[str, Any],
    nav_column: str = "Year-10 NAV Local"
) -> float:
    """
    Calculate how far the selected scenario is below the best scenario.
    Positive value means selected is below best.
    """

    if not selected_scenario or not best_scenario:
        return 0.0

    selected_nav = float(
        selected_scenario.get(
            nav_column,
            selected_scenario.get("Final NAV Local", selected_scenario.get("Final NAV", 0.0))
        )
    )
    best_nav = float(
        best_scenario.get(
            nav_column,
            best_scenario.get("Final NAV Local", best_scenario.get("Final NAV", 0.0))
        )
    )

    return round(best_nav - selected_nav, 2)


def compare_selected_vs_best(
    comparison_df: pd.DataFrame,
    selected_migration_path_label: str,
    selected_life_scenario_label: str
) -> Dict[str, Any]:
    """
    Compare the user's selected scenario against the best available scenario.
    """

    if comparison_df.empty:
        return {
            "selected_scenario": {},
            "best_scenario": {},
            "worst_scenario": {},
            "selected_rank": None,
            "total_scenarios": 0,
            "nav_gap_local": 0.0,
            "nav_gap_lkr": 0.0,
            "nav_gap_aud": 0.0,
            "message": "No scenario comparison results are available."
        }

    ranked_df = rank_scenarios_by_nav(comparison_df)
    nav_column = _get_nav_column(ranked_df)

    selected_rows = ranked_df[
        (ranked_df["Migration Path"] == selected_migration_path_label)
        & (ranked_df["Life Scenario"] == selected_life_scenario_label)
    ]

    if selected_rows.empty:
        selected_scenario = {}
        selected_rank = None
    else:
        selected_scenario = selected_rows.iloc[0].to_dict()
        selected_rank = int(selected_scenario["Rank"])

    best_scenario = get_best_scenario(ranked_df)
    worst_scenario = get_worst_scenario(ranked_df)

    currency = (
        selected_scenario.get("Currency")
        or best_scenario.get("Currency")
        or "LOCAL"
    )

    nav_gap_local = calculate_nav_gap(
        selected_scenario=selected_scenario,
        best_scenario=best_scenario,
        nav_column=nav_column
    )

    selected_lkr = (
        float(selected_scenario.get("Year-10 NAV LKR", selected_scenario.get("Final NAV LKR", 0.0)))
        if selected_scenario
        else 0.0
    )

    best_lkr = (
        float(best_scenario.get("Year-10 NAV LKR", best_scenario.get("Final NAV LKR", 0.0)))
        if best_scenario
        else 0.0
    )

    nav_gap_lkr = round(best_lkr - selected_lkr, 2)
    total_scenarios = len(ranked_df)

    if selected_rank is None:
        message = "The selected scenario was not found in the comparison table."
    elif nav_gap_local <= 0:
        message = (
            f"Your selected scenario ranks {selected_rank} out of {total_scenarios}. "
            "It is currently the best scenario by Year-10 NAV."
        )
    else:
        message = (
            f"Your selected scenario ranks {selected_rank} out of {total_scenarios}. "
            f"It is {currency} {nav_gap_local:,.0f} below the best scenario."
        )

    return {
        "selected_scenario": selected_scenario,
        "best_scenario": best_scenario,
        "worst_scenario": worst_scenario,
        "selected_rank": selected_rank,
        "total_scenarios": total_scenarios,
        "nav_gap_local": nav_gap_local,
        "nav_gap_lkr": nav_gap_lkr,
        "nav_gap_aud": nav_gap_local,
        "currency": currency,
        "message": message
    }
