from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.country_manager import get_country_config
from src.data_loader import load_dataset
from src.scenario_builder import build_scenario_config, get_default_model_inputs
from src.scenario_options import EDUCATION_MODE_OPTIONS
from src.income_model import calculate_yearly_income
from src.expense_model import calculate_yearly_expenses
from src.nav_model import calculate_nav_simulation, get_nav_summary
from src.country_comparison import _calculate_risk_score
from src.comparison_model import get_dataset_country, get_dataset_currency
from src.currency_utils import convert_local_to_lkr


RISK_PREFERENCE_OPTIONS = ["Low", "Medium", "High"]
PATH_PREFERENCE_OPTIONS = ["No preference", "Study path", "Work path"]
FAMILY_STATUS_OPTIONS = ["Single", "Married / partner"]
CHILDREN_PLAN_OPTIONS = [
    "No children",
    "One child",
    "Two children",
    "Not sure"
]
SALARY_OUTLOOK_OPTIONS = {
    "Conservative salary growth": -0.02,
    "Dataset default salary growth": 0.0,
    "Optimistic salary growth": 0.02,
    "Strong salary growth": 0.04
}

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"

SCORING_EXPLANATION = (
    "Python ranks options using final NAV present value in LKR, risk score, "
    "break-even timing, highest debt, tuition, rent, and the user's risk "
    "preference. OpenAI is used only to explain the already-ranked results."
)

AI_RECOMMENDER_INSTRUCTIONS = """
You are explaining migration path simulation results in simple English.

Hard rules:
- Use only the JSON summary provided by the app.
- Do not calculate, recalculate, estimate, convert, or invent any financial number.
- Do not invent country rules, visa promises, job outcomes, or tuition values.
- If a number is missing, say it is not available.
- Explain the ranking as a recommendation, not as guaranteed financial advice.

Your response must include:
- Best recommended country and path
- Why it is best
- Second and third options
- Main risks
- What to avoid
- Simple next steps
"""


def get_optional_openai_api_key(secrets: Any = None) -> str:
    """
    Read an optional OpenAI API key without requiring it.

    Streamlit secrets are checked first, then the process environment. Blank
    values are treated as missing so the app can use its non-AI fallback.
    """

    secret_value = ""

    if secrets is not None:
        try:
            secret_value = str(secrets["OPENAI_API_KEY"]).strip()
        except Exception:
            secret_value = ""

    if secret_value:
        return secret_value

    return str(os.environ.get("OPENAI_API_KEY", "")).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_sum_from_candidates(
    df: pd.DataFrame,
    candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or getattr(df, "empty", True):
        return default

    for column_name in candidates:
        if column_name in df.columns:
            values = pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)
            return float(values.sum())

    return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalize_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = float(numeric_series.min())
    max_value = float(numeric_series.max())

    if max_value == min_value:
        return pd.Series([0.5] * len(numeric_series), index=series.index)

    if higher_is_better:
        return (numeric_series - min_value) / (max_value - min_value)

    return (max_value - numeric_series) / (max_value - min_value)


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, float):
        return round(value, 2)

    return value


def _compact_dict(row: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {
        key: _json_safe_value(row.get(key))
        for key in keys
    }


def _format_lkr(value: Any) -> str:
    return f"LKR {int(round(_safe_float(value))):,}"


def _format_break_even(value: Any) -> str:
    if value in [None, "", "No break-even"]:
        return "No break-even"

    try:
        return f"Year {int(value)}"
    except Exception:
        return "No break-even"


def get_salary_adjustment(label: str) -> float:
    return float(SALARY_OUTLOOK_OPTIONS.get(label, 0.0))


def resolve_migration_paths(path_preference: str) -> List[str]:
    if path_preference == "Study path":
        return ["Student visa path"]

    if path_preference == "Work path":
        return ["Working visa path"]

    return ["Student visa path", "Working visa path"]


def resolve_life_scenario(
    family_status: str,
    children_plan: str
) -> str:
    if children_plan == "One child":
        return "Married one child"

    if children_plan == "Two children":
        return "Married two children"

    if family_status == "Married / partner":
        return "Married no child"

    return "Single"


def resolve_child_timing(children_plan: str) -> Dict[str, str]:
    if children_plan == "One child":
        return {
            "first_child_timing_label": "First child Year 7",
            "second_child_timing_label": "No second child"
        }

    if children_plan == "Two children":
        return {
            "first_child_timing_label": "First child Year 7",
            "second_child_timing_label": "Second child Year 9"
        }

    if children_plan == "No children":
        return {
            "first_child_timing_label": "No child",
            "second_child_timing_label": "No second child"
        }

    return {
        "first_child_timing_label": "Dataset default",
        "second_child_timing_label": "Dataset default"
    }


def _get_student_education_label() -> str:
    for option in EDUCATION_MODE_OPTIONS:
        normalized = str(option).lower()
        if "master" in normalized and "full-time" in normalized:
            return option

    return "No further study"


def _get_education_mode_for_path(migration_path_label: str) -> str:
    if migration_path_label == "Student visa path":
        return _get_student_education_label()

    return "No further study"


def _get_base_option(
    base_options: Dict[str, Any],
    key: str,
    default: Any
) -> Any:
    value = base_options.get(key, default)
    if value is None:
        return default
    return value


def _build_candidate_record(
    country_name: str,
    migration_path_label: str,
    life_scenario_label: str,
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame
) -> Dict[str, Any]:
    nav_summary = get_nav_summary(nav_df)
    currency = get_dataset_currency(dataset)
    country = get_dataset_country(dataset)

    final_nav_local = _safe_float(nav_summary.get("year_10_nav"))
    final_nav_lkr = _safe_float(nav_summary.get("year_10_nav_lkr"))
    final_nav_present_value_lkr = _safe_float(
        nav_summary.get("year_10_nav_present_value_lkr")
    )

    highest_debt_local = _safe_float(nav_summary.get("highest_debt_amount"))
    highest_debt_lkr = _safe_float(nav_summary.get("highest_debt_amount_lkr"))
    break_even_year = nav_summary.get("break_even_year")

    total_tuition_local = _safe_sum_from_candidates(
        expense_df,
        ["Tuition", "Tuition Cost", "Education Cost"]
    )
    total_rent_local = _safe_sum_from_candidates(
        expense_df,
        ["Rent", "Housing", "Accommodation"]
    )

    total_tuition_lkr = convert_local_to_lkr(
        amount=total_tuition_local,
        dataset=dataset
    )
    total_rent_lkr = convert_local_to_lkr(
        amount=total_rent_local,
        dataset=dataset
    )

    risk_score = _calculate_risk_score(
        nav_df=nav_df,
        income_df=income_df,
        final_nav_local=final_nav_local,
        highest_debt=highest_debt_local
    )

    return {
        "Country": country,
        "Country Input": country_name,
        "Currency": currency,
        "Migration Path": migration_path_label,
        "Life Scenario": life_scenario_label,
        "Education Mode": scenario_config["selected_labels"].get("education_mode"),
        "Final NAV Local": round(final_nav_local, 2),
        "Final NAV LKR": round(final_nav_lkr, 2),
        "Final NAV Present Value LKR": round(final_nav_present_value_lkr, 2),
        "Risk Score": round(float(risk_score), 2),
        "Break-even Year": break_even_year if break_even_year is not None else "No break-even",
        "Highest Debt Local": round(highest_debt_local, 2),
        "Highest Debt LKR": round(highest_debt_lkr, 2),
        "Total Tuition Local": round(total_tuition_local, 2),
        "Total Tuition LKR": round(total_tuition_lkr, 2),
        "Total Rent Local": round(total_rent_local, 2),
        "Total Rent LKR": round(total_rent_lkr, 2),
        "Exchange Rate to LKR": round(
            _safe_float(nav_summary.get("exchange_rate_to_lkr"), 1.0),
            4
        ),
        "Salary Growth Rate": round(
            _safe_float(scenario_config["adjustable_inputs"]["salary_growth_rate"]),
            4
        ),
        "Inflation Rate": round(
            _safe_float(scenario_config["adjustable_inputs"]["inflation_rate"]),
            4
        ),
        "Investment Return Rate": round(
            _safe_float(
                scenario_config["adjustable_inputs"]["investment_return_rate"]
            ),
            4
        ),
        "Status": "OK",
        "Error": ""
    }


def build_candidate_options(
    selected_countries: List[str],
    user_profile: Dict[str, Any],
    base_options: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate country/path candidate simulations using existing model code.
    """

    records: List[Dict[str, Any]] = []
    migration_paths = resolve_migration_paths(
        str(user_profile.get("path_preference", "No preference"))
    )
    children_plan = str(user_profile.get("children_plan", "No children"))
    family_status = str(user_profile.get("family_status", "Single"))
    life_scenario_label = resolve_life_scenario(
        family_status=family_status,
        children_plan=children_plan
    )
    child_timing = resolve_child_timing(children_plan)

    for country_name in selected_countries:
        try:
            country_config = get_country_config(country_name)
            dataset = load_dataset(country_config["dataset_path"])
            country_defaults = get_default_model_inputs(dataset)

        except Exception as error:
            records.append(
                {
                    "Country": country_name,
                    "Country Input": country_name,
                    "Currency": "N/A",
                    "Migration Path": "N/A",
                    "Life Scenario": life_scenario_label,
                    "Final NAV Present Value LKR": None,
                    "Risk Score": 100.0,
                    "Break-even Year": None,
                    "Highest Debt LKR": None,
                    "Total Tuition LKR": None,
                    "Total Rent LKR": None,
                    "Status": "Failed",
                    "Error": str(error)
                }
            )
            continue

        salary_growth_rate = _clamp(
            _safe_float(country_defaults["salary_growth_rate"])
            + _safe_float(user_profile.get("salary_growth_adjustment")),
            0.0,
            0.15
        )

        for migration_path_label in migration_paths:
            try:
                scenario_config = build_scenario_config(
                    dataset=dataset,
                    migration_path_label=migration_path_label,
                    life_scenario_label=life_scenario_label,
                    car_option_label=_get_base_option(
                        base_options,
                        "car_option_label",
                        "No car"
                    ),
                    investment_option_label=_get_base_option(
                        base_options,
                        "investment_option_label",
                        "Invest positive cash flow"
                    ),
                    spouse_income_case_label=_get_base_option(
                        base_options,
                        "spouse_income_case_label",
                        "Moderate"
                    ),
                    salary_growth_rate=salary_growth_rate,
                    inflation_rate=_safe_float(country_defaults["inflation_rate"]),
                    investment_return_rate=_safe_float(
                        country_defaults["investment_return_rate"]
                    ),
                    rent_multiplier=_safe_float(
                        base_options.get("rent_multiplier"),
                        1.0
                    ),
                    tuition_multiplier=_safe_float(
                        base_options.get("tuition_multiplier"),
                        1.0
                    ),
                    childcare_multiplier=_safe_float(
                        base_options.get("childcare_multiplier"),
                        1.0
                    ),
                    education_mode_label=_get_education_mode_for_path(
                        migration_path_label
                    ),
                    pr_timing_label=_get_base_option(
                        base_options,
                        "pr_timing_label",
                        "Normal PR"
                    ),
                    custom_pr_year=base_options.get("custom_pr_year"),
                    car_purchase_timing_label=_get_base_option(
                        base_options,
                        "car_purchase_timing_label",
                        "No car"
                    ),
                    first_child_timing_label=child_timing[
                        "first_child_timing_label"
                    ],
                    second_child_timing_label=child_timing[
                        "second_child_timing_label"
                    ],
                    investment_split_label=_get_base_option(
                        base_options,
                        "investment_split_label",
                        "Invest 100%"
                    )
                )

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

                records.append(
                    _build_candidate_record(
                        country_name=country_name,
                        migration_path_label=migration_path_label,
                        life_scenario_label=life_scenario_label,
                        dataset=dataset,
                        scenario_config=scenario_config,
                        income_df=income_df,
                        expense_df=expense_df,
                        nav_df=nav_df
                    )
                )

            except Exception as error:
                records.append(
                    {
                        "Country": country_name,
                        "Country Input": country_name,
                        "Currency": country_config.get("currency", "N/A"),
                        "Migration Path": migration_path_label,
                        "Life Scenario": life_scenario_label,
                        "Final NAV Present Value LKR": None,
                        "Risk Score": 100.0,
                        "Break-even Year": None,
                        "Highest Debt LKR": None,
                        "Total Tuition LKR": None,
                        "Total Rent LKR": None,
                        "Status": "Failed",
                        "Error": str(error)
                    }
                )

    return pd.DataFrame(records)


def _get_scoring_weights(risk_preference: str) -> Dict[str, float]:
    if risk_preference == "Low":
        return {
            "nav": 0.35,
            "risk": 0.25,
            "break_even": 0.15,
            "debt": 0.10,
            "tuition": 0.06,
            "rent": 0.04,
            "profile": 0.05
        }

    if risk_preference == "High":
        return {
            "nav": 0.62,
            "risk": 0.10,
            "break_even": 0.08,
            "debt": 0.06,
            "tuition": 0.06,
            "rent": 0.04,
            "profile": 0.04
        }

    return {
        "nav": 0.48,
        "risk": 0.18,
        "break_even": 0.12,
        "debt": 0.08,
        "tuition": 0.06,
        "rent": 0.04,
        "profile": 0.04
    }


def _break_even_fit(value: Any) -> float:
    if value in [None, "", "No break-even"]:
        return 0.0

    try:
        year = int(value)
    except Exception:
        return 0.0

    return _clamp((11 - year) / 10, 0.0, 1.0)


def _profile_debt_fit(
    current_savings_lkr: float,
    existing_debt_lkr: float,
    highest_debt_lkr: float
) -> float:
    if existing_debt_lkr <= current_savings_lkr:
        return 1.0

    uncovered_debt = existing_debt_lkr - current_savings_lkr
    pressure_base = max(abs(highest_debt_lkr), current_savings_lkr, 1.0)

    return _clamp(1 - (uncovered_debt / pressure_base), 0.0, 1.0)


def rank_candidate_options(
    candidate_df: pd.DataFrame,
    user_profile: Dict[str, Any]
) -> pd.DataFrame:
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame()

    ok_df = candidate_df[candidate_df["Status"] == "OK"].copy()

    if ok_df.empty:
        return ok_df

    risk_preference = str(user_profile.get("risk_preference", "Medium"))
    weights = _get_scoring_weights(risk_preference)

    ok_df["nav_fit"] = _normalize_series(
        ok_df["Final NAV Present Value LKR"],
        higher_is_better=True
    )
    ok_df["risk_fit"] = 1 - (
        pd.to_numeric(ok_df["Risk Score"], errors="coerce").fillna(100.0) / 100
    )
    ok_df["break_even_fit"] = ok_df["Break-even Year"].apply(_break_even_fit)
    ok_df["debt_fit"] = _normalize_series(
        ok_df["Highest Debt LKR"],
        higher_is_better=False
    )
    ok_df["tuition_fit"] = _normalize_series(
        ok_df["Total Tuition LKR"],
        higher_is_better=False
    )
    ok_df["rent_fit"] = _normalize_series(
        ok_df["Total Rent LKR"],
        higher_is_better=False
    )

    current_savings_lkr = _safe_float(user_profile.get("current_savings_lkr"))
    existing_debt_lkr = _safe_float(user_profile.get("existing_debt_lkr"))

    ok_df["profile_fit"] = ok_df["Highest Debt LKR"].apply(
        lambda value: _profile_debt_fit(
            current_savings_lkr=current_savings_lkr,
            existing_debt_lkr=existing_debt_lkr,
            highest_debt_lkr=_safe_float(value)
        )
    )

    ok_df["Recommendation Score"] = (
        100
        * (
            (ok_df["nav_fit"] * weights["nav"])
            + (ok_df["risk_fit"] * weights["risk"])
            + (ok_df["break_even_fit"] * weights["break_even"])
            + (ok_df["debt_fit"] * weights["debt"])
            + (ok_df["tuition_fit"] * weights["tuition"])
            + (ok_df["rent_fit"] * weights["rent"])
            + (ok_df["profile_fit"] * weights["profile"])
        )
    ).round(2)

    ranked_df = ok_df.sort_values(
        by=[
            "Recommendation Score",
            "Final NAV Present Value LKR",
            "Risk Score"
        ],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    ranked_df["Rank"] = ranked_df.index + 1

    return ranked_df


def _risk_threshold_for_profile(risk_preference: str) -> float:
    if risk_preference == "Low":
        return 45.0

    if risk_preference == "High":
        return 85.0

    return 65.0


def _build_main_risks(
    ranked_df: pd.DataFrame,
    user_profile: Dict[str, Any]
) -> List[str]:
    risks: List[str] = []

    if ranked_df.empty:
        return ["No successful recommendation simulations were generated."]

    top_df = ranked_df.head(3)
    risky_top = top_df[
        pd.to_numeric(top_df["Risk Score"], errors="coerce").fillna(100.0) >= 65
    ]

    if not risky_top.empty:
        countries = ", ".join(risky_top["Country"].astype(str).tolist())
        risks.append(f"High model risk score appears in top options: {countries}.")

    no_break_even_df = top_df[top_df["Break-even Year"] == "No break-even"]
    if not no_break_even_df.empty:
        countries = ", ".join(no_break_even_df["Country"].astype(str).tolist())
        risks.append(f"No break-even within 10 years for: {countries}.")

    current_savings_lkr = _safe_float(user_profile.get("current_savings_lkr"))
    existing_debt_lkr = _safe_float(user_profile.get("existing_debt_lkr"))

    if existing_debt_lkr > current_savings_lkr:
        risks.append(
            "Existing debt is higher than current savings, so lower-debt options "
            "should be treated more carefully."
        )

    if not risks:
        risks.append(
            "Main risks are still rent, tuition, debt pressure, and salary growth "
            "sensitivity from the simulation assumptions."
        )

    return risks


def build_personalized_recommendations(
    selected_countries: List[str],
    user_profile: Dict[str, Any],
    base_options: Dict[str, Any]
) -> Dict[str, Any]:
    if not selected_countries:
        return {
            "candidate_options_df": pd.DataFrame(),
            "ranked_options_df": pd.DataFrame(),
            "top_options": [],
            "risky_options": [],
            "failed_options": [],
            "main_risks": ["No countries were selected."],
            "scoring_explanation": SCORING_EXPLANATION
        }

    candidate_df = build_candidate_options(
        selected_countries=selected_countries,
        user_profile=user_profile,
        base_options=base_options
    )
    ranked_df = rank_candidate_options(
        candidate_df=candidate_df,
        user_profile=user_profile
    )

    top_options = ranked_df.head(3).to_dict("records") if not ranked_df.empty else []

    risk_threshold = _risk_threshold_for_profile(
        str(user_profile.get("risk_preference", "Medium"))
    )

    risky_df = ranked_df[
        (
            pd.to_numeric(ranked_df["Risk Score"], errors="coerce").fillna(100.0)
            >= risk_threshold
        )
        | (
            pd.to_numeric(
                ranked_df["Final NAV Present Value LKR"],
                errors="coerce"
            ).fillna(0.0)
            < 0
        )
    ].copy() if not ranked_df.empty else pd.DataFrame()

    if not risky_df.empty:
        risky_df = risky_df[~risky_df["Rank"].isin([1, 2, 3])]

    failed_df = candidate_df[
        candidate_df["Status"] != "OK"
    ].copy() if not candidate_df.empty else pd.DataFrame()

    return {
        "candidate_options_df": candidate_df,
        "ranked_options_df": ranked_df,
        "top_options": top_options,
        "risky_options": risky_df.head(3).to_dict("records")
        if not risky_df.empty
        else [],
        "failed_options": failed_df.to_dict("records")
        if not failed_df.empty
        else [],
        "main_risks": _build_main_risks(
            ranked_df=ranked_df,
            user_profile=user_profile
        ),
        "scoring_explanation": SCORING_EXPLANATION
    }


def build_recommender_payload(
    user_profile: Dict[str, Any],
    recommendation_result: Dict[str, Any]
) -> Dict[str, Any]:
    option_keys = [
        "Rank",
        "Country",
        "Currency",
        "Migration Path",
        "Life Scenario",
        "Recommendation Score",
        "Final NAV Present Value LKR",
        "Risk Score",
        "Break-even Year",
        "Highest Debt LKR",
        "Total Tuition LKR",
        "Total Rent LKR"
    ]

    profile_summary = {
        "current_savings_lkr": _safe_float(
            user_profile.get("current_savings_lkr")
        ),
        "existing_debt_lkr": _safe_float(user_profile.get("existing_debt_lkr")),
        "salary_outlook": user_profile.get("salary_outlook"),
        "salary_growth_adjustment": _safe_float(
            user_profile.get("salary_growth_adjustment")
        ),
        "family_status": user_profile.get("family_status"),
        "children_plan": user_profile.get("children_plan"),
        "risk_preference": user_profile.get("risk_preference"),
        "preferred_countries": user_profile.get("preferred_countries", []),
        "path_preference": user_profile.get("path_preference")
    }

    return {
        "user_profile": profile_summary,
        "top_3_ranked_options": [
            _compact_dict(option, option_keys)
            for option in recommendation_result.get("top_options", [])
        ],
        "rejected_risky_options": [
            _compact_dict(option, option_keys)
            for option in recommendation_result.get("risky_options", [])
        ],
        "main_risks": recommendation_result.get("main_risks", []),
        "scoring_explanation": recommendation_result.get(
            "scoring_explanation",
            SCORING_EXPLANATION
        )
    }


def build_recommendation_display_dataframe(
    recommendation_result: Dict[str, Any]
) -> pd.DataFrame:
    ranked_df = recommendation_result.get("ranked_options_df")

    if ranked_df is None or getattr(ranked_df, "empty", True):
        return pd.DataFrame()

    display_columns = [
        "Rank",
        "Country",
        "Migration Path",
        "Life Scenario",
        "Recommendation Score",
        "Final NAV Present Value LKR",
        "Risk Score",
        "Break-even Year",
        "Highest Debt LKR",
        "Total Tuition LKR",
        "Total Rent LKR"
    ]

    display_df = ranked_df[display_columns].copy()

    currency_columns = [
        "Final NAV Present Value LKR",
        "Highest Debt LKR",
        "Total Tuition LKR",
        "Total Rent LKR"
    ]

    for column_name in currency_columns:
        display_df[column_name] = display_df[column_name].apply(_format_lkr)

    display_df["Break-even Year"] = display_df["Break-even Year"].apply(
        _format_break_even
    )

    return display_df


def build_option_records_display_dataframe(
    options: List[Dict[str, Any]]
) -> pd.DataFrame:
    if not options:
        return pd.DataFrame()

    return build_recommendation_display_dataframe(
        {
            "ranked_options_df": pd.DataFrame(options)
        }
    )


def build_fallback_explanation(
    recommendation_result: Dict[str, Any]
) -> str:
    top_options = recommendation_result.get("top_options", [])

    if not top_options:
        return (
            "No successful recommendation could be generated. Check the selected "
            "countries and input values, then try again."
        )

    best = top_options[0]
    second = top_options[1] if len(top_options) > 1 else None
    third = top_options[2] if len(top_options) > 2 else None
    main_risks = recommendation_result.get("main_risks", [])

    lines = [
        "### Best recommended country and path",
        (
            f"{best['Country']} - {best['Migration Path']} is ranked first. "
            f"It has a recommendation score of "
            f"{_safe_float(best.get('Recommendation Score')):.2f}, final NAV "
            f"present value of {_format_lkr(best.get('Final NAV Present Value LKR'))}, "
            f"risk score {_safe_float(best.get('Risk Score')):.2f}, and "
            f"break-even: {_format_break_even(best.get('Break-even Year'))}."
        ),
        "",
        "### Why it is best",
        (
            "The ranking is based on the existing Python simulation results: "
            "final NAV present value in LKR, risk score, break-even timing, "
            "highest debt, tuition, and rent. OpenAI did not calculate these values."
        )
    ]

    if second:
        lines.extend(
            [
                "",
                "### Second option",
                (
                    f"{second['Country']} - {second['Migration Path']} with "
                    f"score {_safe_float(second.get('Recommendation Score')):.2f}, "
                    f"final NAV present value "
                    f"{_format_lkr(second.get('Final NAV Present Value LKR'))}, "
                    f"and risk score {_safe_float(second.get('Risk Score')):.2f}."
                )
            ]
        )

    if third:
        lines.extend(
            [
                "",
                "### Third option",
                (
                    f"{third['Country']} - {third['Migration Path']} with "
                    f"score {_safe_float(third.get('Recommendation Score')):.2f}, "
                    f"final NAV present value "
                    f"{_format_lkr(third.get('Final NAV Present Value LKR'))}, "
                    f"and risk score {_safe_float(third.get('Risk Score')):.2f}."
                )
            ]
        )

    lines.extend(
        [
            "",
            "### Main risks",
            "\n".join(f"- {risk}" for risk in main_risks),
            "",
            "### What to avoid",
            "- Avoid options with no break-even year if your risk preference is low.",
            "- Avoid high-debt options if your existing debt is already large.",
            "- Avoid treating the result as guaranteed visa, job, or investment advice.",
            "",
            "### Simple next steps",
            "- Review the top 3 table.",
            "- Change salary outlook, countries, and risk preference to test sensitivity.",
            "- Compare the recommendation with the full scenario and country comparison tabs."
        ]
    )

    return "\n".join(lines)


def _extract_response_text(response: Any) -> str:
    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return text

    output_items = getattr(response, "output", []) or []
    chunks: List[str] = []

    for item in output_items:
        content_items = getattr(item, "content", []) or []
        for content in content_items:
            candidate = getattr(content, "text", None)
            if candidate:
                chunks.append(str(candidate))

    return "\n".join(chunks).strip()


def generate_ai_explanation(
    payload: Dict[str, Any],
    api_key: str,
    fallback_text: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ask OpenAI to explain ranked results after Python has calculated them.
    """

    if not api_key:
        return {
            "text": fallback_text,
            "used_ai": False,
            "error": ""
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model or os.environ.get(
                "OPENAI_RECOMMENDER_MODEL",
                DEFAULT_OPENAI_MODEL
            ),
            instructions=AI_RECOMMENDER_INSTRUCTIONS.strip(),
            input=json.dumps(payload, separators=(",", ":")),
            max_output_tokens=900,
            store=False
        )

        response_text = _extract_response_text(response)

        if not response_text:
            raise ValueError("OpenAI returned an empty explanation.")

        return {
            "text": response_text,
            "used_ai": True,
            "error": ""
        }

    except Exception as error:
        return {
            "text": fallback_text,
            "used_ai": False,
            "error": str(error)
        }
