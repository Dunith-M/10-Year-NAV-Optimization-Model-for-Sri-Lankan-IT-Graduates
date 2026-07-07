from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.country_manager import get_available_countries, get_country_config
from src.data_loader import load_dataset
from src.scenario_builder import build_scenario_config
from src.income_model import calculate_yearly_income
from src.expense_model import calculate_yearly_expenses
from src.nav_model import calculate_nav_simulation, get_nav_summary
from src.comparison_model import (
    get_dataset_country,
    get_dataset_currency,
    get_exchange_rate_from_dataset,
    get_final_nav_from_summary
)


def _safe_last_value_from_candidates(
    df: pd.DataFrame,
    candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or getattr(df, "empty", True):
        return default

    for column in candidates:
        if column in df.columns:
            try:
                return float(df[column].iloc[-1])
            except (TypeError, ValueError):
                return default

    return default


def _safe_min_from_candidates(
    df: pd.DataFrame,
    candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or getattr(df, "empty", True):
        return default

    for column in candidates:
        if column in df.columns:
            try:
                return float(pd.to_numeric(df[column], errors="coerce").min())
            except (TypeError, ValueError):
                return default

    return default


def _safe_max_from_candidates(
    df: pd.DataFrame,
    candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or getattr(df, "empty", True):
        return default

    for column in candidates:
        if column in df.columns:
            try:
                return float(pd.to_numeric(df[column], errors="coerce").max())
            except (TypeError, ValueError):
                return default

    return default


def _safe_sum_from_candidates(
    df: pd.DataFrame,
    candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or getattr(df, "empty", True):
        return default

    for column in candidates:
        if column in df.columns:
            try:
                return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())
            except (TypeError, ValueError):
                return default

    return default


def _get_break_even_year(nav_df: pd.DataFrame) -> Optional[int]:
    nav_candidates = [
        "Local Currency NAV",
        "NAV",
        "Net Asset Value",
        "Final NAV Local"
    ]

    if nav_df is None or nav_df.empty or "Year" not in nav_df.columns:
        return None

    nav_column = None

    for candidate in nav_candidates:
        if candidate in nav_df.columns:
            nav_column = candidate
            break

    if nav_column is None:
        return None

    safe_df = nav_df.copy()
    safe_df[nav_column] = pd.to_numeric(
        safe_df[nav_column],
        errors="coerce"
    )

    break_even_rows = safe_df[safe_df[nav_column] >= 0]

    if break_even_rows.empty:
        return None

    try:
        return int(break_even_rows.iloc[0]["Year"])
    except (TypeError, ValueError):
        return None


def _calculate_risk_score(
    nav_df: pd.DataFrame,
    income_df: pd.DataFrame,
    final_nav_local: float,
    highest_debt: float
) -> float:
    """
    Simple cross-country risk score.
    Higher score = more dangerous.

    This is intentionally simple for viva/demo:
    - Negative NAV years
    - No break-even year
    - Debt pressure relative to total net income
    - Negative final NAV
    """

    if nav_df is None or nav_df.empty:
        return 100.0

    nav_column = None

    for candidate in ["Local Currency NAV", "NAV", "Net Asset Value"]:
        if candidate in nav_df.columns:
            nav_column = candidate
            break

    if nav_column is None:
        return 100.0

    safe_nav = pd.to_numeric(
        nav_df[nav_column],
        errors="coerce"
    ).fillna(0.0)

    negative_nav_years = int((safe_nav < 0).sum())
    horizon = max(len(safe_nav), 1)

    negative_nav_component = min(
        30.0,
        (negative_nav_years / horizon) * 30.0
    )

    break_even_component = 0.0 if _get_break_even_year(nav_df) is not None else 25.0

    total_net_income = _safe_sum_from_candidates(
        income_df,
        ["Net Income", "Total Net Income"],
        default=0.0
    )

    debt_ratio = highest_debt / max(abs(total_net_income), 1.0)
    debt_component = min(25.0, debt_ratio * 100.0)

    final_nav_component = 20.0 if final_nav_local < 0 else 0.0

    return round(
        min(
            100.0,
            negative_nav_component
            + break_even_component
            + debt_component
            + final_nav_component
        ),
        2
    )


def _build_scenario_for_country(
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
    education_mode_label: Optional[str],
    pr_timing_label: Optional[str],
    custom_pr_year: Optional[int],
    car_purchase_timing_label: Optional[str],
    first_child_timing_label: Optional[str],
    second_child_timing_label: Optional[str],
    investment_split_label: Optional[str]
) -> Dict[str, Any]:
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
        "first_child_timing_label": first_child_timing_label,
        "second_child_timing_label": second_child_timing_label,
        "investment_split_label": investment_split_label
    }

    for key, value in optional_kwargs.items():
        if value is not None:
            scenario_kwargs[key] = value

    return build_scenario_config(**scenario_kwargs)


def build_country_comparison(
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
) -> pd.DataFrame:
    """
    Compare the same selected scenario across all registered countries.

    Example:
        Single + working visa + no car + invest
        Australia vs Germany vs Japan vs Sri Lanka
    """

    countries = get_available_countries()
    records: List[Dict[str, Any]] = []

    for country_name in countries:
        try:
            country_config = get_country_config(country_name)
            dataset_path = country_config["dataset_path"]
            dataset = load_dataset(dataset_path)

            country = get_dataset_country(dataset)
            currency = get_dataset_currency(dataset)
            exchange_rate = get_exchange_rate_from_dataset(dataset)

            scenario_config = _build_scenario_for_country(
                dataset=dataset,
                migration_path_label=migration_path_label,
                life_scenario_label=life_scenario_label,
                car_option_label=car_option_label,
                investment_option_label=investment_option_label,
                spouse_income_case_label=spouse_income_case_label,
                salary_growth_rate=salary_growth_rate,
                inflation_rate=inflation_rate,
                investment_return_rate=investment_return_rate,
                rent_multiplier=rent_multiplier,
                tuition_multiplier=tuition_multiplier,
                childcare_multiplier=childcare_multiplier,
                education_mode_label=education_mode_label,
                pr_timing_label=pr_timing_label,
                custom_pr_year=custom_pr_year,
                car_purchase_timing_label=car_purchase_timing_label,
                first_child_timing_label=first_child_timing_label,
                second_child_timing_label=second_child_timing_label,
                investment_split_label=investment_split_label
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

            nav_summary = get_nav_summary(nav_df)

            final_nav_local = get_final_nav_from_summary(nav_summary)
            final_nav_lkr = final_nav_local * exchange_rate
            break_even_year = _get_break_even_year(nav_df)

            lowest_nav = _safe_min_from_candidates(
                nav_df,
                ["Local Currency NAV", "NAV", "Net Asset Value"]
            )

            highest_debt = _safe_max_from_candidates(
                nav_df,
                [
                    "Total Debt",
                    "Total Liabilities",
                    "Local Currency Liabilities",
                    "Liabilities"
                ]
            )

            total_tuition = _safe_sum_from_candidates(
                expense_df,
                ["Tuition", "Tuition Cost", "Education Cost"]
            )

            total_rent = _safe_sum_from_candidates(
                expense_df,
                ["Rent", "Housing", "Accommodation"]
            )

            total_tax = _safe_sum_from_candidates(
                income_df,
                ["Tax", "Income Tax", "Tax Paid"]
            )

            risk_score = _calculate_risk_score(
                nav_df=nav_df,
                income_df=income_df,
                final_nav_local=final_nav_local,
                highest_debt=highest_debt
            )

            records.append(
                {
                    "Country": country,
                    "Currency": currency,
                    "Scenario": f"{migration_path_label} + {life_scenario_label}",
                    "Final NAV Local": round(final_nav_local, 2),
                    f"Final NAV {currency}": round(final_nav_local, 2),
                    "Final NAV LKR": round(final_nav_lkr, 2),
                    "Break-even Year": break_even_year if break_even_year is not None else "No break-even",
                    "Lowest NAV": round(lowest_nav, 2),
                    "Highest Debt": round(highest_debt, 2),
                    "Total Tuition": round(total_tuition, 2),
                    "Total Rent": round(total_rent, 2),
                    "Total Tax": round(total_tax, 2),
                    "Risk Score": risk_score,
                    "Exchange Rate to LKR": exchange_rate,
                    "Status": "OK",
                    "Error": ""
                }
            )

        except Exception as error:
            records.append(
                {
                    "Country": country_name,
                    "Currency": country_config.get("currency", "N/A") if "country_config" in locals() else "N/A",
                    "Scenario": f"{migration_path_label} + {life_scenario_label}",
                    "Final NAV Local": None,
                    "Final NAV LKR": None,
                    "Break-even Year": None,
                    "Lowest NAV": None,
                    "Highest Debt": None,
                    "Total Tuition": None,
                    "Total Rent": None,
                    "Total Tax": None,
                    "Risk Score": 100.0,
                    "Exchange Rate to LKR": None,
                    "Status": "Failed",
                    "Error": str(error)
                }
            )

    comparison_df = pd.DataFrame(records)

    if comparison_df.empty:
        return comparison_df

    successful_df = comparison_df[comparison_df["Status"] == "OK"].copy()
    failed_df = comparison_df[comparison_df["Status"] != "OK"].copy()

    if not successful_df.empty:
        successful_df = successful_df.sort_values(
            by="Final NAV LKR",
            ascending=False
        ).reset_index(drop=True)
        successful_df["Country Rank"] = successful_df.index + 1

    if not failed_df.empty:
        failed_df["Country Rank"] = None

    return pd.concat(
        [successful_df, failed_df],
        ignore_index=True
    )


def render_country_comparison_tab(
    country_comparison_df: pd.DataFrame | None,
    selected_country: str | None = None
) -> None:
    """
    Render country comparison tab.
    """

    st.subheader("Country Comparison")

    st.caption(
        "This compares the same selected scenario across all registered countries. "
        "Ranking is based on LKR-converted final NAV, because local currencies are not directly comparable."
    )

    if country_comparison_df is None or country_comparison_df.empty:
        st.warning("Run the simulation to generate the country comparison.")
        return

    failed_count = int((country_comparison_df["Status"] != "OK").sum()) if "Status" in country_comparison_df.columns else 0

    if failed_count > 0:
        st.warning(
            f"{failed_count} country dataset(s) failed during country comparison. "
            "Check the Error column before trusting the ranking."
        )

    ok_df = country_comparison_df[
        country_comparison_df["Status"] == "OK"
    ].copy() if "Status" in country_comparison_df.columns else country_comparison_df.copy()

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

    with metric_col_1:
        st.metric("Countries compared", len(country_comparison_df))

    with metric_col_2:
        if not ok_df.empty:
            best_country = ok_df.iloc[0]["Country"]
            st.metric("Best country", best_country)
        else:
            st.metric("Best country", "N/A")

    with metric_col_3:
        if selected_country and not ok_df.empty and selected_country in ok_df["Country"].values:
            selected_rank = ok_df.loc[
                ok_df["Country"] == selected_country,
                "Country Rank"
            ].iloc[0]
            st.metric("Selected country rank", selected_rank)
        else:
            st.metric("Selected country rank", "N/A")

    st.dataframe(
        country_comparison_df,
        use_container_width=True,
        hide_index=True
    )

    if ok_df.empty:
        st.error("No successful country comparison rows were generated.")
        return

    st.subheader("Final NAV in LKR by Country")

    nav_fig = px.bar(
        ok_df,
        x="Country",
        y="Final NAV LKR",
        text_auto=".2s",
        title="Country Comparison - Final NAV in LKR"
    )

    nav_fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Final NAV in LKR"
    )

    st.plotly_chart(
        nav_fig,
        use_container_width=True
    )

    st.subheader("Risk Score by Country")

    risk_fig = px.bar(
        ok_df.sort_values(by="Risk Score", ascending=True),
        x="Country",
        y="Risk Score",
        text_auto=".2f",
        title="Country Comparison - Risk Score"
    )

    risk_fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Risk Score"
    )

    st.plotly_chart(
        risk_fig,
        use_container_width=True
    )
