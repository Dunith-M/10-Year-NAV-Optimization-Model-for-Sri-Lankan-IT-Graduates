import json

import pandas as pd
import streamlit as st

from src.currency_utils import (
    LKR_PRESENT_VALUE_DISCOUNT_RATE,
    calculate_lkr_present_value,
    format_currency_pair,
    format_lkr,
    format_local_currency,
    get_country_currency,
    get_exchange_rate_key,
    get_exchange_rate_to_lkr
)

from src.scenario_builder import (
    MIGRATION_PATH_OPTIONS,
    LIFE_SCENARIO_OPTIONS,
    build_scenario_config
)

from src.income_model import calculate_yearly_income
from src.expense_model import calculate_yearly_expenses
from src.nav_model import calculate_nav_simulation, get_nav_summary


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def get_exchange_rate(dataset) -> float:
    return get_exchange_rate_to_lkr(dataset)


def to_key_value_dataframe(data: dict, key_name: str, value_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        list(data.items()),
        columns=[key_name, value_name]
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dictionary_to_json_bytes(data: dict) -> bytes:
    return json.dumps(data, indent=4).encode("utf-8")


def render_metric_grid(metrics: list[dict], columns_per_row: int = 3) -> None:
    for start_index in range(0, len(metrics), columns_per_row):
        row_metrics = metrics[start_index:start_index + columns_per_row]
        columns = st.columns(len(row_metrics), gap="medium")

        for column, metric in zip(columns, row_metrics):
            column.metric(
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta")
            )


def create_dataset_key_values_dataframe(summary: dict) -> pd.DataFrame:
    currency = summary["currency"]
    exchange_rate_key = summary["exchange_rate_key"]

    key_values = {
        "Dataset Name": summary["dataset_name"],
        "Country": summary["country"],
        "Currency": currency,
        "Base Year": summary["base_year"],
        "Time Horizon": summary["time_horizon_years"],

        "Student Visa Fee": format_local_currency(
            summary["student_visa_fee"],
            currency=currency
        ),
        "Graduate Visa Fee": format_local_currency(
            summary["graduate_visa_fee"],
            currency=currency
        ),
        "PR Application Fee": format_local_currency(
            summary["pr_application_fee"],
            currency=currency
        ),
        "Annual Tuition Fee": format_local_currency(
            summary["annual_tuition_fee"],
            currency=currency
        ),
        "Graduate Salary": format_local_currency(
            summary["graduate_salary"],
            currency=currency
        ),
        "Mid-Level Salary": format_local_currency(
            summary["mid_level_salary"],
            currency=currency
        ),
        "Senior Salary": format_local_currency(
            summary["senior_salary"],
            currency=currency
        ),
        "Single Monthly Rent": format_local_currency(
            summary["single_monthly_rent"],
            currency=currency
        ),
        "Family Monthly Rent": format_local_currency(
            summary["family_monthly_rent"],
            currency=currency
        ),
        "Inflation Rate": format_percentage(summary["inflation_rate"]),
        "Investment Return Rate": format_percentage(summary["investment_return_rate"]),
        f"{currency} to LKR Exchange Rate": summary["exchange_rate_to_lkr"],
        "Exchange Rate Field": exchange_rate_key
    }

    return to_key_value_dataframe(
        data=key_values,
        key_name="Field",
        value_name="Value"
    )


def build_scenario_comparison(
    dataset,
    car_option_label,
    investment_option_label,
    spouse_income_case_label,
    salary_growth_rate,
    inflation_rate,
    investment_return_rate,
    rent_multiplier,
    tuition_multiplier,
    childcare_multiplier
) -> pd.DataFrame:

    comparison_records = []

    local_currency = get_country_currency(dataset)
    exchange_rate = get_exchange_rate(dataset)
    exchange_rate_key = get_exchange_rate_key(dataset)

    for migration_path_label in MIGRATION_PATH_OPTIONS.keys():
        for life_scenario_label in LIFE_SCENARIO_OPTIONS.keys():

            temp_scenario_config = build_scenario_config(
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
                childcare_multiplier=childcare_multiplier
            )

            temp_income_df = calculate_yearly_income(
                dataset=dataset,
                scenario_config=temp_scenario_config
            )

            temp_expense_df = calculate_yearly_expenses(
                dataset=dataset,
                scenario_config=temp_scenario_config
            )

            temp_nav_df = calculate_nav_simulation(
                dataset=dataset,
                scenario_config=temp_scenario_config,
                income_df=temp_income_df,
                expense_df=temp_expense_df
            )

            temp_nav_summary = get_nav_summary(temp_nav_df)

            nav_local = temp_nav_summary["year_10_nav"]
            nav_lkr = temp_nav_summary.get(
                "year_10_nav_lkr",
                nav_local * exchange_rate
            )
            nav_present_value_lkr = temp_nav_summary.get(
                "year_10_nav_present_value_lkr",
                calculate_lkr_present_value(nav_lkr)
            )

            assets_local = temp_nav_summary["year_10_total_assets"]
            liabilities_local = temp_nav_summary["year_10_total_liabilities"]
            debt_local = temp_nav_summary["year_10_total_debt"]

            comparison_records.append(
                {
                    "Scenario": f"{migration_path_label} | {life_scenario_label}",
                    "Migration Path": migration_path_label,
                    "Life Scenario": life_scenario_label,

                    "Currency": local_currency,
                    "Exchange Rate Field": exchange_rate_key,
                    "Exchange Rate to LKR": exchange_rate,

                    "Year-10 NAV Local": nav_local,
                    f"Year-10 NAV {local_currency}": nav_local,
                    "Year-10 NAV LKR": nav_lkr,
                    "Year-10 NAV Present Value LKR": nav_present_value_lkr,
                    "Present Value Discount Rate": LKR_PRESENT_VALUE_DISCOUNT_RATE,
                    "Present Value Discount Years": 10,

                    "Year-10 Assets Local": assets_local,
                    f"Year-10 Assets {local_currency}": assets_local,
                    "Year-10 Assets LKR": assets_local * exchange_rate,

                    "Year-10 Liabilities Local": liabilities_local,
                    f"Year-10 Liabilities {local_currency}": liabilities_local,
                    "Year-10 Liabilities LKR": liabilities_local * exchange_rate,

                    "Year-10 Debt Local": debt_local,
                    f"Year-10 Debt {local_currency}": debt_local,
                    "Year-10 Debt LKR": debt_local * exchange_rate,

                    "Investment Balance Local": temp_nav_summary["year_10_investment_balance"],
                    "Superannuation Balance Local": temp_nav_summary["year_10_superannuation_balance"],

                    "Formatted NAV": format_currency_pair(nav_local, dataset)
                }
            )

    comparison_df = pd.DataFrame(comparison_records)

    comparison_df = comparison_df.sort_values(
        by="Year-10 NAV Local",
        ascending=False
    ).reset_index(drop=True)

    return comparison_df
