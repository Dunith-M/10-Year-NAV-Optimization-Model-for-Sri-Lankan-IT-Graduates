from typing import Any, Dict, List, Optional
import json
import os

import pandas as pd
import streamlit as st

from src.country_manager import get_available_countries, get_country_config
from src.data_loader import load_dataset
from src.scenario_builder import build_scenario_config
from src.income_model import calculate_yearly_income
from src.expense_model import calculate_yearly_expenses
from src.nav_model import calculate_nav_simulation
from src.comparison_model import (
    get_dataset_country,
    get_dataset_currency,
    get_exchange_rate_from_dataset
)


CURRENCY_CODES = {
    "AUD",
    "LKR",
    "EUR",
    "JPY",
    "USD",
    "CAD",
    "SGD",
    "NZD",
    "AED",
    "GBP",
    "INR"
}


REQUIRED_TOP_LEVEL_SECTIONS = [
    "metadata",
    "visa",
    "education",
    "income",
    "expenses",
    "investment_and_economy",
    "loans_and_debt",
    "scenario_defaults"
]


REQUIRED_METADATA_FIELDS = [
    "currency",
    "base_year",
    "time_horizon_years"
]


def _load_raw_json(dataset_path: str) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _add_test_result(
    records: List[Dict[str, Any]],
    country: str,
    test_name: str,
    status: str,
    details: str
) -> None:
    records.append(
        {
            "Country": country,
            "Test": test_name,
            "Status": status,
            "Details": details
        }
    )


def _get_nested_currency(dataset: Dict[str, Any]) -> str:
    metadata = dataset.get("metadata", {})
    return str(metadata.get("currency", dataset.get("currency", "LOCAL"))).upper()


def _find_exchange_rate_key(data: Any, path: str = "") -> Optional[str]:
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else str(key)
            key_text = str(key).lower()

            if (
                ("to_lkr" in key_text and "exchange" in key_text)
                or "lkr_exchange_rate" in key_text
            ):
                return current_path

            nested_match = _find_exchange_rate_key(value, current_path)

            if nested_match:
                return nested_match

    elif isinstance(data, list):
        for index, item in enumerate(data):
            nested_match = _find_exchange_rate_key(
                item,
                f"{path}[{index}]"
            )

            if nested_match:
                return nested_match

    return None


def _looks_like_currency_code(value: Any) -> bool:
    return str(value).strip().upper() in CURRENCY_CODES


def _detect_currency_mismatches(
    data: Any,
    expected_currency: str,
    path: str = ""
) -> List[str]:
    """
    Detect fields where a money value uses a currency different from metadata.currency.

    LKR is allowed because the app intentionally converts final results to LKR.
    Exchange-rate fields are excluded.
    """

    mismatches = []
    expected_currency = expected_currency.upper()

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else str(key)
            key_text = str(key).lower()
            path_text = current_path.lower()

            is_exchange_rate_path = (
                "exchange" in path_text
                or "to_lkr" in path_text
            )

            is_currency_field = key_text in {
                "currency",
                "currency_or_unit",
                "unit"
            }

            if (
                is_currency_field
                and not is_exchange_rate_path
                and _looks_like_currency_code(value)
            ):
                currency_value = str(value).strip().upper()

                if currency_value not in {expected_currency, "LKR"}:
                    mismatches.append(
                        f"{current_path} = {currency_value}, expected {expected_currency}"
                    )

            mismatches.extend(
                _detect_currency_mismatches(
                    value,
                    expected_currency,
                    current_path
                )
            )

    elif isinstance(data, list):
        for index, item in enumerate(data):
            mismatches.extend(
                _detect_currency_mismatches(
                    item,
                    expected_currency,
                    f"{path}[{index}]"
                )
            )

    return mismatches


def _get_first_existing_column(
    df: pd.DataFrame,
    candidates: List[str]
) -> Optional[str]:
    if df is None or getattr(df, "empty", True):
        return None

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def _build_default_test_scenario(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple default scenario that should run for every country.
    """

    return build_scenario_config(
        dataset=dataset,
        migration_path_label="Working visa path",
        life_scenario_label="Single",
        car_option_label="No car",
        investment_option_label="Save only",
        spouse_income_case_label="Moderate",
        salary_growth_rate=0.03,
        inflation_rate=0.03,
        investment_return_rate=0.05,
        rent_multiplier=1.00,
        tuition_multiplier=1.00,
        childcare_multiplier=1.00,
        education_mode_label="No further study",
        pr_timing_label="Normal PR",
        custom_pr_year=None,
        car_purchase_timing_label="No car",
        first_child_timing_label="No child",
        second_child_timing_label="No second child",
        investment_split_label="Save only"
    )


def run_multi_country_validation_tests() -> pd.DataFrame:
    """
    Test each registered dataset before trusting the app.

    Tests:
        Australia loads
        Sri Lanka loads
        Germany loads
        Japan loads
        Required fields exist
        Currency mismatch detected
        Exchange rate exists
        Simulation produces 10 rows
        No negative cash silently appears
        Final NAV exists
        LKR conversion works
    """

    records: List[Dict[str, Any]] = []

    countries = get_available_countries()

    if not countries:
        return pd.DataFrame(
            [
                {
                    "Country": "N/A",
                    "Test": "Country registry",
                    "Status": "Fail",
                    "Details": "No countries found in country_registry.json."
                }
            ]
        )

    for registry_country_name in countries:
        raw_dataset = None
        dataset = None

        try:
            country_config = get_country_config(registry_country_name)
            dataset_path = country_config["dataset_path"]

            raw_dataset = _load_raw_json(dataset_path)

            country = (
                raw_dataset.get("metadata", {}).get("country")
                or raw_dataset.get("metadata", {}).get("country_name")
                or registry_country_name
            )

            _add_test_result(
                records,
                country,
                f"{registry_country_name} raw JSON loads",
                "Pass",
                f"Loaded {dataset_path}"
            )

        except Exception as error:
            _add_test_result(
                records,
                registry_country_name,
                f"{registry_country_name} raw JSON loads",
                "Fail",
                str(error)
            )
            continue

        expected_currency = _get_nested_currency(raw_dataset)

        missing_sections = [
            section
            for section in REQUIRED_TOP_LEVEL_SECTIONS
            if section not in raw_dataset
        ]

        if missing_sections:
            _add_test_result(
                records,
                country,
                "Required sections exist",
                "Fail",
                f"Missing sections: {', '.join(missing_sections)}"
            )
        else:
            _add_test_result(
                records,
                country,
                "Required sections exist",
                "Pass",
                "All required top-level sections exist."
            )

        metadata = raw_dataset.get("metadata", {})

        missing_metadata_fields = [
            field
            for field in REQUIRED_METADATA_FIELDS
            if field not in metadata
        ]

        if missing_metadata_fields:
            _add_test_result(
                records,
                country,
                "Required metadata fields exist",
                "Fail",
                f"Missing metadata fields: {', '.join(missing_metadata_fields)}"
            )
        else:
            _add_test_result(
                records,
                country,
                "Required metadata fields exist",
                "Pass",
                "currency, base_year, and time_horizon_years exist."
            )

        mismatch_list = _detect_currency_mismatches(
            raw_dataset,
            expected_currency
        )

        if mismatch_list:
            preview = "; ".join(mismatch_list[:3])
            extra = "" if len(mismatch_list) <= 3 else f" + {len(mismatch_list) - 3} more"
            _add_test_result(
                records,
                country,
                "Currency mismatch detected",
                "Warning",
                f"{preview}{extra}"
            )
        else:
            _add_test_result(
                records,
                country,
                "Currency mismatch detected",
                "Pass",
                f"No mismatch found against metadata currency {expected_currency}."
            )

        exchange_rate_key = _find_exchange_rate_key(raw_dataset)

        if exchange_rate_key:
            _add_test_result(
                records,
                country,
                "Exchange rate exists",
                "Pass",
                f"Found exchange-rate field: {exchange_rate_key}"
            )
        else:
            _add_test_result(
                records,
                country,
                "Exchange rate exists",
                "Fail",
                "No local-currency-to-LKR exchange-rate field found."
            )

        try:
            dataset = load_dataset(dataset_path)
            _add_test_result(
                records,
                country,
                f"{registry_country_name} validated dataset loads",
                "Pass",
                "load_dataset completed successfully."
            )

        except Exception as error:
            _add_test_result(
                records,
                country,
                f"{registry_country_name} validated dataset loads",
                "Fail",
                str(error)
            )
            continue

        try:
            scenario_config = _build_default_test_scenario(dataset)

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

            if len(nav_df) == 10:
                _add_test_result(
                    records,
                    country,
                    "Simulation produces 10 rows",
                    "Pass",
                    "NAV simulation produced exactly 10 rows."
                )
            else:
                _add_test_result(
                    records,
                    country,
                    "Simulation produces 10 rows",
                    "Fail",
                    f"NAV simulation produced {len(nav_df)} rows."
                )

            cash_column = _get_first_existing_column(
                nav_df,
                [
                    "Cash Balance",
                    "Cash Savings",
                    "Cash"
                ]
            )

            if cash_column is None:
                _add_test_result(
                    records,
                    country,
                    "No negative cash silently appears",
                    "Warning",
                    "No cash column found to test."
                )
            else:
                min_cash = float(
                    pd.to_numeric(
                        nav_df[cash_column],
                        errors="coerce"
                    ).fillna(0.0).min()
                )

                if min_cash < -0.01:
                    _add_test_result(
                        records,
                        country,
                        "No negative cash silently appears",
                        "Fail",
                        f"{cash_column} becomes negative: {min_cash:,.2f}"
                    )
                else:
                    _add_test_result(
                        records,
                        country,
                        "No negative cash silently appears",
                        "Pass",
                        f"{cash_column} never goes below zero."
                    )

            nav_column = _get_first_existing_column(
                nav_df,
                [
                    "Local Currency NAV",
                    "NAV",
                    "Net Asset Value"
                ]
            )

            if nav_column is None:
                _add_test_result(
                    records,
                    country,
                    "Final NAV exists",
                    "Fail",
                    "No NAV column found."
                )
                final_nav = None
            else:
                final_nav = float(
                    pd.to_numeric(
                        nav_df[nav_column],
                        errors="coerce"
                    ).iloc[-1]
                )

                _add_test_result(
                    records,
                    country,
                    "Final NAV exists",
                    "Pass",
                    f"Final NAV = {final_nav:,.2f} {get_dataset_currency(dataset)}"
                )

            exchange_rate = get_exchange_rate_from_dataset(dataset)

            if final_nav is None:
                _add_test_result(
                    records,
                    country,
                    "LKR conversion works",
                    "Fail",
                    "Final NAV missing, so LKR conversion cannot be tested."
                )
            elif exchange_rate <= 0:
                _add_test_result(
                    records,
                    country,
                    "LKR conversion works",
                    "Fail",
                    f"Invalid exchange rate: {exchange_rate}"
                )
            else:
                final_nav_lkr = final_nav * exchange_rate
                _add_test_result(
                    records,
                    country,
                    "LKR conversion works",
                    "Pass",
                    f"Final NAV LKR = {final_nav_lkr:,.2f}"
                )

        except Exception as error:
            _add_test_result(
                records,
                country,
                "Default simulation run",
                "Fail",
                str(error)
            )

    return pd.DataFrame(records)


def render_multi_country_testing_tab(testing_df: pd.DataFrame | None) -> None:
    """
    Render multi-country dataset testing table.
    """

    st.subheader("Multi-Country Dataset Testing")

    st.caption(
        "This tests every registered country dataset before you trust the app output."
    )

    if testing_df is None or testing_df.empty:
        st.warning("No multi-country testing results available.")
        return

    pass_count = int((testing_df["Status"] == "Pass").sum())
    warning_count = int((testing_df["Status"] == "Warning").sum())
    fail_count = int((testing_df["Status"] == "Fail").sum())

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    with metric_col_1:
        st.metric("Total checks", len(testing_df))

    with metric_col_2:
        st.metric("Passed", pass_count)

    with metric_col_3:
        st.metric("Warnings", warning_count)

    with metric_col_4:
        st.metric("Failed", fail_count)

    if fail_count > 0:
        st.error(
            "Some multi-country tests failed. Fix these before using the app for final demo."
        )
    elif warning_count > 0:
        st.warning(
            "No hard failures, but warnings exist. Currency mismatches or weak data should be fixed before final submission."
        )
    else:
        st.success("All multi-country tests passed.")

    st.dataframe(
        testing_df,
        use_container_width=True,
        hide_index=True
    )
