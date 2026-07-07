from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from src.currency_utils import format_local_currency, format_lkr


def _normalize_text(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
    )


def _find_column(
    df: Optional[pd.DataFrame],
    candidate_names: List[str]
) -> Optional[str]:
    if df is None or df.empty:
        return None

    normalized_candidates = {
        _normalize_text(candidate): candidate
        for candidate in candidate_names
    }

    for column in df.columns:
        normalized_column = _normalize_text(column)

        if normalized_column in normalized_candidates:
            return column

    return None


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []

    return list(df.select_dtypes(include="number").columns)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default

        return float(value)
    except Exception:
        return default


def _safe_sum_column(
    df: Optional[pd.DataFrame],
    candidate_names: List[str]
) -> float:
    column = _find_column(df, candidate_names)

    if column is None:
        return 0.0

    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _safe_sum_matching_columns(
    df: Optional[pd.DataFrame],
    include_terms: List[str],
    exclude_terms: Optional[List[str]] = None
) -> float:
    if df is None or df.empty:
        return 0.0

    exclude_terms = exclude_terms or []

    total = 0.0

    for column in df.columns:
        normalized_column = _normalize_text(column)

        has_include_term = any(
            _normalize_text(term) in normalized_column
            for term in include_terms
        )

        has_exclude_term = any(
            _normalize_text(term) in normalized_column
            for term in exclude_terms
        )

        if has_include_term and not has_exclude_term:
            if pd.api.types.is_numeric_dtype(df[column]):
                total += float(
                    pd.to_numeric(df[column], errors="coerce").fillna(0).sum()
                )

    return total


def _calculate_category_total(
    df: Optional[pd.DataFrame],
    exact_candidates: List[str],
    fallback_terms: List[str],
    exclude_terms: Optional[List[str]] = None
) -> float:
    exact_total = _safe_sum_column(df, exact_candidates)

    if exact_total > 0:
        return exact_total

    return _safe_sum_matching_columns(
        df=df,
        include_terms=fallback_terms,
        exclude_terms=exclude_terms
    )


def _format_local(value: Any, currency: str = "LOCAL") -> str:
    return format_local_currency(value=value, currency=currency)


def _format_lkr(value: Any) -> str:
    return format_lkr(value)


def _format_year(value: Any) -> str:
    if value is None:
        return "Not reached"

    try:
        return f"Year {int(value)}"
    except Exception:
        return "Not reached"


def _clean_variable_name(value: Any) -> str:
    if value is None:
        return "Unavailable"

    raw_value = str(value).strip()

    variable_map = {
        "salary_growth_rate": "Salary Growth",
        "salary growth rate": "Salary Growth",
        "salary growth": "Salary Growth",
        "rent_multiplier": "Rent",
        "rent multiplier": "Rent",
        "rent": "Rent",
        "inflation_rate": "Inflation",
        "inflation rate": "Inflation",
        "inflation": "Inflation",
        "investment_return_rate": "Investment Return",
        "investment return rate": "Investment Return",
        "investment return": "Investment Return",
        "tuition_multiplier": "Tuition",
        "tuition multiplier": "Tuition",
        "tuition": "Tuition",
        "childcare_multiplier": "Childcare",
        "childcare multiplier": "Childcare",
        "childcare": "Childcare",
        "spouse_income": "Spouse Income",
        "spouse income": "Spouse Income",
        "exchange_rate": "Exchange Rate",
        "exchange rate": "Exchange Rate"
    }

    normalized_raw = raw_value.lower().replace("_", " ").strip()

    if normalized_raw in variable_map:
        return variable_map[normalized_raw]

    return raw_value.replace("_", " ").title()


def _get_nav_column(nav_df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        nav_df,
        [
            "Local Currency NAV",
            "NAV",
            "Final NAV",
            "Year-10 NAV Local",
            "Year 10 NAV Local",
            "Year-10 NAV"
        ]
    )


def _get_debt_column(nav_df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        nav_df,
        [
            "Local Currency Debt",
            "Total Debt",
            "Debt",
            "Total Liabilities",
            "Local Currency Liabilities",
            "Negative Cash Debt"
        ]
    )


def _get_break_even_year(nav_df: pd.DataFrame) -> Optional[int]:
    if nav_df is None or nav_df.empty:
        return None

    nav_column = _get_nav_column(nav_df)

    if nav_column is None:
        return None

    break_even_rows = nav_df[
        pd.to_numeric(nav_df[nav_column], errors="coerce").fillna(0) >= 0
    ]

    if len(break_even_rows) == 0:
        return None

    return int(break_even_rows.iloc[0]["Year"])


def _get_highest_debt(nav_df: pd.DataFrame) -> Tuple[float, Optional[int]]:
    if nav_df is None or nav_df.empty:
        return 0.0, None

    debt_column = _get_debt_column(nav_df)

    if debt_column is None:
        return 0.0, None

    highest_debt_index = pd.to_numeric(
        nav_df[debt_column],
        errors="coerce"
    ).fillna(0).idxmax()

    highest_debt_row = nav_df.loc[highest_debt_index]

    return (
        float(highest_debt_row[debt_column]),
        int(highest_debt_row["Year"])
    )


def _get_lowest_nav(nav_df: pd.DataFrame) -> Tuple[float, Optional[int]]:
    if nav_df is None or nav_df.empty:
        return 0.0, None

    nav_column = _get_nav_column(nav_df)

    if nav_column is None:
        return 0.0, None

    lowest_nav_index = pd.to_numeric(
        nav_df[nav_column],
        errors="coerce"
    ).fillna(0).idxmin()

    lowest_nav_row = nav_df.loc[lowest_nav_index]

    return (
        float(lowest_nav_row[nav_column]),
        int(lowest_nav_row["Year"])
    )


def _get_best_scenario(
    comparison_df: Optional[pd.DataFrame],
    selected_final_nav: float
) -> Tuple[str, float, float, float]:
    if comparison_df is None or comparison_df.empty:
        return "Unavailable", selected_final_nav, 0.0, 0.0

    scenario_column = _find_column(
        comparison_df,
        [
            "Scenario",
            "Scenario Name",
            "Name",
            "Case",
            "Configuration",
            "Scenario Type"
        ]
    )

    final_nav_column = _find_column(
        comparison_df,
        [
            "Year-10 NAV Local",
            "Year 10 NAV Local",
            "Final NAV Local",
            "Local Currency NAV",
            "Final NAV",
            "Year 10 NAV",
            "Year-10 NAV",
            "NAV",
            "Year 10 NAV AUD",
            "Year 10 NAV (AUD)"
        ]
    )

    if final_nav_column is None:
        numeric_cols = _numeric_columns(comparison_df)

        if len(numeric_cols) == 0:
            return "Unavailable", selected_final_nav, 0.0, 0.0

        final_nav_column = numeric_cols[-1]

    comparison_copy = comparison_df.copy()
    comparison_copy[final_nav_column] = pd.to_numeric(
        comparison_copy[final_nav_column],
        errors="coerce"
    )

    comparison_copy = comparison_copy.dropna(subset=[final_nav_column])

    if comparison_copy.empty:
        return "Unavailable", selected_final_nav, 0.0, 0.0

    best_row = comparison_copy.loc[comparison_copy[final_nav_column].idxmax()]

    if scenario_column is not None:
        best_scenario_name = str(best_row[scenario_column])
    else:
        best_scenario_name = "Best Available Scenario"

    best_final_nav = float(best_row[final_nav_column])

    best_scenario_gap = best_final_nav - selected_final_nav
    selected_vs_best_difference = selected_final_nav - best_final_nav

    return (
        best_scenario_name,
        best_final_nav,
        best_scenario_gap,
        selected_vs_best_difference
    )


def _get_main_risk_variable_from_tornado(
    tornado_df: Optional[pd.DataFrame]
) -> str:
    if tornado_df is None or tornado_df.empty:
        return "Unavailable"

    variable_column = _find_column(
        tornado_df,
        [
            "Variable",
            "Risk Variable",
            "Input",
            "Parameter",
            "Factor",
            "Assumption"
        ]
    )

    if variable_column is None:
        object_columns = list(tornado_df.select_dtypes(include="object").columns)

        if len(object_columns) == 0:
            return "Unavailable"

        variable_column = object_columns[0]

    impact_column = _find_column(
        tornado_df,
        [
            "Max Impact %",
            "Max Impact LKR",
            "Max Impact Local",
            "Impact",
            "NAV Impact",
            "Absolute Impact",
            "Range",
            "NAV Range",
            "Difference",
            "Final NAV Difference"
        ]
    )

    tornado_copy = tornado_df.copy()

    if impact_column is not None:
        tornado_copy[impact_column] = pd.to_numeric(
            tornado_copy[impact_column],
            errors="coerce"
        ).abs()

        tornado_copy = tornado_copy.dropna(subset=[impact_column])

        if tornado_copy.empty:
            return "Unavailable"

        main_row = tornado_copy.loc[tornado_copy[impact_column].idxmax()]
        return _clean_variable_name(main_row[variable_column])

    numeric_cols = _numeric_columns(tornado_copy)

    if len(numeric_cols) == 0:
        return "Unavailable"

    tornado_copy["_risk_range"] = (
        tornado_copy[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .max(axis=1)
        -
        tornado_copy[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .min(axis=1)
    ).abs()

    tornado_copy = tornado_copy.dropna(subset=["_risk_range"])

    if tornado_copy.empty:
        return "Unavailable"

    main_row = tornado_copy.loc[tornado_copy["_risk_range"].idxmax()]
    return _clean_variable_name(main_row[variable_column])


def _get_main_risk_variable_from_sensitivity(
    sensitivity_df: Optional[pd.DataFrame]
) -> str:
    if sensitivity_df is None or sensitivity_df.empty:
        return "Unavailable"

    variable_column = _find_column(
        sensitivity_df,
        [
            "Variable",
            "Risk Variable",
            "Input",
            "Parameter",
            "Factor",
            "Assumption"
        ]
    )

    final_nav_column = _find_column(
        sensitivity_df,
        [
            "Year-10 NAV Local",
            "Year 10 NAV Local",
            "Final NAV Local",
            "Local Currency NAV",
            "Final NAV",
            "Year 10 NAV",
            "Year-10 NAV",
            "NAV",
            "Year 10 NAV AUD",
            "Year 10 NAV (AUD)"
        ]
    )

    if variable_column is None or final_nav_column is None:
        return "Unavailable"

    sensitivity_copy = sensitivity_df.copy()
    sensitivity_copy[final_nav_column] = pd.to_numeric(
        sensitivity_copy[final_nav_column],
        errors="coerce"
    )

    sensitivity_copy = sensitivity_copy.dropna(subset=[final_nav_column])

    if sensitivity_copy.empty:
        return "Unavailable"

    risk_ranges = (
        sensitivity_copy
        .groupby(variable_column)[final_nav_column]
        .agg(lambda values: values.max() - values.min())
        .abs()
    )

    if risk_ranges.empty:
        return "Unavailable"

    main_variable = risk_ranges.idxmax()
    return _clean_variable_name(main_variable)


def _get_main_risk_variable(
    sensitivity_df: Optional[pd.DataFrame],
    tornado_df: Optional[pd.DataFrame]
) -> str:
    tornado_risk = _get_main_risk_variable_from_tornado(tornado_df)

    if tornado_risk != "Unavailable":
        return tornado_risk

    return _get_main_risk_variable_from_sensitivity(sensitivity_df)


def _get_main_expense_category(category_totals: Dict[str, float]) -> Tuple[str, float]:
    positive_totals = {
        category: amount
        for category, amount in category_totals.items()
        if amount > 0
    }

    if len(positive_totals) == 0:
        return "Unavailable", 0.0

    main_category = max(positive_totals, key=positive_totals.get)
    return main_category, float(positive_totals[main_category])


def build_decision_sentence(decision_summary: Dict[str, Any]) -> str:
    break_even_year = decision_summary.get("break_even_year")
    main_expense_category = decision_summary.get("main_expense_category", "Unavailable")
    main_risk_variable = decision_summary.get("main_risk_variable", "Unavailable")

    if break_even_year is None:
        break_even_text = "This scenario does not become positive within the 10-year period."
    else:
        break_even_text = f"This scenario becomes positive in Year {break_even_year}."

    return (
        f"{break_even_text} "
        f"The biggest cost is {main_expense_category}. "
        f"The biggest risk is {main_risk_variable}."
    )


def build_decision_summary(
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame],
    sensitivity_df: Optional[pd.DataFrame],
    tornado_df: Optional[pd.DataFrame],
    exchange_rate: float,
    local_currency: str = "LOCAL"
) -> Dict[str, Any]:
    final_row = nav_df.iloc[-1]

    nav_column = "Local Currency NAV" if "Local Currency NAV" in nav_df.columns else "NAV"

    final_year_10_nav = float(final_row[nav_column])

    if "LKR NAV" in final_row.index:
        final_nav_lkr = float(final_row["LKR NAV"])
    else:
        final_nav_lkr = final_year_10_nav * float(exchange_rate)

    break_even_year = _get_break_even_year(nav_df)

    highest_debt_amount, highest_debt_year = _get_highest_debt(nav_df)
    lowest_nav_amount, lowest_nav_year = _get_lowest_nav(nav_df)

    total_rent_paid = _calculate_category_total(
        df=expense_df,
        exact_candidates=[
            "Rent",
            "Housing Rent",
            "Annual Rent",
            "Accommodation",
            "Accommodation Cost"
        ],
        fallback_terms=[
            "rent",
            "accommodation",
            "housing"
        ],
        exclude_terms=[
            "rate",
            "multiplier",
            "percentage"
        ]
    )

    total_tuition_paid = _calculate_category_total(
        df=expense_df,
        exact_candidates=[
            "Tuition",
            "Tuition Fee",
            "Tuition Fees",
            "Education Cost",
            "Study Cost"
        ],
        fallback_terms=[
            "tuition",
            "education",
            "study"
        ],
        exclude_terms=[
            "rate",
            "multiplier",
            "percentage"
        ]
    )

    total_childcare_paid = _calculate_category_total(
        df=expense_df,
        exact_candidates=[
            "Childcare",
            "Childcare Cost",
            "Child Care",
            "Child Care Cost"
        ],
        fallback_terms=[
            "childcare",
            "child care"
        ],
        exclude_terms=[
            "rate",
            "multiplier",
            "percentage"
        ]
    )

    total_car_cost = _calculate_category_total(
        df=expense_df,
        exact_candidates=[
            "Car Cost",
            "Total Car Cost",
            "Car Expenses",
            "Vehicle Cost",
            "Vehicle Expenses",
            "Car Loan Payment"
        ],
        fallback_terms=[
            "car",
            "vehicle",
            "fuel",
            "maintenance",
            "insurance",
            "registration"
        ],
        exclude_terms=[
            "car value",
            "vehicle value",
            "resale value",
            "asset value",
            "rate",
            "multiplier",
            "percentage"
        ]
    )

    total_tax_paid = _calculate_category_total(
        df=income_df,
        exact_candidates=[
            "Tax",
            "Tax Paid",
            "Income Tax",
            "Total Tax"
        ],
        fallback_terms=[
            "tax"
        ],
        exclude_terms=[
            "rate",
            "percentage"
        ]
    )

    total_superannuation = _calculate_category_total(
        df=income_df,
        exact_candidates=[
            "Superannuation",
            "Super",
            "Employer Superannuation",
            "Retirement Contribution"
        ],
        fallback_terms=[
            "superannuation",
            "super",
            "retirement"
        ],
        exclude_terms=[
            "rate",
            "percentage"
        ]
    )

    category_totals = {
        "Rent": total_rent_paid,
        "Tuition": total_tuition_paid,
        "Childcare": total_childcare_paid,
        "Car Cost": total_car_cost,
        "Tax": total_tax_paid
    }

    main_expense_category, main_expense_amount = _get_main_expense_category(
        category_totals=category_totals
    )

    main_risk_variable = _get_main_risk_variable(
        sensitivity_df=sensitivity_df,
        tornado_df=tornado_df
    )

    (
        best_scenario_name,
        best_scenario_final_nav,
        best_scenario_gap,
        selected_vs_best_difference
    ) = _get_best_scenario(
        comparison_df=comparison_df,
        selected_final_nav=final_year_10_nav
    )

    decision_summary = {
        "local_currency": local_currency,
        "exchange_rate_to_lkr": exchange_rate,

        "final_year_10_nav_local": final_year_10_nav,
        "final_year_10_nav_lkr": final_nav_lkr,

        # Backward-compatible keys.
        "final_year_10_nav": final_year_10_nav,
        "final_nav_lkr": final_nav_lkr,

        "break_even_year": break_even_year,

        "highest_debt_year": highest_debt_year,
        "highest_debt_amount": highest_debt_amount,

        "lowest_nav_year": lowest_nav_year,
        "lowest_nav_amount": lowest_nav_amount,

        "total_rent_paid": total_rent_paid,
        "total_tuition_paid": total_tuition_paid,
        "total_childcare_paid": total_childcare_paid,
        "total_car_cost": total_car_cost,
        "total_tax_paid": total_tax_paid,
        "total_superannuation": total_superannuation,

        "expense_category_totals": category_totals,
        "main_expense_category": main_expense_category,
        "main_expense_amount": main_expense_amount,

        "main_risk_variable": main_risk_variable,

        "best_scenario_name": best_scenario_name,
        "best_scenario_final_nav": best_scenario_final_nav,
        "best_scenario_gap": best_scenario_gap,
        "selected_vs_best_difference": selected_vs_best_difference
    }

    decision_summary["decision_sentence"] = build_decision_sentence(decision_summary)

    return decision_summary


def render_decision_summary_dashboard(
    decision_summary: Dict[str, Any]
) -> None:
    st.markdown("## Decision Summary")

    local_currency = decision_summary.get("local_currency", "LOCAL")

    final_nav = decision_summary["final_year_10_nav"]
    final_nav_lkr = decision_summary["final_nav_lkr"]
    break_even_year = decision_summary["break_even_year"]

    highest_debt_amount = decision_summary["highest_debt_amount"]
    highest_debt_year = decision_summary["highest_debt_year"]

    main_expense_category = decision_summary["main_expense_category"]
    main_expense_amount = decision_summary["main_expense_amount"]

    best_scenario_gap = decision_summary["best_scenario_gap"]
    best_scenario_name = decision_summary["best_scenario_name"]

    main_risk_variable = decision_summary["main_risk_variable"]

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

    with metric_col_1:
        st.metric(
            label=f"Final Year-10 NAV ({local_currency})",
            value=_format_local(final_nav, local_currency),
            delta=_format_lkr(final_nav_lkr)
        )

    with metric_col_2:
        st.metric(
            label="Break-even Year",
            value=_format_year(break_even_year)
        )

    with metric_col_3:
        st.metric(
            label=f"Peak Debt ({local_currency})",
            value=_format_local(highest_debt_amount, local_currency),
            delta=_format_year(highest_debt_year)
        )

    metric_col_4, metric_col_5, metric_col_6 = st.columns(3)

    with metric_col_4:
        st.metric(
            label="Biggest Expense",
            value=main_expense_category,
            delta=_format_local(main_expense_amount, local_currency)
        )

    with metric_col_5:
        st.metric(
            label=f"Best Scenario Gap ({local_currency})",
            value=_format_local(best_scenario_gap, local_currency),
            delta=best_scenario_name
        )

    with metric_col_6:
        st.metric(
            label="Main Risk Variable",
            value=main_risk_variable
        )

    if final_nav > 0 and break_even_year is not None:
        st.success(decision_summary["decision_sentence"])
    elif final_nav > 0:
        st.info(decision_summary["decision_sentence"])
    else:
        st.warning(decision_summary["decision_sentence"])

    with st.expander("View detailed decision summary"):
        detailed_summary_df = pd.DataFrame(
            [
                {
                    "Metric": f"Final Year-10 NAV ({local_currency})",
                    "Value": _format_local(
                        decision_summary["final_year_10_nav"],
                        local_currency
                    )
                },
                {
                    "Metric": "Final NAV in LKR",
                    "Value": _format_lkr(decision_summary["final_nav_lkr"])
                },
                {
                    "Metric": "Break-even Year",
                    "Value": _format_year(decision_summary["break_even_year"])
                },
                {
                    "Metric": "Highest Debt Year",
                    "Value": _format_year(decision_summary["highest_debt_year"])
                },
                {
                    "Metric": f"Highest Debt Amount ({local_currency})",
                    "Value": _format_local(
                        decision_summary["highest_debt_amount"],
                        local_currency
                    )
                },
                {
                    "Metric": "Lowest NAV Year",
                    "Value": _format_year(decision_summary["lowest_nav_year"])
                },
                {
                    "Metric": f"Lowest NAV Amount ({local_currency})",
                    "Value": _format_local(
                        decision_summary["lowest_nav_amount"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Total Rent Paid ({local_currency})",
                    "Value": _format_local(
                        decision_summary["total_rent_paid"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Total Tuition Paid ({local_currency})",
                    "Value": _format_local(
                        decision_summary["total_tuition_paid"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Total Childcare Paid ({local_currency})",
                    "Value": _format_local(
                        decision_summary["total_childcare_paid"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Total Car Cost ({local_currency})",
                    "Value": _format_local(
                        decision_summary["total_car_cost"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Total Tax Paid ({local_currency})",
                    "Value": _format_local(
                        decision_summary["total_tax_paid"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Total Superannuation ({local_currency})",
                    "Value": _format_local(
                        decision_summary["total_superannuation"],
                        local_currency
                    )
                },
                {
                    "Metric": "Main Expense Category",
                    "Value": decision_summary["main_expense_category"]
                },
                {
                    "Metric": "Main Risk Variable",
                    "Value": decision_summary["main_risk_variable"]
                },
                {
                    "Metric": "Best Scenario",
                    "Value": decision_summary["best_scenario_name"]
                },
                {
                    "Metric": f"Best Scenario Final NAV ({local_currency})",
                    "Value": _format_local(
                        decision_summary["best_scenario_final_nav"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Difference Between Selected and Best Scenario ({local_currency})",
                    "Value": _format_local(
                        decision_summary["selected_vs_best_difference"],
                        local_currency
                    )
                },
                {
                    "Metric": f"Gap to Best Scenario ({local_currency})",
                    "Value": _format_local(
                        decision_summary["best_scenario_gap"],
                        local_currency
                    )
                }
            ]
        )

        st.dataframe(
            detailed_summary_df,
            use_container_width=True,
            hide_index=True
        )