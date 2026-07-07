"""
Advanced Plotly chart builder for the Multi-Country 10-Year NAV Simulator.

This module contains chart-building logic only.
app.py should call these functions and display the returned Plotly figures.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _normalise_name(value: str) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    normalised_lookup = {
        _normalise_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        match = normalised_lookup.get(_normalise_name(candidate))

        if match is not None:
            return match

    return None


def _year_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_column(
        df,
        [
            "Year",
            "Simulation Year",
            "Model Year"
        ]
    )


def _available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    columns = []

    for candidate in candidates:
        column = _resolve_column(df, [candidate])

        if column is not None and column not in columns:
            columns.append(column)

    return columns


def _copy_numeric_df(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = df.copy()

    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce"
            ).fillna(0.0)

    return output


def _country_currency_suffix(
    country_name: str | None = None,
    local_currency: str | None = None
) -> str:
    country = str(country_name or "").strip()
    currency = str(local_currency or "LOCAL").upper().strip()

    if country:
        return f"{country} {currency}"

    return currency


def _with_country_title(
    base_title: str,
    country_name: str | None = None,
    local_currency: str | None = None
) -> str:
    return f"{base_title} - {_country_currency_suffix(country_name, local_currency)}"


def _apply_common_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: str = "LOCAL"
) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=20, r=20, t=70, b=20),
        height=430
    )

    return fig


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14)
    )

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=70, b=20)
    )

    return fig


def _melt_for_chart(
    df: pd.DataFrame,
    year_col: str,
    value_columns: List[str]
) -> pd.DataFrame:
    safe_df = _copy_numeric_df(df, value_columns)

    return safe_df.melt(
        id_vars=[year_col],
        value_vars=value_columns,
        var_name="Category",
        value_name="Amount"
    )


def build_income_breakdown_stacked_bar(
    income_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Income Breakdown by Year",
        country_name,
        local_currency
    )

    year_col = _year_column(income_df)

    income_columns = _available_columns(
        income_df,
        [
            "Student Income",
            "Part-Time Income",
            "Full-Time Income",
            "Full Time Income",
            "Spouse Income"
        ]
    )

    if year_col is None or not income_columns:
        return _empty_figure(
            title,
            "Income breakdown needs Year plus student/full-time/spouse income columns."
        )

    chart_df = _melt_for_chart(
        income_df,
        year_col,
        income_columns
    )

    fig = px.bar(
        chart_df,
        x=year_col,
        y="Amount",
        color="Category",
        barmode="stack"
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_gross_vs_net_income_line_chart(
    income_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Gross Income vs Net Income",
        country_name,
        local_currency
    )

    year_col = _year_column(income_df)

    value_columns = _available_columns(
        income_df,
        [
            "Gross Income",
            "Net Income"
        ]
    )

    if year_col is None or len(value_columns) < 2:
        return _empty_figure(
            title,
            "Gross vs net chart needs Year, Gross Income, and Net Income columns."
        )

    chart_df = _melt_for_chart(
        income_df,
        year_col,
        value_columns
    )

    fig = px.line(
        chart_df,
        x=year_col,
        y="Amount",
        color="Category",
        markers=True
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_tax_over_time_chart(
    income_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Tax Paid Over Time",
        country_name,
        local_currency
    )

    year_col = _year_column(income_df)
    tax_col = _resolve_column(
        income_df,
        [
            "Tax",
            "Income Tax",
            "Tax Paid"
        ]
    )

    if year_col is None or tax_col is None:
        return _empty_figure(
            title,
            "Tax chart needs Year and Tax columns."
        )

    chart_df = _copy_numeric_df(
        income_df,
        [tax_col]
    )

    fig = px.bar(
        chart_df,
        x=year_col,
        y=tax_col
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_spouse_income_contribution_chart(
    income_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Spouse Income Contribution",
        country_name,
        local_currency
    )

    year_col = _year_column(income_df)

    spouse_col = _resolve_column(
        income_df,
        [
            "Spouse Income"
        ]
    )

    gross_col = _resolve_column(
        income_df,
        [
            "Gross Income",
            "Total Gross Income"
        ]
    )

    if year_col is None or spouse_col is None or gross_col is None:
        return _empty_figure(
            title,
            "Spouse contribution chart needs Year, Spouse Income, and Gross Income columns."
        )

    chart_df = _copy_numeric_df(
        income_df,
        [
            spouse_col,
            gross_col
        ]
    )

    chart_df["Spouse Contribution %"] = chart_df.apply(
        lambda row: (row[spouse_col] / row[gross_col] * 100)
        if row[gross_col] > 0
        else 0,
        axis=1
    )

    fig = px.line(
        chart_df,
        x=year_col,
        y="Spouse Contribution %",
        markers=True
    )

    return _apply_common_layout(
        fig,
        title,
        yaxis_title="% of Gross Income"
    )


def build_career_stage_timeline(
    income_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Career Stage Timeline",
        country_name,
        local_currency
    )

    year_col = _year_column(income_df)

    if year_col is None:
        return _empty_figure(
            title,
            "Career timeline needs a Year column."
        )

    stage_col = _resolve_column(
        income_df,
        [
            "Career Stage",
            "Income Stage",
            "Salary Stage",
            "Employment Stage"
        ]
    )

    chart_df = income_df.copy()

    if stage_col is None:
        def infer_stage(year_value):
            try:
                year_number = int(year_value)
            except (TypeError, ValueError):
                return "Unknown"

            if year_number <= 3:
                return "Graduate"

            if year_number <= 7:
                return "Mid-level"

            return "Senior"

        chart_df["Career Stage"] = chart_df[year_col].apply(infer_stage)
        stage_col = "Career Stage"

    chart_df["Timeline"] = 1

    fig = px.bar(
        chart_df,
        x=year_col,
        y="Timeline",
        color=stage_col,
        text=stage_col
    )

    fig.update_yaxes(
        visible=False,
        showticklabels=False
    )

    fig.update_traces(
        textposition="inside"
    )

    return _apply_common_layout(
        fig,
        title,
        yaxis_title="Career Stage"
    )


def build_yearly_stacked_expense_chart(
    expense_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Yearly Expense Breakdown",
        country_name,
        local_currency
    )

    year_col = _year_column(expense_df)

    expense_columns = _available_columns(
        expense_df,
        [
            "Rent",
            "Living",
            "General Living",
            "Tuition",
            "Visa",
            "Healthcare",
            "Health Care",
            "Transport",
            "Childcare",
            "Child Care",
            "Car",
            "Debt Cost",
            "Interest Paid"
        ]
    )

    if year_col is None or not expense_columns:
        return _empty_figure(
            title,
            "Expense breakdown needs Year and expense category columns."
        )

    chart_df = _melt_for_chart(
        expense_df,
        year_col,
        expense_columns
    )

    fig = px.bar(
        chart_df,
        x=year_col,
        y="Amount",
        color="Category",
        barmode="stack"
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_expense_breakdown_donut_chart(
    expense_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "10-Year Expense Breakdown",
        country_name,
        local_currency
    )

    expense_columns = _available_columns(
        expense_df,
        [
            "Rent",
            "Living",
            "General Living",
            "Tuition",
            "Visa",
            "Healthcare",
            "Health Care",
            "Transport",
            "Childcare",
            "Child Care",
            "Car",
            "Debt Cost",
            "Interest Paid"
        ]
    )

    if not expense_columns:
        return _empty_figure(
            title,
            "Donut chart needs expense category columns."
        )

    chart_df = _copy_numeric_df(
        expense_df,
        expense_columns
    )

    total_df = pd.DataFrame(
        {
            "Category": expense_columns,
            "Amount": [
                chart_df[column].sum()
                for column in expense_columns
            ]
        }
    )

    total_df = total_df[
        total_df["Amount"] > 0
    ]

    if total_df.empty:
        return _empty_figure(
            title,
            "All expense category totals are zero for this scenario."
        )

    fig = px.pie(
        total_df,
        names="Category",
        values="Amount",
        hole=0.45
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title_text=""
    )

    return fig


def build_rent_tuition_childcare_car_chart(
    expense_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Rent vs Tuition vs Childcare vs Car Cost",
        country_name,
        local_currency
    )

    year_col = _year_column(expense_df)

    cost_columns = _available_columns(
        expense_df,
        [
            "Rent",
            "Tuition",
            "Childcare",
            "Child Care",
            "Car"
        ]
    )

    if year_col is None or not cost_columns:
        return _empty_figure(
            title,
            "This chart needs Year and at least one of Rent, Tuition, Childcare, or Car."
        )

    chart_df = _melt_for_chart(
        expense_df,
        year_col,
        cost_columns
    )

    fig = px.line(
        chart_df,
        x=year_col,
        y="Amount",
        color="Category",
        markers=True
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_expense_to_income_ratio_chart(
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Expense-to-Income Ratio",
        country_name,
        local_currency
    )

    income_year_col = _year_column(income_df)
    expense_year_col = _year_column(expense_df)

    net_income_col = _resolve_column(
        income_df,
        [
            "Net Income"
        ]
    )

    total_expense_col = _resolve_column(
        expense_df,
        [
            "Total Expenses",
            "Total Expense"
        ]
    )

    if (
        income_year_col is None
        or expense_year_col is None
        or net_income_col is None
        or total_expense_col is None
    ):
        return _empty_figure(
            title,
            "Ratio chart needs Year, Net Income, and Total Expenses columns."
        )

    income_safe = _copy_numeric_df(
        income_df[
            [
                income_year_col,
                net_income_col
            ]
        ],
        [net_income_col]
    )

    expense_safe = _copy_numeric_df(
        expense_df[
            [
                expense_year_col,
                total_expense_col
            ]
        ],
        [total_expense_col]
    )

    merged_df = income_safe.merge(
        expense_safe,
        left_on=income_year_col,
        right_on=expense_year_col,
        how="inner"
    )

    merged_df["Expense-to-Income Ratio %"] = merged_df.apply(
        lambda row: (row[total_expense_col] / row[net_income_col] * 100)
        if row[net_income_col] > 0
        else 0,
        axis=1
    )

    fig = px.line(
        merged_df,
        x=income_year_col,
        y="Expense-to-Income Ratio %",
        markers=True
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="100% danger line"
    )

    return _apply_common_layout(
        fig,
        title,
        yaxis_title="% of Net Income"
    )


def build_nav_over_time_chart(
    nav_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "NAV Over Time",
        country_name,
        local_currency
    )

    year_col = _year_column(nav_df)

    nav_col = _resolve_column(
        nav_df,
        [
            "Local Currency NAV",
            "NAV",
            "Net Asset Value"
        ]
    )

    if year_col is None or nav_col is None:
        return _empty_figure(
            title,
            "NAV chart needs Year and NAV columns."
        )

    chart_df = _copy_numeric_df(
        nav_df,
        [nav_col]
    )

    fig = px.line(
        chart_df,
        x=year_col,
        y=nav_col,
        markers=True
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Break-even"
    )

    return _apply_common_layout(fig, title, yaxis_title=f"NAV ({local_currency})")


def build_cash_flow_over_time_chart(
    nav_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Cash Flow Over Time",
        country_name,
        local_currency
    )

    year_col = _year_column(nav_df)

    cash_flow_col = _resolve_column(
        nav_df,
        [
            "Cash Flow",
            "Net Cash Flow",
            "Annual Cash Flow"
        ]
    )

    if year_col is None or cash_flow_col is None:
        return _empty_figure(
            title,
            "Cash-flow chart needs Year and Cash Flow columns."
        )

    chart_df = _copy_numeric_df(
        nav_df,
        [cash_flow_col]
    )

    fig = px.bar(
        chart_df,
        x=year_col,
        y=cash_flow_col
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Zero cash flow"
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_asset_composition_stacked_bar(
    nav_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Asset Composition",
        country_name,
        local_currency
    )

    year_col = _year_column(nav_df)

    asset_columns = _available_columns(
        nav_df,
        [
            "Cash Balance",
            "Cash Savings",
            "Cash",
            "Investment Balance",
            "Investments",
            "Superannuation Balance",
            "Superannuation",
            "Retirement Balance",
            "Car Value"
        ]
    )

    if year_col is None or not asset_columns:
        return _empty_figure(
            title,
            "Asset composition needs Year and asset category columns."
        )

    chart_df = _melt_for_chart(
        nav_df,
        year_col,
        asset_columns
    )

    fig = px.bar(
        chart_df,
        x=year_col,
        y="Amount",
        color="Category",
        barmode="stack"
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_liability_composition_chart(
    nav_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Liability Composition",
        country_name,
        local_currency
    )

    year_col = _year_column(nav_df)

    liability_columns = _available_columns(
        nav_df,
        [
            "Education Debt",
            "Migration Debt",
            "Car Loan Debt",
            "Negative Cash Debt",
            "Total Debt",
            "Total Liabilities",
            "Local Currency Liabilities",
            "Liabilities"
        ]
    )

    detailed_columns = [
        column
        for column in liability_columns
        if _normalise_name(column)
        not in {
            "total debt",
            "total liabilities",
            "local currency liabilities",
            "liabilities"
        }
    ]

    if detailed_columns:
        liability_columns = detailed_columns

    if year_col is None or not liability_columns:
        return _empty_figure(
            title,
            "Liability composition needs Year and liability/debt columns."
        )

    chart_df = _melt_for_chart(
        nav_df,
        year_col,
        liability_columns
    )

    fig = px.bar(
        chart_df,
        x=year_col,
        y="Amount",
        color="Category",
        barmode="stack"
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_debt_peak_chart(
    nav_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> go.Figure:
    title = _with_country_title(
        "Debt Peak Over Time",
        country_name,
        local_currency
    )

    year_col = _year_column(nav_df)

    debt_col = _resolve_column(
        nav_df,
        [
            "Total Debt",
            "Total Liabilities",
            "Local Currency Liabilities",
            "Liabilities"
        ]
    )

    if year_col is None or debt_col is None:
        return _empty_figure(
            title,
            "Debt peak chart needs Year and Total Debt or Total Liabilities columns."
        )

    chart_df = _copy_numeric_df(
        nav_df,
        [debt_col]
    )

    if chart_df.empty:
        return _empty_figure(
            title,
            "Debt dataframe is empty."
        )

    peak_index = chart_df[debt_col].idxmax()
    peak_year = chart_df.loc[peak_index, year_col]
    peak_debt = chart_df.loc[peak_index, debt_col]

    fig = px.line(
        chart_df,
        x=year_col,
        y=debt_col,
        markers=True
    )

    fig.add_scatter(
        x=[peak_year],
        y=[peak_debt],
        mode="markers+text",
        text=["Peak debt"],
        textposition="top center",
        name="Peak Debt"
    )

    return _apply_common_layout(fig, title, yaxis_title=local_currency)


def build_all_advanced_charts(
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    local_currency: str = "LOCAL",
    country_name: str | None = None
) -> Dict[str, Dict[str, go.Figure]]:
    """
    Build all advanced charts and return them grouped for Streamlit display.
    """

    return {
        "Income charts": {
            "Income breakdown stacked bar": build_income_breakdown_stacked_bar(
                income_df,
                local_currency,
                country_name
            ),
            "Gross vs net income line chart": build_gross_vs_net_income_line_chart(
                income_df,
                local_currency,
                country_name
            ),
            "Tax over time": build_tax_over_time_chart(
                income_df,
                local_currency,
                country_name
            ),
            "Spouse income contribution": build_spouse_income_contribution_chart(
                income_df,
                local_currency,
                country_name
            ),
            "Career stage timeline": build_career_stage_timeline(
                income_df,
                local_currency,
                country_name
            )
        },
        "Expense charts": {
            "Yearly stacked expense chart": build_yearly_stacked_expense_chart(
                expense_df,
                local_currency,
                country_name
            ),
            "10-year expense breakdown donut chart": build_expense_breakdown_donut_chart(
                expense_df,
                local_currency,
                country_name
            ),
            "Rent vs tuition vs childcare vs car cost": build_rent_tuition_childcare_car_chart(
                expense_df,
                local_currency,
                country_name
            ),
            "Expense-to-income ratio": build_expense_to_income_ratio_chart(
                income_df,
                expense_df,
                local_currency,
                country_name
            )
        },
        "NAV charts": {
            "NAV over time": build_nav_over_time_chart(
                nav_df,
                local_currency,
                country_name
            ),
            "Cash flow over time": build_cash_flow_over_time_chart(
                nav_df,
                local_currency,
                country_name
            ),
            "Asset composition stacked bar": build_asset_composition_stacked_bar(
                nav_df,
                local_currency,
                country_name
            ),
            "Liability composition chart": build_liability_composition_chart(
                nav_df,
                local_currency,
                country_name
            ),
            "Debt peak chart": build_debt_peak_chart(
                nav_df,
                local_currency,
                country_name
            )
        }
    }
