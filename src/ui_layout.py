import streamlit as st
import plotly.express as px

from src.scenario_builder import (
    MIGRATION_PATH_OPTIONS,
    LIFE_SCENARIO_OPTIONS,
    CAR_OPTIONS,
    INVESTMENT_OPTIONS,
    SPOUSE_INCOME_OPTIONS
)

from src.ui_helpers import (
    format_local_currency,
    format_lkr,
    to_key_value_dataframe,
    dataframe_to_csv_bytes,
    dictionary_to_json_bytes,
    render_metric_grid,
    create_dataset_key_values_dataframe
)


def get_local_currency_from_summary(summary: dict) -> str:
    return str(summary.get("currency", "LOCAL")).upper()


def get_local_currency_from_outputs(simulation_outputs: dict | None) -> str:
    if simulation_outputs is None:
        return "LOCAL"

    return str(simulation_outputs.get("local_currency", "LOCAL")).upper()


def get_country_from_outputs(
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> str:
    if selected_country:
        return selected_country

    if simulation_outputs is None:
        return "Selected Country"

    return str(simulation_outputs.get("selected_country", "Selected Country"))


def format_country_context(
    selected_country: str | None,
    local_currency: str | None
) -> str:
    country = selected_country or "Selected Country"
    currency = str(local_currency or "LOCAL").upper()

    return f"{country} ({currency})"


def format_local(value, currency: str) -> str:
    return format_local_currency(
        value=value,
        currency=currency
    )


def format_probability(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def get_first_existing_column(df, column_candidates: list[str]) -> str | None:
    if df is None or getattr(df, "empty", True):
        return None

    for column in column_candidates:
        if column in df.columns:
            return column

    return None


def render_header() -> None:
    st.title("Multi-Country 10-Year NAV Simulator")
    st.caption(
        "Financial simulator for migration path, income, expenses, assets, debt, "
        "scenario comparison, country comparison, sensitivity analysis, testing, and export."
    )


def render_sidebar_inputs(default_inputs: dict) -> dict:
    st.sidebar.header("Scenario Inputs")

    migration_path_label = st.sidebar.selectbox(
        "Migration path",
        options=list(MIGRATION_PATH_OPTIONS.keys())
    )

    life_scenario_label = st.sidebar.selectbox(
        "Life scenario",
        options=list(LIFE_SCENARIO_OPTIONS.keys())
    )

    car_option_label = st.sidebar.selectbox(
        "Car option",
        options=list(CAR_OPTIONS.keys())
    )

    investment_option_label = st.sidebar.selectbox(
        "Investment option",
        options=list(INVESTMENT_OPTIONS.keys())
    )

    spouse_income_case_label = st.sidebar.selectbox(
        "Spouse income case",
        options=list(SPOUSE_INCOME_OPTIONS.keys()),
        index=1
    )

    st.sidebar.divider()
    st.sidebar.header("Adjustable Inputs")

    salary_growth_rate = st.sidebar.slider(
        "Salary growth rate",
        min_value=0.00,
        max_value=0.10,
        value=float(default_inputs["salary_growth_rate"]),
        step=0.01,
        format="%.2f"
    )

    inflation_rate = st.sidebar.slider(
        "Inflation rate",
        min_value=0.00,
        max_value=0.10,
        value=float(default_inputs["inflation_rate"]),
        step=0.01,
        format="%.2f"
    )

    investment_return_rate = st.sidebar.slider(
        "Investment return rate",
        min_value=0.00,
        max_value=0.15,
        value=float(default_inputs["investment_return_rate"]),
        step=0.01,
        format="%.2f"
    )

    rent_multiplier = st.sidebar.slider(
        "Rent multiplier",
        min_value=0.50,
        max_value=1.50,
        value=float(default_inputs["rent_multiplier"]),
        step=0.05,
        format="%.2f"
    )

    tuition_multiplier = st.sidebar.slider(
        "Tuition multiplier",
        min_value=0.50,
        max_value=1.50,
        value=float(default_inputs["tuition_multiplier"]),
        step=0.05,
        format="%.2f"
    )

    childcare_multiplier = st.sidebar.slider(
        "Childcare multiplier",
        min_value=0.50,
        max_value=1.50,
        value=float(default_inputs["childcare_multiplier"]),
        step=0.05,
        format="%.2f"
    )

    run_simulation = st.sidebar.button(
        "Run Simulation",
        type="primary",
        use_container_width=True
    )

    return {
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
        "childcare_multiplier": childcare_multiplier,
        "run_simulation": run_simulation
    }


def render_simulation_required_message() -> None:
    st.warning("Choose country and scenario inputs from the sidebar, then click Run Simulation.")


def render_dashboard_tab(
    summary: dict,
    scenario_config: dict,
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    local_currency = get_local_currency_from_summary(summary)
    country_context = format_country_context(
        selected_country or summary.get("country"),
        local_currency
    )

    st.subheader(f"Dataset Snapshot - {country_context}")

    dataset_metrics = [
        {
            "label": "Country",
            "value": summary.get("country", selected_country or "N/A")
        },
        {
            "label": "Currency",
            "value": local_currency
        },
        {
            "label": "Base Year",
            "value": summary.get("base_year", "N/A")
        },
        {
            "label": "Time Horizon",
            "value": f"{summary.get('time_horizon_years', 'N/A')} years"
        }
    ]

    render_metric_grid(dataset_metrics, columns_per_row=4)

    st.divider()

    st.subheader(f"Selected Scenario - {country_context}")

    selected_labels = scenario_config["selected_labels"]

    selected_metrics = [
        {
            "label": "Migration Path",
            "value": selected_labels["migration_path"]
        },
        {
            "label": "Life Scenario",
            "value": selected_labels["life_scenario"]
        },
        {
            "label": "Car Option",
            "value": selected_labels["car_option"]
        },
        {
            "label": "Investment",
            "value": selected_labels["investment_option"]
        }
    ]

    render_metric_grid(selected_metrics, columns_per_row=4)

    if simulation_outputs is None:
        st.divider()
        render_simulation_required_message()
        return

    st.divider()
    st.success(f"Simulation completed successfully for {country_context}.")

    income_summary = simulation_outputs["income_summary"]
    expense_summary = simulation_outputs["expense_summary"]
    nav_summary = simulation_outputs["nav_summary"]
    exchange_rate = simulation_outputs["exchange_rate"]

    year_10_nav_local = nav_summary["year_10_nav"]
    year_10_nav_lkr = nav_summary.get(
        "year_10_nav_lkr",
        year_10_nav_local * exchange_rate
    )
    year_10_nav_present_value_lkr = nav_summary.get(
        "year_10_nav_present_value_lkr",
        year_10_nav_lkr
    )

    key_result_metrics = [
        {
            "label": f"Final Year-10 NAV ({local_currency})",
            "value": format_local(year_10_nav_local, local_currency)
        },
        {
            "label": "Year-10 NAV in LKR",
            "value": format_lkr(year_10_nav_lkr)
        },
        {
            "label": "Today's Value of Year-10 NAV",
            "value": format_lkr(year_10_nav_present_value_lkr)
        },
        {
            "label": f"Total Income ({local_currency})",
            "value": format_local(income_summary["total_net_income"], local_currency)
        },
        {
            "label": f"Total Expenses ({local_currency})",
            "value": format_local(expense_summary["total_expenses"], local_currency)
        },
        {
            "label": f"Total Debt ({local_currency})",
            "value": format_local(nav_summary["year_10_total_liabilities"], local_currency)
        },
        {
            "label": f"Final Investment Balance ({local_currency})",
            "value": format_local(nav_summary["year_10_investment_balance"], local_currency)
        },
        {
            "label": f"Final Retirement Balance ({local_currency})",
            "value": format_local(nav_summary["year_10_superannuation_balance"], local_currency)
        },
        {
            "label": f"Cash Savings ({local_currency})",
            "value": format_local(nav_summary["year_10_cash_savings"], local_currency)
        },
        {
            "label": f"Car Value ({local_currency})",
            "value": format_local(nav_summary["year_10_car_value"], local_currency)
        }
    ]

    st.subheader(f"Key Results - {country_context}")
    render_metric_grid(key_result_metrics, columns_per_row=3)


def render_scenario_builder_tab(
    scenario_summary: dict,
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    st.subheader(f"Scenario Builder - {selected_country or 'Selected Country'}")

    scenario_summary_df = to_key_value_dataframe(
        data=scenario_summary,
        key_name="Input",
        value_name="Selected Value"
    )

    st.dataframe(
        scenario_summary_df,
        use_container_width=True
    )

    if simulation_outputs is None:
        render_simulation_required_message()


def render_assumptions_tab(
    summary: dict,
    dataset: dict,
    selected_country: str | None = None
) -> None:
    local_currency = get_local_currency_from_summary(summary)

    st.subheader(
        f"Model Assumptions - {format_country_context(selected_country or summary.get('country'), local_currency)}"
    )

    key_values_df = create_dataset_key_values_dataframe(summary)

    st.dataframe(
        key_values_df,
        use_container_width=True
    )

    with st.expander("View raw selected-country dataset"):
        st.json(dataset)


def render_income_tab(
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    local_currency = get_local_currency_from_outputs(simulation_outputs)
    country_context = format_country_context(
        get_country_from_outputs(simulation_outputs, selected_country),
        local_currency
    )

    st.subheader(f"Income Analysis - {country_context}")

    if simulation_outputs is None:
        render_simulation_required_message()
        return

    income_df = simulation_outputs["income_df"]
    final_simulation_df = simulation_outputs["final_simulation_df"]
    income_summary = simulation_outputs["income_summary"]

    income_metrics = [
        {
            "label": f"Total Net Income ({local_currency})",
            "value": format_local(income_summary["total_net_income"], local_currency)
        }
    ]

    if "total_gross_income" in income_summary:
        income_metrics.append(
            {
                "label": f"Total Gross Income ({local_currency})",
                "value": format_local(income_summary["total_gross_income"], local_currency)
            }
        )

    render_metric_grid(income_metrics, columns_per_row=3)

    st.subheader("Income Table")
    st.dataframe(
        income_df,
        use_container_width=True
    )

    income_chart_columns = [
        column for column in ["Gross Income", "Net Income"]
        if column in final_simulation_df.columns
    ]

    if income_chart_columns:
        st.subheader(f"Income Over Time - {country_context}")

        income_fig = px.line(
            final_simulation_df,
            x="Year",
            y=income_chart_columns,
            markers=True,
            title=f"Income Over Time - {country_context}"
        )

        income_fig.update_layout(
            xaxis_title="Year",
            yaxis_title=local_currency,
            legend_title="Income Metric"
        )

        st.plotly_chart(
            income_fig,
            use_container_width=True
        )


def render_expense_tab(
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    local_currency = get_local_currency_from_outputs(simulation_outputs)
    country_context = format_country_context(
        get_country_from_outputs(simulation_outputs, selected_country),
        local_currency
    )

    st.subheader(f"Expense Analysis - {country_context}")

    if simulation_outputs is None:
        render_simulation_required_message()
        return

    expense_df = simulation_outputs["expense_df"]
    final_simulation_df = simulation_outputs["final_simulation_df"]
    expense_summary = simulation_outputs["expense_summary"]

    expense_metrics = [
        {
            "label": f"Total Expenses ({local_currency})",
            "value": format_local(expense_summary["total_expenses"], local_currency)
        }
    ]

    render_metric_grid(expense_metrics, columns_per_row=3)

    st.subheader("Expense Table")
    st.dataframe(
        expense_df,
        use_container_width=True
    )

    if "Expenses" in final_simulation_df.columns:
        st.subheader(f"Expenses Over Time - {country_context}")

        expense_fig = px.line(
            final_simulation_df,
            x="Year",
            y="Expenses",
            markers=True,
            title=f"Expenses Over Time - {country_context}"
        )

        expense_fig.update_layout(
            xaxis_title="Year",
            yaxis_title=local_currency
        )

        st.plotly_chart(
            expense_fig,
            use_container_width=True
        )


def render_nav_tab(
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    local_currency = get_local_currency_from_outputs(simulation_outputs)
    country_context = format_country_context(
        get_country_from_outputs(simulation_outputs, selected_country),
        local_currency
    )

    st.subheader(f"NAV Analysis - {country_context}")

    if simulation_outputs is None:
        render_simulation_required_message()
        return

    final_simulation_df = simulation_outputs["final_simulation_df"]
    income_df = simulation_outputs["income_df"]
    expense_df = simulation_outputs["expense_df"]
    nav_df = simulation_outputs["nav_df"]
    nav_summary = simulation_outputs["nav_summary"]

    st.caption(
        "Present value uses: Year-10 NAV in LKR / (1 + 0.055) ^ 10."
    )

    nav_metrics = [
        {
            "label": f"Year-10 NAV ({local_currency})",
            "value": format_local(nav_summary["year_10_nav"], local_currency)
        },
        {
            "label": "Year-10 NAV in LKR",
            "value": format_lkr(nav_summary["year_10_nav_lkr"])
        },
        {
            "label": "Today's Value in LKR",
            "value": format_lkr(nav_summary["year_10_nav_present_value_lkr"])
        }
    ]

    render_metric_grid(nav_metrics, columns_per_row=3)

    st.subheader("Final Simulation Table")
    st.dataframe(
        final_simulation_df,
        use_container_width=True
    )

    with st.expander("View detailed income table"):
        st.dataframe(
            income_df,
            use_container_width=True
        )

    with st.expander("View detailed expense table"):
        st.dataframe(
            expense_df,
            use_container_width=True
        )

    with st.expander("View detailed NAV table"):
        st.dataframe(
            nav_df,
            use_container_width=True
        )

    st.subheader(f"NAV Over 10 Years - {country_context}")

    nav_column = get_first_existing_column(
        nav_df,
        [
            "Local Currency NAV",
            "NAV",
            "Net Asset Value"
        ]
    )

    if nav_column is not None:
        nav_over_time_fig = px.line(
            nav_df,
            x="Year",
            y=nav_column,
            markers=True,
            title=f"NAV Over Time - {country_context}"
        )

        nav_over_time_fig.update_layout(
            xaxis_title="Year",
            yaxis_title=f"NAV ({local_currency})"
        )

        st.plotly_chart(
            nav_over_time_fig,
            use_container_width=True
        )

    present_value_nav_column = get_first_existing_column(
        nav_df,
        [
            "Present Value LKR NAV",
            "Today's Value LKR NAV"
        ]
    )

    if present_value_nav_column is not None:
        present_value_fig = px.line(
            nav_df,
            x="Year",
            y=present_value_nav_column,
            markers=True,
            title=f"Present Value of NAV Over Time - {country_context}"
        )

        present_value_fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Present Value NAV (LKR)"
        )

        st.plotly_chart(
            present_value_fig,
            use_container_width=True
        )

    st.subheader(f"Income vs Expenses - {country_context}")

    income_expense_chart_columns = [
        column for column in ["Gross Income", "Net Income", "Expenses"]
        if column in final_simulation_df.columns
    ]

    if income_expense_chart_columns:
        income_expense_fig = px.line(
            final_simulation_df,
            x="Year",
            y=income_expense_chart_columns,
            markers=True,
            title=f"Income vs Expenses - {country_context}"
        )

        income_expense_fig.update_layout(
            xaxis_title="Year",
            yaxis_title=local_currency,
            legend_title="Metric"
        )

        st.plotly_chart(
            income_expense_fig,
            use_container_width=True
        )

    st.subheader(f"Assets vs Liabilities - {country_context}")

    assets_liabilities_chart_columns = [
        column for column in [
            "Local Currency Assets",
            "Local Currency Liabilities",
            "Local Currency NAV",
            "Total Assets",
            "Total Liabilities",
            "NAV"
        ]
        if column in nav_df.columns
    ]

    if assets_liabilities_chart_columns:
        assets_liabilities_fig = px.line(
            nav_df,
            x="Year",
            y=assets_liabilities_chart_columns,
            markers=True,
            title=f"Assets vs Liabilities - {country_context}"
        )

        assets_liabilities_fig.update_layout(
            xaxis_title="Year",
            yaxis_title=local_currency,
            legend_title="Metric"
        )

        st.plotly_chart(
            assets_liabilities_fig,
            use_container_width=True
        )


def render_scenario_comparison_tab(
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    local_currency = get_local_currency_from_outputs(simulation_outputs)
    country_context = format_country_context(
        get_country_from_outputs(simulation_outputs, selected_country),
        local_currency
    )

    st.subheader(f"Scenario Comparison - {country_context}")

    if simulation_outputs is None:
        render_simulation_required_message()
        return

    comparison_df = simulation_outputs["comparison_df"]

    st.caption(
        "This compares scenarios inside one selected country only. "
        "Use Country Comparison for Australia vs Germany vs Japan vs Sri Lanka. "
        "Present value uses: Year-10 NAV in LKR / (1 + 0.055) ^ 10."
    )

    st.subheader("Scenario Ranking by Year-10 NAV")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.subheader(f"Today's Value of Final NAV by Scenario - {country_context}")

    present_value_nav_column = get_first_existing_column(
        comparison_df,
        [
            "Year-10 NAV Present Value LKR",
            "Final NAV Present Value LKR"
        ]
    )

    nav_column = present_value_nav_column or get_first_existing_column(
        comparison_df,
        [
            "Year-10 NAV Local",
            "Final NAV Local",
            "Local Currency NAV",
            "NAV",
            "Final NAV"
        ]
    )

    if nav_column is None:
        st.warning(
            "Scenario comparison NAV column was not found. "
            "Expected `Year-10 NAV Local`."
        )
        return

    comparison_fig = px.bar(
        comparison_df,
        x="Scenario",
        y=nav_column,
        title=(
            f"Present Value of Final Year-10 NAV by Scenario - {country_context}"
            if present_value_nav_column
            else f"Final Year-10 NAV by Scenario - {country_context}"
        ),
        text_auto=".2s"
    )

    comparison_fig.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Present Value NAV (LKR)" if present_value_nav_column else f"Year-10 NAV ({local_currency})",
        xaxis_tickangle=-30
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )

    best_scenario = comparison_df.iloc[0]

    st.success(
        f"Best scenario in {selected_country or 'selected country'}: "
        f"{best_scenario['Scenario']} with Year-10 NAV of "
        f"{format_lkr(best_scenario[nav_column]) if present_value_nav_column else format_local(best_scenario[nav_column], local_currency)}."
    )


def render_sensitivity_risk_tab(
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    local_currency = get_local_currency_from_outputs(simulation_outputs)
    country_context = format_country_context(
        get_country_from_outputs(simulation_outputs, selected_country),
        local_currency
    )

    st.subheader(f"Sensitivity & Risk - {country_context}")

    if simulation_outputs is None:
        render_simulation_required_message()
        return

    sensitivity_df = simulation_outputs["sensitivity_df"]
    tornado_df = simulation_outputs["tornado_df"]

    st.write(
        "Sensitivity analysis changes one variable at a time by -20%, -10%, "
        "Base, +10%, and +20%, then recalculates Year-10 NAV and its present-day LKR value."
    )

    st.subheader("Sensitivity Table")

    st.dataframe(
        sensitivity_df,
        use_container_width=True
    )

    st.subheader(f"Sensitivity Line Chart - {country_context}")

    sensitivity_present_value_column = get_first_existing_column(
        sensitivity_df,
        [
            "Year-10 NAV Present Value LKR"
        ]
    )

    sensitivity_nav_column = sensitivity_present_value_column or get_first_existing_column(
        sensitivity_df,
        [
            "Year-10 NAV Local",
            "Final NAV Local",
            "Local Currency NAV",
            "NAV"
        ]
    )

    if sensitivity_nav_column is not None:
        sensitivity_line_fig = px.line(
            sensitivity_df,
            x="Change Label",
            y=sensitivity_nav_column,
            color="Variable",
            markers=True,
            title=(
                f"Present Value NAV Sensitivity by Variable - {country_context}"
                if sensitivity_present_value_column
                else f"Year-10 NAV Sensitivity by Variable - {country_context}"
            )
        )

        sensitivity_line_fig.update_layout(
            xaxis_title="Sensitivity Level",
            yaxis_title="Present Value NAV (LKR)" if sensitivity_present_value_column else f"Year-10 NAV ({local_currency})",
            legend_title="Variable"
        )

        st.plotly_chart(
            sensitivity_line_fig,
            use_container_width=True
        )
    else:
        st.warning(
            "Sensitivity NAV column was not found. "
            "Expected `Year-10 NAV Local`."
        )

    st.subheader(f"Tornado-Style Impact Chart - {country_context}")

    tornado_impact_column = get_first_existing_column(
        tornado_df,
        [
            "Max Impact %",
            "Max Impact Local"
        ]
    )

    if tornado_impact_column is not None:
        tornado_fig = px.bar(
            tornado_df,
            x=tornado_impact_column,
            y="Variable",
            orientation="h",
            title=f"Tornado-Style Chart: Maximum NAV Impact by Variable - {country_context}",
            text=tornado_impact_column
        )

        tornado_fig.update_layout(
            xaxis_title="Maximum NAV Impact",
            yaxis_title="Variable"
        )

        st.plotly_chart(
            tornado_fig,
            use_container_width=True
        )
    else:
        st.warning(
            "Tornado impact column was not found. "
            "Expected `Max Impact %` or `Max Impact Local`."
        )

    st.subheader("Tornado Impact Table")

    sort_column = "Max Impact %" if "Max Impact %" in tornado_df.columns else tornado_impact_column

    if sort_column is not None:
        st.dataframe(
            tornado_df.sort_values(
                by=sort_column,
                ascending=False
            ),
            use_container_width=True
        )
    else:
        st.dataframe(
            tornado_df,
            use_container_width=True
        )

    st.info(
        "Exchange-rate sensitivity affects LKR-converted NAV, not the local-currency NAV. "
        "That is why the sensitivity table includes both local currency and LKR NAV."
    )

    st.divider()

    render_monte_carlo_risk_section(
        simulation_outputs=simulation_outputs,
        country_context=country_context,
        local_currency=local_currency
    )


def render_monte_carlo_risk_section(
    simulation_outputs: dict,
    country_context: str,
    local_currency: str
) -> None:
    st.subheader("Monte Carlo Risk Simulator")

    monte_carlo_results_df = simulation_outputs.get("monte_carlo_results_df")
    monte_carlo_summary_df = simulation_outputs.get("monte_carlo_summary_df")
    monte_carlo_percentiles_df = simulation_outputs.get(
        "monte_carlo_percentiles_df"
    )
    probability_summary = simulation_outputs.get(
        "monte_carlo_probability_summary",
        {}
    )

    if (
        monte_carlo_results_df is None
        or getattr(monte_carlo_results_df, "empty", True)
    ):
        st.info("Monte Carlo results are not available for this run.")
        return

    simulation_count = int(probability_summary.get("simulation_count", 0))

    st.write(
        f"This runs {simulation_count:,} randomized future cases through the "
        "same NAV model and estimates downside probabilities."
    )

    monte_carlo_metrics = [
        {
            "label": "Probability Positive NAV",
            "value": format_probability(
                probability_summary.get("positive_nav_probability", 0.0)
            )
        },
        {
            "label": "Probability High Debt",
            "value": format_probability(
                probability_summary.get("high_debt_probability", 0.0)
            )
        },
        {
            "label": "Probability Very Bad Outcome",
            "value": format_probability(
                probability_summary.get("very_bad_outcome_probability", 0.0)
            )
        },
        {
            "label": f"Median Final NAV ({local_currency})",
            "value": format_local(
                probability_summary.get("median_final_nav_local", 0.0),
                local_currency
            )
        },
        {
            "label": f"5th Percentile NAV ({local_currency})",
            "value": format_local(
                probability_summary.get("p5_final_nav_local", 0.0),
                local_currency
            )
        },
        {
            "label": f"95th Percentile NAV ({local_currency})",
            "value": format_local(
                probability_summary.get("p95_final_nav_local", 0.0),
                local_currency
            )
        }
    ]

    render_metric_grid(monte_carlo_metrics, columns_per_row=3)

    final_nav_column = get_first_existing_column(
        monte_carlo_results_df,
        [
            "Final NAV Local",
            f"Final NAV {local_currency}"
        ]
    )

    if final_nav_column is not None:
        st.subheader(f"Final NAV Distribution - {country_context}")

        histogram_fig = px.histogram(
            monte_carlo_results_df,
            x=final_nav_column,
            nbins=40,
            title=f"Monte Carlo Final NAV Distribution - {country_context}"
        )

        histogram_fig.add_vline(
            x=0,
            line_dash="dash",
            annotation_text="Break-even"
        )

        histogram_fig.update_layout(
            xaxis_title=f"Final NAV ({local_currency})",
            yaxis_title="Simulation Count"
        )

        st.plotly_chart(
            histogram_fig,
            use_container_width=True
        )

    probability_rows = [
        {
            "Outcome": "Positive NAV",
            "Probability": probability_summary.get(
                "positive_nav_probability",
                0.0
            )
        },
        {
            "Outcome": "High Debt",
            "Probability": probability_summary.get(
                "high_debt_probability",
                0.0
            )
        },
        {
            "Outcome": "Very Bad Outcome",
            "Probability": probability_summary.get(
                "very_bad_outcome_probability",
                0.0
            )
        }
    ]

    probability_fig = px.bar(
        probability_rows,
        x="Outcome",
        y="Probability",
        text="Probability",
        title=f"Monte Carlo Risk Probabilities - {country_context}"
    )

    probability_fig.update_traces(texttemplate="%{text:.1%}")
    probability_fig.update_layout(
        xaxis_title="Risk Outcome",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        probability_fig,
        use_container_width=True
    )

    if (
        final_nav_column is not None
        and "Max Debt-to-Income Ratio" in monte_carlo_results_df.columns
    ):
        st.subheader(f"Final NAV vs Debt Pressure - {country_context}")

        hover_columns = [
            column
            for column in [
                "Simulation",
                "Final Debt",
                "Peak Debt",
                "Break-even Year",
                "Sampled Salary Growth Rate",
                "Sampled Inflation Rate",
                "Sampled Investment Return Rate"
            ]
            if column in monte_carlo_results_df.columns
        ]

        scatter_fig = px.scatter(
            monte_carlo_results_df,
            x="Max Debt-to-Income Ratio",
            y=final_nav_column,
            color="Very Bad Outcome"
            if "Very Bad Outcome" in monte_carlo_results_df.columns
            else None,
            hover_data=hover_columns,
            title=f"Final NAV vs Max Debt-to-Income Ratio - {country_context}"
        )

        scatter_fig.add_hline(
            y=0,
            line_dash="dash",
            annotation_text="Break-even NAV"
        )
        scatter_fig.add_vline(
            x=0.50,
            line_dash="dash",
            annotation_text="High debt threshold"
        )
        scatter_fig.update_layout(
            xaxis_title="Max Debt-to-Income Ratio",
            yaxis_title=f"Final NAV ({local_currency})",
            xaxis_tickformat=".0%"
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True
        )

    with st.expander("View Monte Carlo summary"):
        st.dataframe(
            monte_carlo_summary_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View Monte Carlo percentiles"):
        st.dataframe(
            monte_carlo_percentiles_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View full Monte Carlo simulation results"):
        st.dataframe(
            monte_carlo_results_df,
            use_container_width=True,
            hide_index=True
        )


def render_testing_tab(
    simulation_outputs: dict | None,
    selected_country: str | None = None
) -> None:
    st.subheader(f"Selected Country Model Testing - {selected_country or 'Selected Country'}")

    if simulation_outputs is None:
        render_simulation_required_message()
        return

    testing_df = simulation_outputs["testing_df"]

    st.write(
        "This checks the current selected-country model output directly. "
        "Multi-country dataset testing is shown below this section in app.py."
    )

    st.dataframe(
        testing_df,
        use_container_width=True
    )

    if "All Checks Passed" in testing_df.columns and testing_df["All Checks Passed"].all():
        st.success("Selected-country testing cases passed.")
    else:
        st.error(
            "Some selected-country testing cases failed. Review the testing table before demo."
        )


def render_export_tab(
    simulation_outputs: dict | None,
    scenario_config: dict,
    selected_country: str | None = None
) -> None:
    st.subheader(f"Export Results - {selected_country or 'Selected Country'}")

    if simulation_outputs is None:
        render_simulation_required_message()
        return

    safe_country = str(selected_country or "selected_country").lower().replace(" ", "_")

    final_simulation_df = simulation_outputs["final_simulation_df"]
    income_df = simulation_outputs["income_df"]
    expense_df = simulation_outputs["expense_df"]
    nav_df = simulation_outputs["nav_df"]
    comparison_df = simulation_outputs["comparison_df"]
    sensitivity_df = simulation_outputs["sensitivity_df"]
    tornado_df = simulation_outputs["tornado_df"]
    monte_carlo_summary_df = simulation_outputs.get("monte_carlo_summary_df")
    monte_carlo_results_df = simulation_outputs.get("monte_carlo_results_df")
    testing_df = simulation_outputs["testing_df"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download Final Results CSV",
            data=dataframe_to_csv_bytes(final_simulation_df),
            file_name=f"{safe_country}_final_simulation_results.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Income CSV",
            data=dataframe_to_csv_bytes(income_df),
            file_name=f"{safe_country}_income_results.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Testing CSV",
            data=dataframe_to_csv_bytes(testing_df),
            file_name=f"{safe_country}_testing_results.csv",
            mime="text/csv"
        )

        if monte_carlo_summary_df is not None:
            st.download_button(
                label="Download Monte Carlo Summary CSV",
                data=dataframe_to_csv_bytes(monte_carlo_summary_df),
                file_name=f"{safe_country}_monte_carlo_summary.csv",
                mime="text/csv"
            )

    with col2:
        st.download_button(
            label="Download Expense CSV",
            data=dataframe_to_csv_bytes(expense_df),
            file_name=f"{safe_country}_expense_results.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download NAV CSV",
            data=dataframe_to_csv_bytes(nav_df),
            file_name=f"{safe_country}_nav_results.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Tornado CSV",
            data=dataframe_to_csv_bytes(tornado_df),
            file_name=f"{safe_country}_tornado_impact_results.csv",
            mime="text/csv"
        )

        if monte_carlo_results_df is not None:
            st.download_button(
                label="Download Monte Carlo Results CSV",
                data=dataframe_to_csv_bytes(monte_carlo_results_df),
                file_name=f"{safe_country}_monte_carlo_results.csv",
                mime="text/csv"
            )

    with col3:
        st.download_button(
            label="Download Scenario JSON",
            data=dictionary_to_json_bytes(scenario_config),
            file_name=f"{safe_country}_selected_scenario_config.json",
            mime="application/json"
        )

        st.download_button(
            label="Download Scenario Comparison CSV",
            data=dataframe_to_csv_bytes(comparison_df),
            file_name=f"{safe_country}_scenario_comparison.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Sensitivity CSV",
            data=dataframe_to_csv_bytes(sensitivity_df),
            file_name=f"{safe_country}_sensitivity_analysis.csv",
            mime="text/csv"
        )

    with st.expander("View selected scenario JSON"):
        st.json(scenario_config)
