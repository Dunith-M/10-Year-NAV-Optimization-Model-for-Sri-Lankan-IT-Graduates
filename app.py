import os

import streamlit as st

from src.country_manager import (
    get_available_countries,
    get_country_config,
    get_default_country
)

from src.data_loader import (
    load_dataset,
    get_dataset_summary,
    DatasetValidationError
)

from src.currency_utils import (
    format_local_currency,
    format_lkr
)

from src.assumption_explorer import (
    build_assumption_table,
    render_assumption_explorer
)

from src.source_explorer import (
    build_source_table,
    has_unclean_source_references,
    render_source_explorer
)

from src.scenario_builder import (
    get_default_model_inputs,
    build_scenario_config,
    create_scenario_summary
)

from src.scenario_options import (
    EDUCATION_MODE_OPTIONS,
    PR_TIMING_OPTIONS,
    CAR_PURCHASE_TIMING_OPTIONS,
    FIRST_CHILD_TIMING_OPTIONS,
    SECOND_CHILD_TIMING_OPTIONS,
    INVESTMENT_SPLIT_OPTIONS
)

from src.income_model import (
    calculate_yearly_income,
    get_income_summary
)

from src.expense_model import (
    calculate_yearly_expenses,
    get_expense_summary
)

from src.nav_model import (
    calculate_nav_simulation,
    get_nav_summary,
    create_final_simulation_table
)

from src.sensitivity import build_sensitivity_analysis
from src.testing_helpers import run_model_validation_tests

from src.report_builder import (
    get_dataset_last_updated,
    build_executive_summary_dataframe,
    build_full_simulation_excel,
    build_scenario_report_json,
    build_simple_html_report,
    build_model_limitations_dataframe
)

from src.comparison_model import (
    build_scenario_comparison,
    compare_selected_vs_best
)

from src.country_comparison import (
    build_country_comparison,
    render_country_comparison_tab
)

from src.multi_country_testing import (
    run_multi_country_validation_tests,
    render_multi_country_testing_tab
)

from src.risk_model import (
    get_most_sensitive_variable,
    get_best_case_nav,
    get_worst_case_nav,
    get_risk_summary_text,
    get_risk_level
)

from src.decision_summary import (
    build_decision_summary,
    render_decision_summary_dashboard
)

from src.chart_builder import build_all_advanced_charts

from src.ui_layout import (
    render_header,
    render_sidebar_inputs,
    render_dashboard_tab,
    render_scenario_builder_tab,
    render_assumptions_tab,
    render_income_tab,
    render_expense_tab,
    render_nav_tab,
    render_scenario_comparison_tab,
    render_sensitivity_risk_tab,
    render_testing_tab,
    render_export_tab
)


STYLE_PATH = "assets/style.css"
PROJECT_TITLE = "Multi-Country 10-Year NAV Simulator"


st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🌍",
    layout="wide"
)


def load_custom_css(css_path: str = STYLE_PATH) -> None:
    """
    Load optional CSS polish if assets/style.css exists.
    The app still works normally if the file is missing.
    """

    if not os.path.exists(css_path):
        return

    with open(css_path, "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )


def get_option_index(options, selected_value):
    """
    Safe helper for Streamlit selectbox index.
    """

    if selected_value in options:
        return options.index(selected_value)

    return 0


def render_country_selection():
    """
    Render country selector and return selected country config.

    Country selector -> registry -> dataset path -> selected JSON load.
    """

    available_countries = get_available_countries()
    default_country = get_default_country()

    if not available_countries:
        st.sidebar.error("No countries found in country_registry.json.")
        st.stop()

    st.sidebar.markdown("## Country Selection")

    selected_country = st.sidebar.selectbox(
        "Country",
        available_countries,
        index=get_option_index(
            available_countries,
            default_country
        )
    )

    country_config = get_country_config(selected_country)
    dataset_path = country_config["dataset_path"]

    st.sidebar.caption(f"Dataset: `{dataset_path}`")
    st.sidebar.caption(f"Currency: `{country_config.get('currency', 'N/A')}`")

    return selected_country, country_config, dataset_path


def reset_simulation_if_country_changed(selected_country: str) -> None:
    """
    Prevent stale simulation results when user changes country.
    """

    previous_country = st.session_state.get("selected_country_name")

    if previous_country != selected_country:
        st.session_state["selected_country_name"] = selected_country
        st.session_state["simulation_has_run"] = False


def get_exchange_rate_to_lkr(summary: dict) -> float:
    """
    Read selected country's local-currency-to-LKR exchange rate from summary.
    """

    try:
        return float(summary.get("exchange_rate_to_lkr", 1.0))

    except (TypeError, ValueError):
        return 1.0


def get_numeric_last_value(df, column_candidates) -> float:
    """
    Read the final numeric value from the first matching column.
    Used only for simple UI indicators.
    """

    if df is None or getattr(df, "empty", True):
        return 0.0

    for column in column_candidates:
        if column in df.columns:
            try:
                return float(df[column].iloc[-1])
            except (TypeError, ValueError):
                return 0.0

    return 0.0


def get_dict_numeric_value(data: dict, key_candidates: list[str]) -> float:
    """
    Read a numeric value from the first matching dictionary key.
    """

    if not isinstance(data, dict):
        return 0.0

    for key in key_candidates:
        if key in data:
            try:
                return float(data.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0

    return 0.0


def get_final_nav_local(simulation_outputs) -> float:
    """
    Get final NAV from nav_df in selected local currency.
    """

    if simulation_outputs is None:
        return 0.0

    nav_df = simulation_outputs.get("nav_df")

    return get_numeric_last_value(
        df=nav_df,
        column_candidates=[
            "Local Currency NAV",
            "NAV",
            "Net Asset Value",
            "Year-10 NAV Local",
            "Final NAV Local"
        ]
    )


def get_final_debt_local(simulation_outputs) -> float:
    """
    Get final debt from nav_df in selected local currency.
    """

    if simulation_outputs is None:
        return 0.0

    nav_df = simulation_outputs.get("nav_df")

    return get_numeric_last_value(
        df=nav_df,
        column_candidates=[
            "Local Currency Debt",
            "Total Debt",
            "Local Currency Liabilities",
            "Total Liabilities",
            "Liabilities",
            "Debt"
        ]
    )


def render_dataset_status_panel(
    summary: dict,
    dataset_last_updated: str,
    unclean_source_warning: bool
) -> None:
    """
    Show dataset freshness and source warnings clearly.
    """

    st.markdown("### Dataset Status")

    columns = st.columns(5)

    with columns[0]:
        st.metric(
            "Country",
            summary.get("country", "N/A")
        )

    with columns[1]:
        st.metric(
            "Currency",
            summary.get("currency", "N/A")
        )

    with columns[2]:
        st.metric(
            "Base year",
            summary.get("base_year", "N/A")
        )

    with columns[3]:
        st.metric(
            "Time horizon",
            f"{summary.get('time_horizon_years', 'N/A')} years"
        )

    with columns[4]:
        st.metric(
            "Dataset last updated",
            dataset_last_updated
        )

    st.caption(
        f"Exchange-rate field: `{summary.get('exchange_rate_key', 'N/A')}`"
    )

    if unclean_source_warning:
        st.warning(
            "Some dataset source fields still contain placeholder-style references. "
            "Clean source URLs before final submission or viva."
        )
    else:
        st.success(
            "Source fields look clean enough for demo use."
        )


def render_nav_health_indicator(simulation_outputs, dataset) -> None:
    """
    Show green/red NAV indicator for the selected country and scenario.
    """

    if simulation_outputs is None:
        st.info("Run the simulation to see the financial health indicator.")
        return

    final_nav_local = get_final_nav_local(simulation_outputs)
    final_debt_local = get_final_debt_local(simulation_outputs)

    st.markdown("### Financial Health Indicator")

    if final_nav_local > 0 and final_debt_local <= 0:
        st.success(
            f"Strong result: final NAV is positive at "
            f"{format_local_currency(value=final_nav_local, dataset=dataset)} "
            "and final debt is not a major problem."
        )
    elif final_nav_local > 0:
        st.warning(
            f"Mixed result: final NAV is positive at "
            f"{format_local_currency(value=final_nav_local, dataset=dataset)}, "
            f"but final debt/liabilities are still around "
            f"{format_local_currency(value=final_debt_local, dataset=dataset)}."
        )
    elif final_nav_local == 0:
        st.info(
            "Neutral result: final NAV is around zero. This scenario has little financial margin."
        )
    else:
        st.error(
            f"Weak result: final NAV is negative at "
            f"{format_local_currency(value=final_nav_local, dataset=dataset)}. "
            "This scenario is financially risky unless income, rent, tuition, or debt assumptions improve."
        )


def render_model_limitations_section() -> None:
    """
    Show model limitations for academic transparency.
    """

    st.markdown("### Model Limitations")

    limitations_df = build_model_limitations_dataframe()

    st.dataframe(
        limitations_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These limitations should be mentioned during viva. "
        "They make the project more credible because the model is transparent about uncertainty."
    )


def add_advanced_sidebar_inputs(sidebar_inputs: dict) -> dict:
    """
    Add advanced scenario options.

    These advanced options override model logic only.
    The JSON dataset is not changed.
    """

    with st.sidebar.expander("Advanced scenario options", expanded=False):
        st.caption(
            "These options override model logic only. The JSON dataset is not changed."
        )

        migration_path_label = sidebar_inputs["migration_path_label"]

        if migration_path_label == "Student visa path":
            default_education_mode = "Master’s full-time"
        else:
            default_education_mode = "No further study"

        education_mode_label = st.selectbox(
            "Education mode",
            EDUCATION_MODE_OPTIONS,
            index=get_option_index(
                EDUCATION_MODE_OPTIONS,
                default_education_mode
            )
        )

        pr_timing_label = st.selectbox(
            "PR timing",
            PR_TIMING_OPTIONS,
            index=get_option_index(
                PR_TIMING_OPTIONS,
                "Normal PR"
            )
        )

        custom_pr_year = None

        if pr_timing_label == "Custom PR year":
            custom_pr_year = int(
                st.number_input(
                    "Custom PR application year",
                    min_value=1,
                    max_value=10,
                    value=6,
                    step=1
                )
            )

        if sidebar_inputs.get("car_option_label") == "No car":
            default_car_purchase_timing = "No car"
        else:
            default_car_purchase_timing = "Buy car in Year 3"

        car_purchase_timing_label = st.selectbox(
            "Car purchase timing",
            CAR_PURCHASE_TIMING_OPTIONS,
            index=get_option_index(
                CAR_PURCHASE_TIMING_OPTIONS,
                default_car_purchase_timing
            )
        )

        first_child_timing_label = st.selectbox(
            "First child timing",
            FIRST_CHILD_TIMING_OPTIONS,
            index=get_option_index(
                FIRST_CHILD_TIMING_OPTIONS,
                "Dataset default"
            )
        )

        second_child_timing_label = st.selectbox(
            "Second child timing",
            SECOND_CHILD_TIMING_OPTIONS,
            index=get_option_index(
                SECOND_CHILD_TIMING_OPTIONS,
                "Dataset default"
            )
        )

        if sidebar_inputs.get("investment_option_label") == "Save only":
            default_investment_split = "Save only"
        else:
            default_investment_split = "Invest 100%"

        investment_split_label = st.selectbox(
            "Investment split",
            INVESTMENT_SPLIT_OPTIONS,
            index=get_option_index(
                INVESTMENT_SPLIT_OPTIONS,
                default_investment_split
            )
        )

    sidebar_inputs["education_mode_label"] = education_mode_label
    sidebar_inputs["pr_timing_label"] = pr_timing_label
    sidebar_inputs["custom_pr_year"] = custom_pr_year
    sidebar_inputs["car_purchase_timing_label"] = car_purchase_timing_label
    sidebar_inputs["first_child_timing_label"] = first_child_timing_label
    sidebar_inputs["second_child_timing_label"] = second_child_timing_label
    sidebar_inputs["investment_split_label"] = investment_split_label

    if car_purchase_timing_label == "No car":
        sidebar_inputs["car_option_label"] = "No car"
    else:
        sidebar_inputs["car_option_label"] = "Buy car"

    if investment_split_label == "Save only":
        sidebar_inputs["investment_option_label"] = "Save only"
    else:
        sidebar_inputs["investment_option_label"] = "Invest positive cash flow"

    return sidebar_inputs


def run_nav_simulation(
    dataset,
    scenario_config,
    sidebar_inputs,
    exchange_rate,
    local_currency,
    selected_country
) -> dict:
    """
    Run the full simulation pipeline for the selected country and scenario.
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

    final_simulation_df = create_final_simulation_table(
        income_df=income_df,
        expense_df=expense_df,
        nav_df=nav_df
    )

    income_summary = get_income_summary(income_df)
    expense_summary = get_expense_summary(expense_df)
    nav_summary = get_nav_summary(nav_df)

    comparison_df = build_scenario_comparison(
        dataset=dataset,
        car_option_label=sidebar_inputs["car_option_label"],
        investment_option_label=sidebar_inputs["investment_option_label"],
        spouse_income_case_label=sidebar_inputs["spouse_income_case_label"],
        salary_growth_rate=sidebar_inputs["salary_growth_rate"],
        inflation_rate=sidebar_inputs["inflation_rate"],
        investment_return_rate=sidebar_inputs["investment_return_rate"],
        rent_multiplier=sidebar_inputs["rent_multiplier"],
        tuition_multiplier=sidebar_inputs["tuition_multiplier"],
        childcare_multiplier=sidebar_inputs["childcare_multiplier"],
        education_mode_label=sidebar_inputs["education_mode_label"],
        pr_timing_label=sidebar_inputs["pr_timing_label"],
        custom_pr_year=sidebar_inputs["custom_pr_year"],
        car_purchase_timing_label=sidebar_inputs["car_purchase_timing_label"],
        first_child_timing_label=sidebar_inputs["first_child_timing_label"],
        second_child_timing_label=sidebar_inputs["second_child_timing_label"],
        investment_split_label=sidebar_inputs["investment_split_label"]
    )

    country_comparison_df = build_country_comparison(
        migration_path_label=sidebar_inputs["migration_path_label"],
        life_scenario_label=sidebar_inputs["life_scenario_label"],
        car_option_label=sidebar_inputs["car_option_label"],
        investment_option_label=sidebar_inputs["investment_option_label"],
        spouse_income_case_label=sidebar_inputs["spouse_income_case_label"],
        salary_growth_rate=sidebar_inputs["salary_growth_rate"],
        inflation_rate=sidebar_inputs["inflation_rate"],
        investment_return_rate=sidebar_inputs["investment_return_rate"],
        rent_multiplier=sidebar_inputs["rent_multiplier"],
        tuition_multiplier=sidebar_inputs["tuition_multiplier"],
        childcare_multiplier=sidebar_inputs["childcare_multiplier"],
        education_mode_label=sidebar_inputs["education_mode_label"],
        pr_timing_label=sidebar_inputs["pr_timing_label"],
        custom_pr_year=sidebar_inputs["custom_pr_year"],
        car_purchase_timing_label=sidebar_inputs["car_purchase_timing_label"],
        first_child_timing_label=sidebar_inputs["first_child_timing_label"],
        second_child_timing_label=sidebar_inputs["second_child_timing_label"],
        investment_split_label=sidebar_inputs["investment_split_label"]
    )

    sensitivity_result = build_sensitivity_analysis(
        dataset=dataset,
        base_scenario_config=scenario_config
    )

    comparison_result = compare_selected_vs_best(
        comparison_df=comparison_df,
        selected_migration_path_label=sidebar_inputs["migration_path_label"],
        selected_life_scenario_label=sidebar_inputs["life_scenario_label"]
    )

    most_sensitive_variable = get_most_sensitive_variable(
        sensitivity_result["tornado_df"]
    )

    best_case_nav = get_best_case_nav(
        sensitivity_result["sensitivity_df"]
    )

    worst_case_nav = get_worst_case_nav(
        sensitivity_result["sensitivity_df"]
    )

    risk_level = get_risk_level(
        sensitivity_df=sensitivity_result["sensitivity_df"],
        tornado_df=sensitivity_result["tornado_df"]
    )

    risk_summary_text = get_risk_summary_text(
        comparison_result=comparison_result,
        most_sensitive_variable=most_sensitive_variable,
        best_case_nav=best_case_nav,
        worst_case_nav=worst_case_nav,
        risk_level=risk_level
    )

    risk_result = {
        "most_sensitive_variable": most_sensitive_variable,
        "best_case_nav": best_case_nav,
        "worst_case_nav": worst_case_nav,
        "risk_level": risk_level,
        "risk_summary_text": risk_summary_text
    }

    decision_summary = build_decision_summary(
        income_df=income_df,
        expense_df=expense_df,
        nav_df=nav_df,
        comparison_df=comparison_df,
        sensitivity_df=sensitivity_result["sensitivity_df"],
        tornado_df=sensitivity_result["tornado_df"],
        exchange_rate=exchange_rate,
        local_currency=local_currency
    )

    testing_df = run_model_validation_tests(
        dataset=dataset,
        scenario_config=scenario_config,
        income_df=income_df,
        expense_df=expense_df,
        nav_df=nav_df
    )

    multi_country_testing_df = run_multi_country_validation_tests()

    return {
        "selected_country": selected_country,
        "dataset": dataset,
        "local_currency": local_currency,
        "income_df": income_df,
        "expense_df": expense_df,
        "nav_df": nav_df,
        "final_simulation_df": final_simulation_df,
        "income_summary": income_summary,
        "expense_summary": expense_summary,
        "nav_summary": nav_summary,
        "comparison_df": comparison_df,
        "comparison_result": comparison_result,
        "country_comparison_df": country_comparison_df,
        "sensitivity_df": sensitivity_result["sensitivity_df"],
        "tornado_df": sensitivity_result["tornado_df"],
        "risk_result": risk_result,
        "testing_df": testing_df,
        "multi_country_testing_df": multi_country_testing_df,
        "decision_summary": decision_summary,
        "exchange_rate": exchange_rate
    }


def render_scenario_decision_panel(simulation_outputs, dataset):
    """
    Show plain-English scenario comparison results.
    """

    st.subheader("Scenario Decision Result")

    if simulation_outputs is None:
        st.info(
            "Run the simulation to see the best scenario, worst scenario, and selected scenario rank."
        )
        return

    comparison_result = simulation_outputs["comparison_result"]
    best_scenario = comparison_result.get("best_scenario", {})
    worst_scenario = comparison_result.get("worst_scenario", {})
    selected_scenario = comparison_result.get("selected_scenario", {})

    st.info(
        comparison_result.get(
            "message",
            "Scenario comparison is not available."
        )
    )

    best_scenario_nav_local = get_dict_numeric_value(
        best_scenario,
        [
            "Year-10 NAV Local",
            "Final NAV Local",
            "Local Currency NAV",
            "NAV",
            "Final NAV"
        ]
    )

    worst_scenario_nav_local = get_dict_numeric_value(
        worst_scenario,
        [
            "Year-10 NAV Local",
            "Final NAV Local",
            "Local Currency NAV",
            "NAV",
            "Final NAV"
        ]
    )

    nav_gap_local = get_dict_numeric_value(
        comparison_result,
        [
            "nav_gap_local",
            "nav_gap",
            "difference_from_best_local",
            "selected_vs_best_difference",
            "nav_gap_aud"
        ]
    )

    nav_gap_lkr = get_dict_numeric_value(
        comparison_result,
        [
            "nav_gap_lkr",
            "difference_from_best_lkr"
        ]
    )

    nav_gap_present_value_lkr = get_dict_numeric_value(
        comparison_result,
        [
            "nav_gap_present_value_lkr",
            "present_value_difference_from_best_lkr"
        ]
    )

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "Best scenario",
            best_scenario.get("Scenario", "N/A"),
            format_local_currency(
                value=best_scenario_nav_local,
                dataset=dataset
            )
        )

    with columns[1]:
        st.metric(
            "Worst scenario",
            worst_scenario.get("Scenario", "N/A"),
            format_local_currency(
                value=worst_scenario_nav_local,
                dataset=dataset
            )
        )

    with columns[2]:
        selected_rank = comparison_result.get("selected_rank")
        total_scenarios = comparison_result.get("total_scenarios", 0)

        if selected_rank is None:
            rank_label = "N/A"
        else:
            rank_label = f"{selected_rank} / {total_scenarios}"

        st.metric(
            "Selected rank",
            rank_label,
            selected_scenario.get("Scenario", "")
        )

    with columns[3]:
        st.metric(
            "Difference from best",
            format_local_currency(
                value=nav_gap_local,
                dataset=dataset
            ),
            f"Today's value: {format_lkr(nav_gap_present_value_lkr)}"
        )


def render_risk_decision_panel(simulation_outputs, dataset):
    """
    Show plain-English sensitivity and risk results.
    """

    st.subheader("Sensitivity Risk Result")

    if simulation_outputs is None:
        st.info(
            "Run the simulation to see the biggest risk variable and best/worst sensitivity cases."
        )
        return

    risk_result = simulation_outputs["risk_result"]
    most_sensitive_variable = risk_result["most_sensitive_variable"]
    best_case_nav = risk_result["best_case_nav"]
    worst_case_nav = risk_result["worst_case_nav"]

    risk_level = risk_result.get("risk_level", "Unknown")

    if risk_level == "High":
        st.error(risk_result["risk_summary_text"])
    elif risk_level == "Medium":
        st.warning(risk_result["risk_summary_text"])
    else:
        st.success(risk_result["risk_summary_text"])

    max_impact_local = get_dict_numeric_value(
        most_sensitive_variable,
        [
            "max_impact_local",
            "Max Impact Local",
            "max_impact"
        ]
    )

    best_case_nav_local = get_dict_numeric_value(
        best_case_nav,
        [
            "nav_local",
            "Year-10 NAV Local",
            "Final NAV Local",
            "nav"
        ]
    )

    worst_case_nav_local = get_dict_numeric_value(
        worst_case_nav,
        [
            "nav_local",
            "Year-10 NAV Local",
            "Final NAV Local",
            "nav"
        ]
    )

    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "Most dangerous variable",
            most_sensitive_variable.get("variable", "N/A"),
            format_local_currency(
                value=max_impact_local,
                dataset=dataset
            )
        )

    with columns[1]:
        st.metric(
            "Best-case NAV",
            format_local_currency(
                value=best_case_nav_local,
                dataset=dataset
            ),
            f"{best_case_nav.get('variable', 'N/A')} {best_case_nav.get('change_label', '')}"
        )

    with columns[2]:
        st.metric(
            "Worst-case NAV",
            format_local_currency(
                value=worst_case_nav_local,
                dataset=dataset
            ),
            f"{worst_case_nav.get('variable', 'N/A')} {worst_case_nav.get('change_label', '')}"
        )

    with columns[3]:
        st.metric(
            "Risk level",
            risk_level,
            "Based on downside sensitivity"
        )


def render_visual_analysis_section(
    simulation_outputs,
    selected_country: str,
    local_currency: str
):
    """
    Display advanced country-aware Plotly figures generated by src.chart_builder.
    """

    st.subheader(f"Advanced Visual Analysis - {selected_country} {local_currency}")

    if simulation_outputs is None:
        st.info("Run the simulation to view the advanced visual analysis charts.")
        return

    chart_groups = build_all_advanced_charts(
        income_df=simulation_outputs["income_df"],
        expense_df=simulation_outputs["expense_df"],
        nav_df=simulation_outputs["nav_df"],
        local_currency=local_currency,
        country_name=selected_country
    )

    st.caption(
        "These charts explain the financial story behind income, expenses, assets, liabilities, debt, and NAV."
    )

    for group_name, figures in chart_groups.items():
        with st.expander(group_name, expanded=False):
            figure_items = list(figures.items())

            for index in range(0, len(figure_items), 2):
                columns = st.columns(2)
                row_items = figure_items[index:index + 2]

                for column, (chart_name, figure) in zip(columns, row_items):
                    with column:
                        st.markdown(f"#### {chart_name}")
                        st.plotly_chart(
                            figure,
                            use_container_width=True,
                            key=f"advanced_chart_{selected_country}_{group_name}_{chart_name}"
                        )


def get_safe_country_filename_prefix(selected_country: str) -> str:
    return str(selected_country).lower().replace(" ", "_").replace("-", "_")


def render_report_export_tab(
    simulation_outputs,
    scenario_config,
    scenario_summary,
    dataset,
    assumption_df,
    source_df,
    dataset_last_updated,
    selected_country
) -> None:
    """
    Render clean report export options.
    """

    st.subheader(f"Clean Report Export - {selected_country}")

    st.caption(
        "Use these exports for submission, viva preparation, or sharing with a supervisor."
    )

    if simulation_outputs is None:
        st.info("Run the simulation first to generate report exports.")
        return

    safe_country = get_safe_country_filename_prefix(selected_country)

    executive_summary_df = build_executive_summary_dataframe(
        dataset=dataset,
        scenario_config=scenario_config,
        scenario_summary=scenario_summary,
        simulation_outputs=simulation_outputs,
        dataset_last_updated=dataset_last_updated
    )

    excel_bytes = build_full_simulation_excel(
        dataset=dataset,
        scenario_config=scenario_config,
        scenario_summary=scenario_summary,
        simulation_outputs=simulation_outputs,
        assumption_df=assumption_df,
        dataset_last_updated=dataset_last_updated
    )

    scenario_report_json = build_scenario_report_json(
        dataset=dataset,
        scenario_config=scenario_config,
        scenario_summary=scenario_summary,
        simulation_outputs=simulation_outputs,
        dataset_last_updated=dataset_last_updated
    )

    html_report = build_simple_html_report(
        dataset=dataset,
        scenario_config=scenario_config,
        scenario_summary=scenario_summary,
        simulation_outputs=simulation_outputs,
        assumption_df=assumption_df,
        dataset_last_updated=dataset_last_updated
    )

    country_comparison_df = simulation_outputs.get("country_comparison_df")
    multi_country_testing_df = simulation_outputs.get("multi_country_testing_df")

    columns = st.columns(4)

    with columns[0]:
        st.download_button(
            label="Executive summary CSV",
            data=executive_summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_country}_executive_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    with columns[1]:
        st.download_button(
            label="Full simulation Excel",
            data=excel_bytes,
            file_name=f"{safe_country}_full_simulation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with columns[2]:
        st.download_button(
            label="Scenario report JSON",
            data=scenario_report_json.encode("utf-8"),
            file_name=f"{safe_country}_scenario_report.json",
            mime="application/json",
            use_container_width=True
        )

    with columns[3]:
        st.download_button(
            label="Simple HTML report",
            data=html_report.encode("utf-8"),
            file_name=f"{safe_country}_scenario_report.html",
            mime="text/html",
            use_container_width=True
        )

    st.divider()

    extra_col_1, extra_col_2, extra_col_3 = st.columns(3)

    with extra_col_1:
        st.download_button(
            label="Assumptions CSV",
            data=assumption_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_country}_assumptions.csv",
            mime="text/csv",
            use_container_width=True
        )

    with extra_col_2:
        st.download_button(
            label="Sources CSV",
            data=source_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_country}_sources.csv",
            mime="text/csv",
            use_container_width=True
        )

    with extra_col_3:
        if country_comparison_df is not None:
            st.download_button(
                label="Country comparison CSV",
                data=country_comparison_df.to_csv(index=False).encode("utf-8"),
                file_name="country_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

    if multi_country_testing_df is not None:
        st.download_button(
            label="Download multi-country testing CSV",
            data=multi_country_testing_df.to_csv(index=False).encode("utf-8"),
            file_name="multi_country_testing_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    st.markdown("### Executive Summary Preview")
    st.dataframe(
        executive_summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    with st.expander("Older detailed CSV/JSON export buttons", expanded=False):
        render_export_tab(
            simulation_outputs=simulation_outputs,
            scenario_config=scenario_config,
            selected_country=selected_country
        )


def render_footer(dataset_last_updated: str, selected_country: str) -> None:
    """
    Footer for final project polish.
    """

    st.divider()

    st.caption(
        f"{PROJECT_TITLE} | Selected country: {selected_country} | "
        f"Dataset last updated: {dataset_last_updated}"
    )


load_custom_css()
render_header()

try:
    selected_country, selected_country_config, dataset_path = render_country_selection()
    reset_simulation_if_country_changed(selected_country)

    dataset = load_dataset(dataset_path)
    summary = get_dataset_summary(dataset)

    local_currency = summary.get(
        "currency",
        selected_country_config.get("currency", "LOCAL")
    )

    st.caption(
        f"Selected country: **{summary.get('country', selected_country)}** | "
        f"Currency: **{local_currency}** | "
        f"Dataset: `{dataset_path}`"
    )

    default_inputs = get_default_model_inputs(dataset)
    exchange_rate = get_exchange_rate_to_lkr(summary)
    dataset_last_updated = get_dataset_last_updated(dataset)

    assumption_df = build_assumption_table(
        dataset=dataset,
        country_name=selected_country
    )

    source_df = build_source_table(
        dataset=dataset,
        country_name=selected_country
    )

    unclean_source_warning = has_unclean_source_references(dataset)

    sidebar_inputs = render_sidebar_inputs(default_inputs)
    sidebar_inputs = add_advanced_sidebar_inputs(sidebar_inputs)

    if "simulation_has_run" not in st.session_state:
        st.session_state["simulation_has_run"] = False

    if sidebar_inputs["run_simulation"]:
        st.session_state["simulation_has_run"] = True

    scenario_config = build_scenario_config(
        dataset=dataset,
        migration_path_label=sidebar_inputs["migration_path_label"],
        life_scenario_label=sidebar_inputs["life_scenario_label"],
        car_option_label=sidebar_inputs["car_option_label"],
        investment_option_label=sidebar_inputs["investment_option_label"],
        spouse_income_case_label=sidebar_inputs["spouse_income_case_label"],
        salary_growth_rate=sidebar_inputs["salary_growth_rate"],
        inflation_rate=sidebar_inputs["inflation_rate"],
        investment_return_rate=sidebar_inputs["investment_return_rate"],
        rent_multiplier=sidebar_inputs["rent_multiplier"],
        tuition_multiplier=sidebar_inputs["tuition_multiplier"],
        childcare_multiplier=sidebar_inputs["childcare_multiplier"],
        education_mode_label=sidebar_inputs["education_mode_label"],
        pr_timing_label=sidebar_inputs["pr_timing_label"],
        custom_pr_year=sidebar_inputs["custom_pr_year"],
        car_purchase_timing_label=sidebar_inputs["car_purchase_timing_label"],
        first_child_timing_label=sidebar_inputs["first_child_timing_label"],
        second_child_timing_label=sidebar_inputs["second_child_timing_label"],
        investment_split_label=sidebar_inputs["investment_split_label"]
    )

    scenario_summary = create_scenario_summary(scenario_config)

    simulation_outputs = None

    if st.session_state["simulation_has_run"]:
        simulation_outputs = run_nav_simulation(
            dataset=dataset,
            scenario_config=scenario_config,
            sidebar_inputs=sidebar_inputs,
            exchange_rate=exchange_rate,
            local_currency=local_currency,
            selected_country=selected_country
        )

        render_decision_summary_dashboard(
            decision_summary=simulation_outputs["decision_summary"]
        )

        st.divider()
        render_nav_health_indicator(
            simulation_outputs=simulation_outputs,
            dataset=dataset
        )

        st.divider()
        render_scenario_decision_panel(
            simulation_outputs=simulation_outputs,
            dataset=dataset
        )
        render_risk_decision_panel(
            simulation_outputs=simulation_outputs,
            dataset=dataset
        )

    (
        dashboard_tab,
        scenario_builder_tab,
        assumptions_tab,
        income_analysis_tab,
        expense_analysis_tab,
        nav_analysis_tab,
        scenario_comparison_tab,
        country_comparison_tab,
        sensitivity_risk_tab,
        testing_tab,
        export_tab
    ) = st.tabs(
        [
            "Dashboard",
            "Scenario Builder",
            "Assumptions",
            "Income Analysis",
            "Expense Analysis",
            "NAV Analysis",
            "Scenario Comparison",
            "Country Comparison",
            "Sensitivity & Risk",
            "Testing",
            "Export"
        ]
    )

    with dashboard_tab:
        render_dataset_status_panel(
            summary=summary,
            dataset_last_updated=dataset_last_updated,
            unclean_source_warning=unclean_source_warning
        )

        st.divider()

        render_dashboard_tab(
            summary=summary,
            scenario_config=scenario_config,
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

        st.divider()

        render_nav_health_indicator(
            simulation_outputs=simulation_outputs,
            dataset=dataset
        )

        with st.expander("Model limitations", expanded=False):
            render_model_limitations_section()

    with scenario_builder_tab:
        render_scenario_builder_tab(
            scenario_summary=scenario_summary,
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

    with assumptions_tab:
        render_assumptions_tab(
            summary=summary,
            dataset=dataset,
            selected_country=selected_country
        )

        st.divider()

        render_dataset_status_panel(
            summary=summary,
            dataset_last_updated=dataset_last_updated,
            unclean_source_warning=unclean_source_warning
        )

        st.divider()

        render_assumption_explorer(
            assumption_df=assumption_df,
            selected_country=selected_country
        )

        st.divider()

        render_source_explorer(
            source_df=source_df,
            show_unclean_source_warning=unclean_source_warning,
            selected_country=selected_country
        )

    with income_analysis_tab:
        st.caption(
            f"Income analysis for {selected_country} explains salary growth, spouse income, tax, net income, and retirement contributions."
        )
        render_income_tab(
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

    with expense_analysis_tab:
        st.caption(
            f"Expense analysis for {selected_country} explains rent, living cost, tuition, visa cost, childcare, transport, car cost, and debt pressure."
        )
        render_expense_tab(
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

    with nav_analysis_tab:
        st.caption(
            f"NAV analysis for {selected_country} shows assets, liabilities, debt, cash flow, investment, and retirement balance."
        )
        render_nav_tab(
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

        st.divider()

        render_visual_analysis_section(
            simulation_outputs=simulation_outputs,
            selected_country=selected_country,
            local_currency=local_currency
        )

    with scenario_comparison_tab:
        render_scenario_decision_panel(
            simulation_outputs=simulation_outputs,
            dataset=dataset
        )
        st.divider()
        render_scenario_comparison_tab(
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

    with country_comparison_tab:
        country_comparison_df = (
            simulation_outputs.get("country_comparison_df")
            if simulation_outputs is not None
            else None
        )

        render_country_comparison_tab(
            country_comparison_df=country_comparison_df,
            selected_country=selected_country
        )

    with sensitivity_risk_tab:
        render_risk_decision_panel(
            simulation_outputs=simulation_outputs,
            dataset=dataset
        )
        st.divider()
        render_sensitivity_risk_tab(
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

    with testing_tab:
        st.caption(
            "Testing is split into selected-country model testing and full multi-country dataset testing."
        )

        render_testing_tab(
            simulation_outputs=simulation_outputs,
            selected_country=selected_country
        )

        st.divider()

        multi_country_testing_df = (
            simulation_outputs.get("multi_country_testing_df")
            if simulation_outputs is not None
            else None
        )

        render_multi_country_testing_tab(
            testing_df=multi_country_testing_df
        )

    with export_tab:
        render_report_export_tab(
            simulation_outputs=simulation_outputs,
            scenario_config=scenario_config,
            scenario_summary=scenario_summary,
            dataset=dataset,
            assumption_df=assumption_df,
            source_df=source_df,
            dataset_last_updated=dataset_last_updated,
            selected_country=selected_country
        )

    render_footer(
        dataset_last_updated=dataset_last_updated,
        selected_country=selected_country
    )

except FileNotFoundError as error:
    st.error("Required file was not found.")
    st.code(str(error))

except DatasetValidationError as error:
    st.error("Dataset validation failed.")
    st.code(str(error))

except Exception as error:
    st.error("Unexpected error occurred.")
    st.code(str(error))

print("Hello World")