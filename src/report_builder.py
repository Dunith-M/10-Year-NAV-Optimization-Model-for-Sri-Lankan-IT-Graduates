import io
import json
import html
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.currency_utils import calculate_lkr_present_value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def _format_percent(value: Any, decimals: int = 1) -> str:
    return f"{_safe_float(value) * 100:.{decimals}f}%"


def _json_default(value: Any):
    """
    JSON fallback for pandas/numpy values.
    """

    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def _ensure_dataframe(value: Any) -> pd.DataFrame:
    """
    Convert common values into a DataFrame safely.
    """

    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):
        return pd.DataFrame(value)

    if isinstance(value, dict):
        return pd.DataFrame([value])

    return pd.DataFrame({"Value": [value]})


def _dataframe_to_records(value: Any) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame-like object into JSON-safe records.
    """

    df = _ensure_dataframe(value)

    if df.empty:
        return []

    return df.fillna("").to_dict(orient="records")


def _flatten_dict(
    data: Any,
    parent_key: str = "",
    rows: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Flatten nested dictionaries/lists into a table structure.
    """

    if rows is None:
        rows = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else str(key)
            _flatten_dict(value, new_key, rows)

    elif isinstance(data, list):
        for index, value in enumerate(data):
            new_key = f"{parent_key}[{index}]"
            _flatten_dict(value, new_key, rows)

    else:
        rows.append(
            {
                "Input": parent_key,
                "Value": data
            }
        )

    return rows


def _find_nested_key(data: Any, target_keys: List[str]) -> Optional[Any]:
    """
    Search nested dict/list data for possible key names.
    """

    normalized_targets = {key.lower() for key in target_keys}

    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in normalized_targets:
                return value

            found_value = _find_nested_key(value, target_keys)

            if found_value is not None:
                return found_value

    elif isinstance(data, list):
        for item in data:
            found_value = _find_nested_key(item, target_keys)

            if found_value is not None:
                return found_value

    return None


def get_dataset_last_updated(dataset: Dict[str, Any]) -> str:
    """
    Extract dataset last updated information if available.

    If the JSON does not contain a clear last updated field, return a transparent
    fallback instead of inventing a date.
    """

    possible_keys = [
        "last_updated",
        "last_update",
        "updated_at",
        "updated",
        "dataset_last_updated",
        "data_last_updated",
        "source_last_updated",
        "last_reviewed",
        "reviewed_at"
    ]

    value = _find_nested_key(dataset, possible_keys)

    if value:
        return str(value)

    metadata = dataset.get("metadata", {}) if isinstance(dataset, dict) else {}

    if isinstance(metadata, dict):
        base_year = metadata.get("base_year")

        if base_year:
            return f"Not specified; base year {base_year}"

    return "Not specified"


def _get_final_nav_from_outputs(simulation_outputs: Dict[str, Any]) -> float:
    nav_summary = simulation_outputs.get("nav_summary", {})

    if isinstance(nav_summary, dict) and "year_10_nav" in nav_summary:
        return _safe_float(nav_summary.get("year_10_nav"))

    nav_df = _ensure_dataframe(simulation_outputs.get("nav_df"))

    if nav_df.empty:
        return 0.0

    column_candidates = [
        "Local Currency NAV",
        "NAV",
        "Net Asset Value",
        "Net Asset Value AUD",
        "Year-10 NAV AUD",
        "Final NAV AUD"
    ]

    for column in column_candidates:
        if column in nav_df.columns:
            return _safe_float(nav_df[column].iloc[-1])

    return 0.0


def _get_final_nav_lkr_from_outputs(simulation_outputs: Dict[str, Any]) -> float:
    nav_summary = simulation_outputs.get("nav_summary", {})

    if isinstance(nav_summary, dict) and "year_10_nav_lkr" in nav_summary:
        return _safe_float(nav_summary.get("year_10_nav_lkr"))

    nav_df = _ensure_dataframe(simulation_outputs.get("nav_df"))

    if nav_df.empty:
        return 0.0

    for column in ["LKR NAV", "Year-10 NAV LKR", "Final NAV LKR"]:
        if column in nav_df.columns:
            return _safe_float(nav_df[column].iloc[-1])

    final_nav_local = _get_final_nav_from_outputs(simulation_outputs)
    exchange_rate = _get_exchange_rate_from_outputs(simulation_outputs)
    return final_nav_local * exchange_rate if exchange_rate else 0.0


def _get_final_nav_present_value_lkr_from_outputs(
    simulation_outputs: Dict[str, Any]
) -> float:
    nav_summary = simulation_outputs.get("nav_summary", {})

    if isinstance(nav_summary, dict) and "year_10_nav_present_value_lkr" in nav_summary:
        return _safe_float(nav_summary.get("year_10_nav_present_value_lkr"))

    nav_df = _ensure_dataframe(simulation_outputs.get("nav_df"))

    if not nav_df.empty and "Present Value LKR NAV" in nav_df.columns:
        return _safe_float(nav_df["Present Value LKR NAV"].iloc[-1])

    return calculate_lkr_present_value(
        _get_final_nav_lkr_from_outputs(simulation_outputs)
    )


def _get_final_debt_from_outputs(simulation_outputs: Dict[str, Any]) -> float:
    nav_df = _ensure_dataframe(simulation_outputs.get("nav_df"))

    if nav_df.empty:
        return 0.0

    column_candidates = [
        "Total Debt",
        "Total Liabilities",
        "Liabilities",
        "Debt"
    ]

    for column in column_candidates:
        if column in nav_df.columns:
            return _safe_float(nav_df[column].iloc[-1])

    return 0.0


def _get_exchange_rate_from_outputs(simulation_outputs: Dict[str, Any]) -> float:
    return _safe_float(simulation_outputs.get("exchange_rate"), default=0.0)


def _format_aud(value: Any) -> str:
    return f"AUD {_safe_float(value):,.0f}"


def _format_local(value: Any, currency: str) -> str:
    return f"{currency} {_safe_float(value):,.0f}"


def _format_lkr(value: Any) -> str:
    return f"LKR {_safe_float(value):,.0f}"


def _get_dict_numeric_value(data: Dict[str, Any], key_candidates: List[str]) -> float:
    if not isinstance(data, dict):
        return 0.0

    for key in key_candidates:
        if key in data:
            return _safe_float(data.get(key))

    return 0.0


def build_scenario_inputs_dataframe(scenario_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert scenario_config into a clean table for Excel and HTML reports.
    """

    rows = _flatten_dict(scenario_config)

    if not rows:
        return pd.DataFrame(
            columns=[
                "Input",
                "Value"
            ]
        )

    return pd.DataFrame(rows)


def build_model_limitations_dataframe() -> pd.DataFrame:
    """
    Standard model limitations for the final app.
    """

    rows = [
        {
            "Limitation": "Country-level model",
            "Why it matters": "The current version uses Australia-level assumptions, not city-specific costs such as Sydney, Melbourne, Brisbane, or Perth."
        },
        {
            "Limitation": "Dataset-driven assumptions",
            "Why it matters": "Results are only as reliable as the salary, rent, tuition, tax, inflation, and exchange-rate values in the JSON dataset."
        },
        {
            "Limitation": "Simplified visa and PR timing",
            "Why it matters": "Real migration outcomes depend on occupation lists, points, policy changes, English scores, work experience, and state nomination."
        },
        {
            "Limitation": "Simplified family model",
            "Why it matters": "Marriage, spouse employment, childcare usage, and child timing are modeled as assumptions, not guaranteed real-life behavior."
        },
        {
            "Limitation": "Simplified debt model",
            "Why it matters": "Loan approval, repayment terms, compounding rules, credit limits, and refinancing are simplified for research-demo clarity."
        },
        {
            "Limitation": "No probability model yet",
            "Why it matters": "Sensitivity analysis shows risk exposure, but it does not assign real probabilities to each future outcome."
        },
        {
            "Limitation": "No behavioral uncertainty",
            "Why it matters": "The model does not fully capture spending discipline, job loss, illness, family support, or unexpected lifestyle changes."
        }
    ]

    return pd.DataFrame(rows)


def build_executive_summary_dataframe(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    scenario_summary: Dict[str, Any],
    simulation_outputs: Dict[str, Any],
    dataset_last_updated: str
) -> pd.DataFrame:
    """
    Build one clean executive summary table.

    This is useful for:
        - viva
        - supervisor review
        - final report appendix
        - quick CSV export
    """

    local_currency = str(
        simulation_outputs.get("local_currency", "LOCAL")
    ).upper()
    final_nav_local = _get_final_nav_from_outputs(simulation_outputs)
    exchange_rate = _get_exchange_rate_from_outputs(simulation_outputs)
    final_nav_lkr = _get_final_nav_lkr_from_outputs(simulation_outputs)
    final_nav_present_value_lkr = _get_final_nav_present_value_lkr_from_outputs(
        simulation_outputs
    )
    final_debt_local = _get_final_debt_from_outputs(simulation_outputs)

    comparison_result = simulation_outputs.get("comparison_result", {})
    risk_result = simulation_outputs.get("risk_result", {})
    monte_carlo_probability_summary = simulation_outputs.get(
        "monte_carlo_probability_summary",
        {}
    )
    decision_summary = simulation_outputs.get("decision_summary", {})

    best_scenario = comparison_result.get("best_scenario", {})
    worst_scenario = comparison_result.get("worst_scenario", {})

    selected_rank = comparison_result.get("selected_rank")
    total_scenarios = comparison_result.get("total_scenarios")

    if selected_rank and total_scenarios:
        selected_rank_text = f"{selected_rank} / {total_scenarios}"
    else:
        selected_rank_text = "N/A"

    if final_nav_local > 0 and final_debt_local <= 0:
        nav_status = "Good"
        nav_interpretation = "The scenario ends with positive NAV and low final debt pressure."
    elif final_nav_local > 0:
        nav_status = "Mixed"
        nav_interpretation = "The scenario ends with positive NAV, but debt/liability pressure still matters."
    else:
        nav_status = "Risky"
        nav_interpretation = "The scenario ends with weak or negative NAV and needs assumption improvement."

    rows = [
        {
            "Section": "Project",
            "Metric": "Report generated",
            "Value": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Interpretation": "Timestamp of report export."
        },
        {
            "Section": "Dataset",
            "Metric": "Dataset last updated",
            "Value": dataset_last_updated,
            "Interpretation": "Used to show data freshness in submission and viva."
        },
        {
            "Section": "Scenario",
            "Metric": "Selected scenario",
            "Value": scenario_summary.get("Scenario", scenario_summary.get("scenario", "Selected scenario")),
            "Interpretation": "The currently simulated migration and life scenario."
        },
        {
            "Section": "Financial result",
            "Metric": f"Final Year-10 NAV ({local_currency})",
            "Value": _format_local(final_nav_local, local_currency),
            "Interpretation": nav_interpretation
        },
        {
            "Section": "Financial result",
            "Metric": "Final NAV LKR",
            "Value": _format_lkr(final_nav_lkr),
            "Interpretation": "Converted using the dataset exchange-rate assumption."
        },
        {
            "Section": "Financial result",
            "Metric": "Today's value of final NAV LKR",
            "Value": _format_lkr(final_nav_present_value_lkr),
            "Interpretation": "Final LKR NAV discounted using 5.5% Sri Lanka inflation for 10 years."
        },
        {
            "Section": "Financial result",
            "Metric": f"Final debt/liabilities ({local_currency})",
            "Value": _format_local(final_debt_local, local_currency),
            "Interpretation": "Higher final debt reduces the quality of the scenario."
        },
        {
            "Section": "Financial result",
            "Metric": "NAV health",
            "Value": nav_status,
            "Interpretation": nav_interpretation
        },
        {
            "Section": "Scenario comparison",
            "Metric": "Best scenario",
            "Value": best_scenario.get("Scenario", "N/A"),
            "Interpretation": _format_local(
                _get_dict_numeric_value(
                    best_scenario,
                    ["Year-10 NAV Local", "Final NAV Local", "Final NAV"]
                ),
                local_currency
            )
        },
        {
            "Section": "Scenario comparison",
            "Metric": "Worst scenario",
            "Value": worst_scenario.get("Scenario", "N/A"),
            "Interpretation": _format_local(
                _get_dict_numeric_value(
                    worst_scenario,
                    ["Year-10 NAV Local", "Final NAV Local", "Final NAV"]
                ),
                local_currency
            )
        },
        {
            "Section": "Scenario comparison",
            "Metric": "Selected scenario rank",
            "Value": selected_rank_text,
            "Interpretation": comparison_result.get("message", "")
        },
        {
            "Section": "Scenario comparison",
            "Metric": "NAV gap from best",
            "Value": _format_local(
                comparison_result.get("nav_gap_local", 0.0),
                local_currency
            ),
            "Interpretation": "Opportunity cost of choosing the selected scenario instead of the best scenario."
        },
        {
            "Section": "Scenario comparison",
            "Metric": "Present-value NAV gap from best",
            "Value": _format_lkr(
                comparison_result.get("nav_gap_present_value_lkr", 0.0)
            ),
            "Interpretation": "Same gap discounted into today's LKR value."
        },
        {
            "Section": "Risk",
            "Metric": "Risk level",
            "Value": risk_result.get("risk_level", "N/A"),
            "Interpretation": risk_result.get("risk_summary_text", "")
        },
        {
            "Section": "Risk",
            "Metric": "Most sensitive variable",
            "Value": risk_result.get("most_sensitive_variable", {}).get("variable", "N/A"),
            "Interpretation": "This variable has the highest downside/upside effect in the sensitivity model."
        }
    ]

    if monte_carlo_probability_summary:
        rows.extend(
            [
                {
                    "Section": "Monte Carlo risk",
                    "Metric": "Simulation count",
                    "Value": monte_carlo_probability_summary.get(
                        "simulation_count",
                        "N/A"
                    ),
                    "Interpretation": "Number of randomized future cases tested."
                },
                {
                    "Section": "Monte Carlo risk",
                    "Metric": "Probability final NAV is positive",
                    "Value": _format_percent(
                        monte_carlo_probability_summary.get(
                            "positive_nav_probability",
                            0.0
                        )
                    ),
                    "Interpretation": "Share of randomized cases ending with positive NAV."
                },
                {
                    "Section": "Monte Carlo risk",
                    "Metric": "Probability high debt",
                    "Value": _format_percent(
                        monte_carlo_probability_summary.get(
                            "high_debt_probability",
                            0.0
                        )
                    ),
                    "Interpretation": "Share of randomized cases where debt pressure crosses the model threshold."
                },
                {
                    "Section": "Monte Carlo risk",
                    "Metric": "Probability very bad outcome",
                    "Value": _format_percent(
                        monte_carlo_probability_summary.get(
                            "very_bad_outcome_probability",
                            0.0
                        )
                    ),
                    "Interpretation": "Share of randomized cases with negative final NAV and high debt."
                },
                {
                    "Section": "Monte Carlo risk",
                    "Metric": f"Median final NAV ({local_currency})",
                    "Value": _format_local(
                        monte_carlo_probability_summary.get(
                            "median_final_nav_local",
                            0.0
                        ),
                        local_currency
                    ),
                    "Interpretation": "Middle final NAV across all randomized cases."
                },
                {
                    "Section": "Monte Carlo risk",
                    "Metric": f"5th percentile final NAV ({local_currency})",
                    "Value": _format_local(
                        monte_carlo_probability_summary.get(
                            "p5_final_nav_local",
                            0.0
                        ),
                        local_currency
                    ),
                    "Interpretation": "Downside case where only 5% of simulations are worse."
                },
                {
                    "Section": "Monte Carlo risk",
                    "Metric": f"95th percentile final NAV ({local_currency})",
                    "Value": _format_local(
                        monte_carlo_probability_summary.get(
                            "p95_final_nav_local",
                            0.0
                        ),
                        local_currency
                    ),
                    "Interpretation": "Upside case where only 5% of simulations are better."
                }
            ]
        )

    rows.extend(
        [
        {
            "Section": "Decision summary",
            "Metric": "Break-even year",
            "Value": decision_summary.get("break_even_year", "N/A"),
            "Interpretation": "First year where NAV becomes positive, if available."
        },
        {
            "Section": "Decision summary",
            "Metric": "Main expense category",
            "Value": decision_summary.get("main_expense_category", "N/A"),
            "Interpretation": "Largest cost driver in the selected scenario, if available."
        }
        ]
    )

    return pd.DataFrame(rows)


def _write_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    dataframe: pd.DataFrame
) -> None:
    """
    Write a DataFrame to Excel with simple formatting.
    """

    safe_sheet_name = sheet_name[:31]
    df = _ensure_dataframe(dataframe)

    if df.empty:
        df = pd.DataFrame({"Message": ["No data available"]})

    df.to_excel(
        writer,
        index=False,
        sheet_name=safe_sheet_name
    )

    worksheet = writer.sheets[safe_sheet_name]

    try:
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                try:
                    cell_value_length = len(str(cell.value))
                    max_length = max(max_length, cell_value_length)
                except Exception:
                    pass

            adjusted_width = min(max(max_length + 2, 12), 45)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    except Exception:
        pass


def build_full_simulation_excel(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    scenario_summary: Dict[str, Any],
    simulation_outputs: Dict[str, Any],
    assumption_df: Optional[pd.DataFrame] = None,
    dataset_last_updated: str = "Not specified"
) -> bytes:
    """
    Build one full Excel report with multiple sheets.

    Required sheets:
        - Summary
        - Scenario Inputs
        - Income
        - Expenses
        - NAV
        - Scenario Comparison
        - Sensitivity
        - Testing
        - Assumptions
    """

    output = io.BytesIO()

    summary_df = build_executive_summary_dataframe(
        dataset=dataset,
        scenario_config=scenario_config,
        scenario_summary=scenario_summary,
        simulation_outputs=simulation_outputs,
        dataset_last_updated=dataset_last_updated
    )

    scenario_inputs_df = build_scenario_inputs_dataframe(scenario_config)

    income_df = _ensure_dataframe(simulation_outputs.get("income_df"))
    expense_df = _ensure_dataframe(simulation_outputs.get("expense_df"))
    nav_df = _ensure_dataframe(simulation_outputs.get("nav_df"))
    comparison_df = _ensure_dataframe(simulation_outputs.get("comparison_df"))
    sensitivity_df = _ensure_dataframe(simulation_outputs.get("sensitivity_df"))
    monte_carlo_summary_df = _ensure_dataframe(
        simulation_outputs.get("monte_carlo_summary_df")
    )
    monte_carlo_percentiles_df = _ensure_dataframe(
        simulation_outputs.get("monte_carlo_percentiles_df")
    )
    monte_carlo_results_df = _ensure_dataframe(
        simulation_outputs.get("monte_carlo_results_df")
    )
    testing_df = _ensure_dataframe(simulation_outputs.get("testing_df"))
    assumptions_export_df = _ensure_dataframe(assumption_df)
    limitations_df = build_model_limitations_dataframe()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write_sheet(writer, "Summary", summary_df)
        _write_sheet(writer, "Scenario Inputs", scenario_inputs_df)
        _write_sheet(writer, "Income", income_df)
        _write_sheet(writer, "Expenses", expense_df)
        _write_sheet(writer, "NAV", nav_df)
        _write_sheet(writer, "Scenario Comparison", comparison_df)
        _write_sheet(writer, "Sensitivity", sensitivity_df)
        _write_sheet(writer, "Monte Carlo Summary", monte_carlo_summary_df)
        _write_sheet(writer, "Monte Carlo Percentiles", monte_carlo_percentiles_df)
        _write_sheet(writer, "Monte Carlo Results", monte_carlo_results_df)
        _write_sheet(writer, "Testing", testing_df)
        _write_sheet(writer, "Assumptions", assumptions_export_df)
        _write_sheet(writer, "Model Limitations", limitations_df)

    return output.getvalue()


def build_scenario_report_json(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    scenario_summary: Dict[str, Any],
    simulation_outputs: Dict[str, Any],
    dataset_last_updated: str
) -> str:
    """
    Build a clean JSON report for the selected scenario.
    """

    payload = {
        "project": "Australia 10-Year NAV Simulator",
        "report_generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_last_updated": dataset_last_updated,
        "scenario_summary": scenario_summary,
        "scenario_config": scenario_config,
        "executive_summary": _dataframe_to_records(
            build_executive_summary_dataframe(
                dataset=dataset,
                scenario_config=scenario_config,
                scenario_summary=scenario_summary,
                simulation_outputs=simulation_outputs,
                dataset_last_updated=dataset_last_updated
            )
        ),
        "income": _dataframe_to_records(simulation_outputs.get("income_df")),
        "expenses": _dataframe_to_records(simulation_outputs.get("expense_df")),
        "nav": _dataframe_to_records(simulation_outputs.get("nav_df")),
        "scenario_comparison": _dataframe_to_records(simulation_outputs.get("comparison_df")),
        "sensitivity": _dataframe_to_records(simulation_outputs.get("sensitivity_df")),
        "monte_carlo": {
            "probability_summary": simulation_outputs.get(
                "monte_carlo_probability_summary",
                {}
            ),
            "summary": _dataframe_to_records(
                simulation_outputs.get("monte_carlo_summary_df")
            ),
            "percentiles": _dataframe_to_records(
                simulation_outputs.get("monte_carlo_percentiles_df")
            ),
            "results": _dataframe_to_records(
                simulation_outputs.get("monte_carlo_results_df")
            )
        },
        "testing": _dataframe_to_records(simulation_outputs.get("testing_df")),
        "risk_result": simulation_outputs.get("risk_result", {}),
        "comparison_result": simulation_outputs.get("comparison_result", {}),
        "decision_summary": simulation_outputs.get("decision_summary", {}),
        "model_limitations": _dataframe_to_records(build_model_limitations_dataframe())
    }

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=_json_default
    )


def _html_table(title: str, dataframe: Any) -> str:
    df = _ensure_dataframe(dataframe)

    if df.empty:
        table_html = "<p>No data available.</p>"
    else:
        table_html = df.fillna("").to_html(
            index=False,
            escape=True,
            border=0,
            classes="report-table"
        )

    return f"""
    <section class="report-section">
        <h2>{html.escape(title)}</h2>
        {table_html}
    </section>
    """


def build_simple_html_report(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    scenario_summary: Dict[str, Any],
    simulation_outputs: Dict[str, Any],
    assumption_df: Optional[pd.DataFrame] = None,
    dataset_last_updated: str = "Not specified"
) -> str:
    """
    Build a simple standalone HTML report.

    This is intentionally easier than PDF and works well for sharing.
    """

    summary_df = build_executive_summary_dataframe(
        dataset=dataset,
        scenario_config=scenario_config,
        scenario_summary=scenario_summary,
        simulation_outputs=simulation_outputs,
        dataset_last_updated=dataset_last_updated
    )

    scenario_inputs_df = build_scenario_inputs_dataframe(scenario_config)
    limitations_df = build_model_limitations_dataframe()

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Australia 10-Year NAV Simulator Report</title>",
        """
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 32px;
                color: #222;
                background: #fafafa;
            }

            .report-header {
                padding: 24px;
                border-radius: 16px;
                background: #ffffff;
                border: 1px solid #e6e6e6;
                margin-bottom: 24px;
            }

            .report-header h1 {
                margin: 0 0 8px 0;
            }

            .report-meta {
                color: #666;
                font-size: 14px;
            }

            .report-section {
                background: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 24px;
            }

            .report-section h2 {
                margin-top: 0;
            }

            .report-table {
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
            }

            .report-table th {
                text-align: left;
                background: #f2f2f2;
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }

            .report-table td {
                padding: 8px;
                border-bottom: 1px solid #eee;
                vertical-align: top;
            }

            .footer {
                color: #777;
                font-size: 12px;
                margin-top: 32px;
            }
        </style>
        """,
        "</head>",
        "<body>",
        """
        <div class="report-header">
            <h1>Australia 10-Year NAV Simulator</h1>
            <div class="report-meta">
        """,
        f"Report generated: {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M'))}<br>",
        f"Dataset last updated: {html.escape(dataset_last_updated)}",
        """
            </div>
        </div>
        """,
        _html_table("Executive Summary", summary_df),
        _html_table("Scenario Inputs", scenario_inputs_df),
        _html_table("Income", simulation_outputs.get("income_df")),
        _html_table("Expenses", simulation_outputs.get("expense_df")),
        _html_table("NAV", simulation_outputs.get("nav_df")),
        _html_table("Scenario Comparison", simulation_outputs.get("comparison_df")),
        _html_table("Sensitivity", simulation_outputs.get("sensitivity_df")),
        _html_table("Monte Carlo Summary", simulation_outputs.get("monte_carlo_summary_df")),
        _html_table("Monte Carlo Percentiles", simulation_outputs.get("monte_carlo_percentiles_df")),
        _html_table("Monte Carlo Results", simulation_outputs.get("monte_carlo_results_df")),
        _html_table("Testing", simulation_outputs.get("testing_df")),
        _html_table("Assumptions", assumption_df),
        _html_table("Model Limitations", limitations_df),
        """
        <div class="footer">
            Australia 10-Year NAV Simulator | Research demo report
        </div>
        """,
        "</body>",
        "</html>"
    ]

    return "\n".join(html_parts)
