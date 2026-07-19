from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ai_recommender import DEFAULT_OPENAI_MODEL


AI_DECISION_EXPLAINER_INSTRUCTIONS = """
You explain migration and financial simulation results in simple English.

Hard rules:
- Use only the JSON data provided by the app.
- Do not invent financial values, immigration rules, visa promises, job outcomes, or country facts.
- Do not recalculate the model.
- If the data is missing, say it is not available.
- Explain the result as a simulation-based insight, not guaranteed financial or immigration advice.
- Keep the answer short, practical, and easy for a non-technical user to understand.
"""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


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

    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(nested_value)
            for key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]

    return value


def _compact_dict(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {
        key: _json_safe_value(data.get(key))
        for key in keys
        if key in data
    }


def _compact_dataframe(
    df: Optional[pd.DataFrame],
    preferred_columns: List[str],
    max_rows: int = 8
) -> List[Dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []

    available_columns = [
        column_name
        for column_name in preferred_columns
        if column_name in df.columns
    ]

    compact_df = df[available_columns].copy() if available_columns else df.copy()

    if "Country Rank" in compact_df.columns:
        compact_df = compact_df.sort_values(by="Country Rank", na_position="last")
    elif "Rank" in compact_df.columns:
        compact_df = compact_df.sort_values(by="Rank", na_position="last")

    return [
        _json_safe_value(record)
        for record in compact_df.head(max_rows).to_dict("records")
    ]


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


def build_decision_explanation_payload(
    simulation_outputs: Dict[str, Any],
    scenario_config: Dict[str, Any],
    selected_country: str
) -> Dict[str, Any]:
    """
    Build a compact JSON summary for answering post-simulation questions.
    """

    decision_summary = simulation_outputs.get("decision_summary", {}) or {}
    nav_summary = simulation_outputs.get("nav_summary", {}) or {}
    selected_labels = scenario_config.get("selected_labels", {}) or {}

    return {
        "selected_country": selected_country,
        "selected_migration_path": selected_labels.get("migration_path"),
        "selected_life_scenario": selected_labels.get("life_scenario"),
        "local_currency": simulation_outputs.get("local_currency"),
        "final_nav": _json_safe_value(
            decision_summary.get(
                "final_year_10_nav_local",
                nav_summary.get("year_10_nav")
            )
        ),
        "final_nav_lkr": _json_safe_value(
            decision_summary.get(
                "final_year_10_nav_lkr",
                nav_summary.get("year_10_nav_lkr")
            )
        ),
        "present_value_nav_lkr": _json_safe_value(
            decision_summary.get(
                "final_year_10_nav_present_value_lkr",
                nav_summary.get("year_10_nav_present_value_lkr")
            )
        ),
        "break_even_year": _json_safe_value(
            decision_summary.get("break_even_year")
        ),
        "highest_debt": {
            "year": _json_safe_value(decision_summary.get("highest_debt_year")),
            "amount": _json_safe_value(decision_summary.get("highest_debt_amount"))
        },
        "biggest_expense": {
            "category": _json_safe_value(
                decision_summary.get("main_expense_category")
            ),
            "amount": _json_safe_value(decision_summary.get("main_expense_amount"))
        },
        "biggest_risk_variable": _json_safe_value(
            decision_summary.get("main_risk_variable")
        ),
        "decision_sentence": _json_safe_value(
            decision_summary.get("decision_sentence")
        ),
        "scenario_comparison_result": _json_safe_value(
            simulation_outputs.get("comparison_result", {})
        ),
        "risk_result": _json_safe_value(simulation_outputs.get("risk_result", {})),
        "country_comparison": _compact_dataframe(
            simulation_outputs.get("country_comparison_df"),
            [
                "Country Rank",
                "Country",
                "Currency",
                "Final NAV Local",
                "Final NAV LKR",
                "Final NAV Present Value LKR",
                "Break-even Year",
                "Highest Debt Local",
                "Highest Debt LKR",
                "Risk Score",
                "Status",
                "Error"
            ],
            max_rows=10
        ),
        "scenario_comparison_table": _compact_dataframe(
            simulation_outputs.get("comparison_df"),
            [
                "Rank",
                "Scenario",
                "Migration Path",
                "Life Scenario",
                "Final NAV Local",
                "Final NAV LKR",
                "Final NAV Present Value LKR",
                "Year-10 NAV Local",
                "Year-10 NAV LKR",
                "Year-10 NAV Present Value LKR"
            ],
            max_rows=10
        ),
        "decision_summary": _compact_dict(
            decision_summary,
            [
                "final_year_10_nav_local",
                "final_year_10_nav_lkr",
                "final_year_10_nav_present_value_lkr",
                "break_even_year",
                "highest_debt_year",
                "highest_debt_amount",
                "lowest_nav_year",
                "lowest_nav_amount",
                "main_expense_category",
                "main_expense_amount",
                "main_risk_variable",
                "best_scenario_name",
                "best_scenario_final_nav",
                "best_scenario_gap",
                "selected_vs_best_difference"
            ]
        )
    }


def build_fallback_decision_answer(
    payload: Dict[str, Any],
    question: str
) -> str:
    """
    Create a deterministic plain-English answer when OpenAI is unavailable.
    """

    question_text = str(question).strip()
    selected_country = payload.get("selected_country") or "the selected country"
    migration_path = payload.get("selected_migration_path") or "the selected path"
    final_nav = _safe_float(payload.get("final_nav"))
    final_nav_lkr = _safe_float(payload.get("final_nav_lkr"))
    present_value_nav_lkr = _safe_float(payload.get("present_value_nav_lkr"))
    break_even_year = payload.get("break_even_year")
    biggest_expense = payload.get("biggest_expense", {}) or {}
    highest_debt = payload.get("highest_debt", {}) or {}
    risk_result = payload.get("risk_result", {}) or {}
    comparison_result = payload.get("scenario_comparison_result", {}) or {}

    break_even_text = (
        f"Year {int(break_even_year)}"
        if break_even_year not in [None, "", "No break-even"]
        else "not reached within 10 years"
    )

    lines = [
        f"Based on the simulation, {selected_country} with {migration_path} ends with final NAV {final_nav:,.0f} in local currency.",
        f"In LKR, that is {final_nav_lkr:,.0f}; today's value is {present_value_nav_lkr:,.0f} LKR.",
        f"Break-even is {break_even_text}.",
        (
            f"The biggest expense is {biggest_expense.get('category', 'not available')} "
            f"at { _safe_float(biggest_expense.get('amount')):,.0f} local currency."
        ),
        (
            f"The biggest risk variable is "
            f"{risk_result.get('most_sensitive_variable', {}).get('variable') or payload.get('biggest_risk_variable') or 'not available'}."
        ),
        (
            f"Peak debt is { _safe_float(highest_debt.get('amount')):,.0f} "
            f"in Year {highest_debt.get('year') or 'not available'}."
        )
    ]

    comparison_message = comparison_result.get("message")
    if comparison_message:
        lines.append(str(comparison_message))

    if question_text:
        lines.append(
            "This is a basic fallback answer because AI explanation is not available."
        )

    return "\n\n".join(lines)


def generate_decision_explanation(
    payload: Dict[str, Any],
    user_question: str,
    api_key: str,
    fallback_text: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ask OpenAI to explain already-calculated simulation results.
    """

    if not api_key:
        return {
            "text": fallback_text,
            "used_ai": False,
            "error": ""
        }

    try:
        from openai import OpenAI

        request_payload = {
            "user_question": str(user_question).strip(),
            "simulation_result": payload
        }

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model or os.environ.get(
                "OPENAI_DECISION_EXPLAINER_MODEL",
                DEFAULT_OPENAI_MODEL
            ),
            instructions=AI_DECISION_EXPLAINER_INSTRUCTIONS.strip(),
            input=json.dumps(request_payload, separators=(",", ":")),
            max_output_tokens=650,
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
