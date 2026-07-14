from typing import Any, Dict, Optional

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_column(df: pd.DataFrame, possible_columns: list[str]) -> Optional[str]:
    for column_name in possible_columns:
        if column_name in df.columns:
            return column_name

    return None


def get_most_sensitive_variable(tornado_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return the variable with the largest sensitivity impact.
    """

    if tornado_df.empty:
        return {
            "variable": None,
            "max_impact_local": 0.0,
            "max_impact_aud": 0.0,
            "max_impact_lkr": 0.0,
            "max_impact_present_value_lkr": 0.0,
            "max_impact_percent": 0.0
        }

    impact_column = _find_column(
        tornado_df,
        ["Max Impact %", "Max Impact Local", "Max Impact LKR"]
    )

    if impact_column is None:
        return {
            "variable": None,
            "max_impact_local": 0.0,
            "max_impact_aud": 0.0,
            "max_impact_lkr": 0.0,
            "max_impact_present_value_lkr": 0.0,
            "max_impact_percent": 0.0
        }

    working_df = tornado_df.copy()
    working_df[impact_column] = working_df[impact_column].abs()
    row = working_df.sort_values(by=impact_column, ascending=False).iloc[0]

    return {
        "variable": row.get("Variable"),
        "max_impact_local": _safe_float(row.get("Max Impact Local")),
        "max_impact_aud": _safe_float(row.get("Max Impact AUD")),
        "max_impact_lkr": _safe_float(row.get("Max Impact LKR")),
        "max_impact_present_value_lkr": _safe_float(
            row.get("Max Impact Present Value LKR")
        ),
        "max_impact_percent": _safe_float(row.get("Max Impact %"))
    }


def _get_sensitivity_nav_column(sensitivity_df: pd.DataFrame) -> Optional[str]:
    return _find_column(
        sensitivity_df,
        [
            "Year-10 NAV Local",
            "Final NAV Local",
            "Year-10 NAV AUD",
            "Final NAV AUD",
            "NAV"
        ]
    )


def _empty_nav_case() -> Dict[str, Any]:
    return {
        "variable": None,
        "change_label": None,
        "currency": "LOCAL",
        "nav_local": 0.0,
        "nav_aud": 0.0,
        "nav_lkr": 0.0,
        "nav_present_value_lkr": 0.0,
        "delta_local": 0.0,
        "delta_aud": 0.0,
        "delta_lkr": 0.0,
        "delta_present_value_lkr": 0.0
    }


def get_best_case_nav(sensitivity_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return the sensitivity test case with the highest Year-10 NAV.
    """

    if sensitivity_df.empty:
        return _empty_nav_case()

    nav_column = _get_sensitivity_nav_column(sensitivity_df)

    if nav_column is None:
        return _empty_nav_case()

    row = sensitivity_df.sort_values(by=nav_column, ascending=False).iloc[0]

    return {
        "variable": row.get("Variable"),
        "change_label": row.get("Change Label"),
        "currency": row.get("Local Currency", "LOCAL"),
        "nav_local": _safe_float(row.get(nav_column)),
        "nav_aud": _safe_float(row.get("Year-10 NAV AUD", row.get(nav_column))),
        "nav_lkr": _safe_float(row.get("Year-10 NAV LKR")),
        "nav_present_value_lkr": _safe_float(
            row.get("Year-10 NAV Present Value LKR")
        ),
        "delta_local": _safe_float(row.get("Delta NAV Local")),
        "delta_aud": _safe_float(row.get("Delta NAV AUD", row.get("Delta NAV Local"))),
        "delta_lkr": _safe_float(row.get("Delta NAV LKR")),
        "delta_present_value_lkr": _safe_float(
            row.get("Delta NAV Present Value LKR")
        )
    }


def get_worst_case_nav(sensitivity_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return the sensitivity test case with the lowest Year-10 NAV.
    """

    if sensitivity_df.empty:
        return _empty_nav_case()

    nav_column = _get_sensitivity_nav_column(sensitivity_df)

    if nav_column is None:
        return _empty_nav_case()

    row = sensitivity_df.sort_values(by=nav_column, ascending=True).iloc[0]

    return {
        "variable": row.get("Variable"),
        "change_label": row.get("Change Label"),
        "currency": row.get("Local Currency", "LOCAL"),
        "nav_local": _safe_float(row.get(nav_column)),
        "nav_aud": _safe_float(row.get("Year-10 NAV AUD", row.get(nav_column))),
        "nav_lkr": _safe_float(row.get("Year-10 NAV LKR")),
        "nav_present_value_lkr": _safe_float(
            row.get("Year-10 NAV Present Value LKR")
        ),
        "delta_local": _safe_float(row.get("Delta NAV Local")),
        "delta_aud": _safe_float(row.get("Delta NAV AUD", row.get("Delta NAV Local"))),
        "delta_lkr": _safe_float(row.get("Delta NAV LKR")),
        "delta_present_value_lkr": _safe_float(
            row.get("Delta NAV Present Value LKR")
        )
    }


def get_risk_level(
    sensitivity_df: pd.DataFrame,
    tornado_df: pd.DataFrame
) -> str:
    """
    Convert sensitivity downside into a simple risk label.

    High risk:
    - Worst-case NAV becomes negative, or
    - Downside is at least 50% of base NAV.

    Medium risk:
    - Downside is at least 20% of base NAV.

    Low risk:
    - Downside is below 20% of base NAV.
    """

    if sensitivity_df.empty:
        return "Unknown"

    base_rows = (
        sensitivity_df[sensitivity_df["Change"] == 0]
        if "Change" in sensitivity_df.columns
        else pd.DataFrame()
    )

    if base_rows.empty:
        base_nav = _safe_float(
            sensitivity_df.iloc[0].get(
                "Base NAV Local",
                sensitivity_df.iloc[0].get("Base NAV AUD")
            )
        )
    else:
        base_nav = _safe_float(
            base_rows.iloc[0].get(
                "Base NAV Local",
                base_rows.iloc[0].get("Base NAV AUD")
            )
        )

    worst_case = get_worst_case_nav(sensitivity_df)
    worst_nav = worst_case["nav_local"]
    downside = max(0.0, base_nav - worst_nav)

    if worst_nav < 0:
        return "High"

    if base_nav == 0:
        most_sensitive = get_most_sensitive_variable(tornado_df)
        max_impact_percent = abs(most_sensitive["max_impact_percent"])
    else:
        max_impact_percent = downside / abs(base_nav)

    if max_impact_percent >= 0.50:
        return "High"

    if max_impact_percent >= 0.20:
        return "Medium"

    return "Low"


def get_risk_summary_text(
    comparison_result: Dict[str, Any],
    most_sensitive_variable: Dict[str, Any],
    best_case_nav: Dict[str, Any],
    worst_case_nav: Dict[str, Any],
    risk_level: str
) -> str:
    """
    Build plain-English interpretation for the selected scenario.
    """

    comparison_message = comparison_result.get(
        "message",
        "Scenario ranking is not available."
    )

    variable_name = most_sensitive_variable.get("variable") or "unknown"

    best_variable = best_case_nav.get("variable") or "unknown"
    best_change = best_case_nav.get("change_label") or "unknown change"

    worst_variable = worst_case_nav.get("variable") or "unknown"
    worst_change = worst_case_nav.get("change_label") or "unknown change"

    return (
        f"{comparison_message} "
        f"The biggest risk variable is {variable_name}. "
        f"Best-case NAV is {best_case_nav.get('currency', 'LOCAL')} {best_case_nav.get('nav_local', 0.0):,.0f} "
        f"when {best_variable} is {best_change}. "
        f"Worst-case NAV is {worst_case_nav.get('currency', 'LOCAL')} {worst_case_nav.get('nav_local', 0.0):,.0f} "
        f"when {worst_variable} is {worst_change}. "
        f"Overall risk level: {risk_level}."
    )
