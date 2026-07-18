import copy
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.currency_utils import (
    calculate_lkr_present_value,
    get_country_currency,
    get_exchange_rate_to_lkr
)
from src.expense_model import calculate_yearly_expenses
from src.income_model import calculate_yearly_income
from src.nav_model import calculate_nav_simulation, get_nav_summary


HIGH_DEBT_TO_INCOME_THRESHOLD = 0.50


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _sample_rate(
    rng: np.random.Generator,
    base_value: float,
    minimum: float,
    maximum: float,
    relative_width: float,
    minimum_width: float
) -> float:
    spread = max(abs(base_value) * relative_width, minimum_width)
    low = max(minimum, base_value - spread)
    high = min(maximum, base_value + spread)
    mode = _clamp(base_value, low, high)

    if low == high:
        return low

    return float(rng.triangular(low, mode, high))


def _sample_multiplier(
    rng: np.random.Generator,
    base_value: float,
    low_factor: float,
    high_factor: float,
    minimum: float,
    maximum: float
) -> float:
    low = max(minimum, base_value * low_factor)
    high = min(maximum, base_value * high_factor)
    mode = _clamp(base_value, low, high)

    if low == high:
        return low

    return float(rng.triangular(low, mode, high))


def _sample_probability(
    rng: np.random.Generator,
    base_value: float,
    relative_width: float,
    minimum_width: float
) -> float:
    spread = max(abs(base_value) * relative_width, minimum_width)
    low = max(0.0, base_value - spread)
    high = min(1.0, base_value + spread)
    mode = _clamp(base_value, low, high)

    if low == high:
        return low

    return float(rng.triangular(low, mode, high))


def _sample_scenario_config(
    rng: np.random.Generator,
    base_scenario_config: Dict[str, Any]
) -> Dict[str, Any]:
    scenario_config = copy.deepcopy(base_scenario_config)
    adjustable_inputs = scenario_config["adjustable_inputs"]

    salary_growth_rate = _safe_float(adjustable_inputs.get("salary_growth_rate"))
    inflation_rate = _safe_float(adjustable_inputs.get("inflation_rate"))
    investment_return_rate = _safe_float(
        adjustable_inputs.get("investment_return_rate")
    )
    rent_multiplier = _safe_float(adjustable_inputs.get("rent_multiplier"), 1.0)
    tuition_multiplier = _safe_float(adjustable_inputs.get("tuition_multiplier"), 1.0)
    childcare_multiplier = _safe_float(
        adjustable_inputs.get("childcare_multiplier"),
        1.0
    )

    spouse_settings = scenario_config.get("spouse_income_settings", {})
    spouse_income_percentage = _safe_float(
        spouse_settings.get("income_percentage"),
        0.0
    )

    sampled_values = {
        "salary_growth_rate": _sample_rate(
            rng=rng,
            base_value=salary_growth_rate,
            minimum=0.0,
            maximum=0.15,
            relative_width=0.50,
            minimum_width=0.02
        ),
        "inflation_rate": _sample_rate(
            rng=rng,
            base_value=inflation_rate,
            minimum=0.0,
            maximum=0.15,
            relative_width=0.50,
            minimum_width=0.02
        ),
        "investment_return_rate": _sample_rate(
            rng=rng,
            base_value=investment_return_rate,
            minimum=-0.10,
            maximum=0.25,
            relative_width=0.75,
            minimum_width=0.04
        ),
        "rent_multiplier": _sample_multiplier(
            rng=rng,
            base_value=rent_multiplier,
            low_factor=0.80,
            high_factor=1.30,
            minimum=0.25,
            maximum=3.00
        ),
        "tuition_multiplier": _sample_multiplier(
            rng=rng,
            base_value=tuition_multiplier,
            low_factor=0.75,
            high_factor=1.35,
            minimum=0.25,
            maximum=3.00
        ),
        "childcare_multiplier": _sample_multiplier(
            rng=rng,
            base_value=childcare_multiplier,
            low_factor=0.75,
            high_factor=1.35,
            minimum=0.25,
            maximum=3.00
        ),
        "spouse_income_percentage": _sample_probability(
            rng=rng,
            base_value=spouse_income_percentage,
            relative_width=0.35,
            minimum_width=0.10
        ),
        "exchange_rate_multiplier": _sample_multiplier(
            rng=rng,
            base_value=1.0,
            low_factor=0.85,
            high_factor=1.20,
            minimum=0.50,
            maximum=1.75
        )
    }

    adjustable_inputs["salary_growth_rate"] = sampled_values["salary_growth_rate"]
    adjustable_inputs["inflation_rate"] = sampled_values["inflation_rate"]
    adjustable_inputs["investment_return_rate"] = sampled_values[
        "investment_return_rate"
    ]
    adjustable_inputs["rent_multiplier"] = sampled_values["rent_multiplier"]
    adjustable_inputs["tuition_multiplier"] = sampled_values["tuition_multiplier"]
    adjustable_inputs["childcare_multiplier"] = sampled_values[
        "childcare_multiplier"
    ]

    spouse_settings["income_percentage"] = sampled_values[
        "spouse_income_percentage"
    ]
    scenario_config["spouse_income_settings"] = spouse_settings

    return {
        "scenario_config": scenario_config,
        "sampled_values": sampled_values
    }


def _get_final_value(
    df: pd.DataFrame,
    column_candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or df.empty:
        return default

    for column in column_candidates:
        if column in df.columns:
            return _safe_float(df[column].iloc[-1], default)

    return default


def _get_max_value(
    df: pd.DataFrame,
    column_candidates: List[str],
    default: float = 0.0
) -> float:
    if df is None or df.empty:
        return default

    for column in column_candidates:
        if column in df.columns:
            return float(
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(default).max()
            )

    return default


def _build_probability_summary(
    results_df: pd.DataFrame,
    local_currency: str
) -> Dict[str, Any]:
    simulation_count = int(len(results_df))

    if simulation_count == 0:
        return {
            "simulation_count": 0,
            "currency": local_currency,
            "positive_nav_probability": 0.0,
            "high_debt_probability": 0.0,
            "very_bad_outcome_probability": 0.0,
            "mean_final_nav_local": 0.0,
            "median_final_nav_local": 0.0,
            "p5_final_nav_local": 0.0,
            "p95_final_nav_local": 0.0,
            "worst_final_nav_local": 0.0,
            "best_final_nav_local": 0.0
        }

    final_nav = pd.to_numeric(
        results_df["Final NAV Local"],
        errors="coerce"
    ).fillna(0.0)

    return {
        "simulation_count": simulation_count,
        "currency": local_currency,
        "positive_nav_probability": float(results_df["Positive NAV"].mean()),
        "high_debt_probability": float(results_df["High Debt"].mean()),
        "very_bad_outcome_probability": float(
            results_df["Very Bad Outcome"].mean()
        ),
        "mean_final_nav_local": float(final_nav.mean()),
        "median_final_nav_local": float(final_nav.median()),
        "p5_final_nav_local": float(final_nav.quantile(0.05)),
        "p95_final_nav_local": float(final_nav.quantile(0.95)),
        "worst_final_nav_local": float(final_nav.min()),
        "best_final_nav_local": float(final_nav.max())
    }


def _build_summary_dataframe(
    probability_summary: Dict[str, Any],
    local_currency: str
) -> pd.DataFrame:
    rows = [
        {
            "Metric": "Simulation count",
            "Value": int(probability_summary["simulation_count"]),
            "Interpretation": "Number of random future cases tested."
        },
        {
            "Metric": "Probability final NAV is positive",
            "Value": probability_summary["positive_nav_probability"],
            "Interpretation": "Share of simulations where final local-currency NAV is above zero."
        },
        {
            "Metric": "Probability high debt",
            "Value": probability_summary["high_debt_probability"],
            "Interpretation": "Share of simulations where peak debt-to-income ratio is at least 50%."
        },
        {
            "Metric": "Probability very bad outcome",
            "Value": probability_summary["very_bad_outcome_probability"],
            "Interpretation": "Share of simulations with negative final NAV and high debt."
        },
        {
            "Metric": f"Mean final NAV ({local_currency})",
            "Value": probability_summary["mean_final_nav_local"],
            "Interpretation": "Average final NAV across all random cases."
        },
        {
            "Metric": f"Median final NAV ({local_currency})",
            "Value": probability_summary["median_final_nav_local"],
            "Interpretation": "Middle outcome across all random cases."
        },
        {
            "Metric": f"5th percentile final NAV ({local_currency})",
            "Value": probability_summary["p5_final_nav_local"],
            "Interpretation": "Downside case where only 5% of simulations are worse."
        },
        {
            "Metric": f"95th percentile final NAV ({local_currency})",
            "Value": probability_summary["p95_final_nav_local"],
            "Interpretation": "Upside case where only 5% of simulations are better."
        },
        {
            "Metric": f"Worst simulated final NAV ({local_currency})",
            "Value": probability_summary["worst_final_nav_local"],
            "Interpretation": "Lowest final NAV observed in the simulation run."
        },
        {
            "Metric": f"Best simulated final NAV ({local_currency})",
            "Value": probability_summary["best_final_nav_local"],
            "Interpretation": "Highest final NAV observed in the simulation run."
        }
    ]

    return pd.DataFrame(rows)


def _build_percentiles_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "Percentile",
                "Final NAV Local",
                "Final NAV LKR",
                "Final NAV Present Value LKR"
            ]
        )

    percentile_values = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    rows = []

    for percentile in percentile_values:
        rows.append(
            {
                "Percentile": f"P{int(percentile * 100)}",
                "Final NAV Local": round(
                    float(results_df["Final NAV Local"].quantile(percentile)),
                    2
                ),
                "Final NAV LKR": round(
                    float(results_df["Final NAV LKR"].quantile(percentile)),
                    2
                ),
                "Final NAV Present Value LKR": round(
                    float(
                        results_df["Final NAV Present Value LKR"].quantile(
                            percentile
                        )
                    ),
                    2
                )
            }
        )

    return pd.DataFrame(rows)


def build_monte_carlo_analysis(
    dataset: Dict[str, Any],
    base_scenario_config: Dict[str, Any],
    simulation_count: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run randomized future cases through the existing deterministic NAV model.

    Monte Carlo changes uncertain assumptions before each run, then records the
    final NAV and debt outcomes. The base dataset and scenario are not mutated.
    """

    safe_simulation_count = int(_clamp(int(simulation_count), 1, 5000))
    rng = np.random.default_rng(int(random_seed))
    local_currency = get_country_currency(dataset)
    base_exchange_rate = get_exchange_rate_to_lkr(dataset)

    records: List[Dict[str, Any]] = []

    for simulation_number in range(1, safe_simulation_count + 1):
        sampled_result = _sample_scenario_config(
            rng=rng,
            base_scenario_config=base_scenario_config
        )

        sampled_config = sampled_result["scenario_config"]
        sampled_values = sampled_result["sampled_values"]

        income_df = calculate_yearly_income(
            dataset=dataset,
            scenario_config=sampled_config
        )
        expense_df = calculate_yearly_expenses(
            dataset=dataset,
            scenario_config=sampled_config
        )
        nav_df = calculate_nav_simulation(
            dataset=dataset,
            scenario_config=sampled_config,
            income_df=income_df,
            expense_df=expense_df
        )
        nav_summary = get_nav_summary(nav_df)

        final_year = int(nav_df.iloc[-1].get("Year", 10)) if not nav_df.empty else 10
        final_nav_local = _safe_float(nav_summary.get("year_10_nav"))
        sampled_exchange_rate = (
            base_exchange_rate * sampled_values["exchange_rate_multiplier"]
        )
        final_nav_lkr = final_nav_local * sampled_exchange_rate
        final_nav_present_value_lkr = calculate_lkr_present_value(
            future_lkr_value=final_nav_lkr,
            years=final_year
        )

        final_debt = _get_final_value(
            df=nav_df,
            column_candidates=[
                "Local Currency Debt",
                "Total Debt",
                "Local Currency Liabilities",
                "Total Liabilities"
            ]
        )
        peak_debt = _get_max_value(
            df=nav_df,
            column_candidates=[
                "Local Currency Debt",
                "Total Debt",
                "Local Currency Liabilities",
                "Total Liabilities"
            ]
        )
        max_debt_to_income_ratio = _get_max_value(
            df=nav_df,
            column_candidates=["Debt-to-Income Ratio"]
        )

        positive_nav = final_nav_local > 0
        high_debt = max_debt_to_income_ratio >= HIGH_DEBT_TO_INCOME_THRESHOLD
        very_bad_outcome = final_nav_local < 0 and high_debt

        records.append(
            {
                "Simulation": simulation_number,
                "Currency": local_currency,
                "Final NAV Local": round(final_nav_local, 2),
                f"Final NAV {local_currency}": round(final_nav_local, 2),
                "Final NAV LKR": round(final_nav_lkr, 2),
                "Final NAV Present Value LKR": round(
                    final_nav_present_value_lkr,
                    2
                ),
                "Final Debt": round(final_debt, 2),
                "Peak Debt": round(peak_debt, 2),
                "Max Debt-to-Income Ratio": round(max_debt_to_income_ratio, 4),
                "Break-even Year": nav_summary.get("break_even_year"),
                "Sampled Salary Growth Rate": round(
                    sampled_values["salary_growth_rate"],
                    4
                ),
                "Sampled Inflation Rate": round(
                    sampled_values["inflation_rate"],
                    4
                ),
                "Sampled Investment Return Rate": round(
                    sampled_values["investment_return_rate"],
                    4
                ),
                "Sampled Rent Multiplier": round(
                    sampled_values["rent_multiplier"],
                    4
                ),
                "Sampled Tuition Multiplier": round(
                    sampled_values["tuition_multiplier"],
                    4
                ),
                "Sampled Childcare Multiplier": round(
                    sampled_values["childcare_multiplier"],
                    4
                ),
                "Sampled Spouse Income Percentage": round(
                    sampled_values["spouse_income_percentage"],
                    4
                ),
                "Sampled Exchange Rate Multiplier": round(
                    sampled_values["exchange_rate_multiplier"],
                    4
                ),
                "Positive NAV": positive_nav,
                "High Debt": high_debt,
                "Very Bad Outcome": very_bad_outcome
            }
        )

    monte_carlo_results_df = pd.DataFrame(records)
    probability_summary = _build_probability_summary(
        results_df=monte_carlo_results_df,
        local_currency=local_currency
    )
    monte_carlo_summary_df = _build_summary_dataframe(
        probability_summary=probability_summary,
        local_currency=local_currency
    )
    monte_carlo_percentiles_df = _build_percentiles_dataframe(
        results_df=monte_carlo_results_df
    )

    return {
        "monte_carlo_results_df": monte_carlo_results_df,
        "monte_carlo_summary_df": monte_carlo_summary_df,
        "monte_carlo_probability_summary": probability_summary,
        "monte_carlo_percentiles_df": monte_carlo_percentiles_df
    }
