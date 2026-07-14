import copy
from typing import Any, Dict, List

import pandas as pd

from src.currency_utils import (
    LKR_PRESENT_VALUE_DISCOUNT_RATE,
    calculate_lkr_present_value,
    get_country_currency,
    get_exchange_rate_to_lkr
)
from src.scenario_builder import build_scenario_config
from src.income_model import calculate_yearly_income
from src.expense_model import calculate_yearly_expenses
from src.nav_model import calculate_nav_simulation, get_nav_summary


SENSITIVITY_LEVELS = [-0.20, -0.10, 0.00, 0.10, 0.20]


SENSITIVITY_VARIABLES = [
    "Salary growth",
    "Rent",
    "Tuition",
    "Childcare",
    "Investment return",
    "Exchange rate",
    "Spouse income percentage"
]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def run_single_nav_simulation(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any]
) -> Dict[str, Any]:

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

    nav_summary = get_nav_summary(nav_df)

    return {
        "income_df": income_df,
        "expense_df": expense_df,
        "nav_df": nav_df,
        "nav_summary": nav_summary
    }


def get_impact_direction(delta_nav_lkr: float) -> str:
    if delta_nav_lkr > 0:
        return "Improves NAV"

    if delta_nav_lkr < 0:
        return "Reduces NAV"

    return "No change"


def get_risk_interpretation(
    variable_name: str,
    change: float,
    delta_nav_lkr: float
) -> str:
    direction = get_impact_direction(delta_nav_lkr)

    if change == 0:
        return "Base case"

    change_label = f"{change * 100:+.0f}%"

    if direction == "Reduces NAV":
        return f"{variable_name} at {change_label} hurts the selected scenario."

    if direction == "Improves NAV":
        return f"{variable_name} at {change_label} helps the selected scenario."

    return f"{variable_name} at {change_label} has no material NAV effect."


def apply_sensitivity_change(
    base_scenario_config: Dict[str, Any],
    variable_name: str,
    change: float
) -> Dict[str, Any]:

    scenario_config = copy.deepcopy(base_scenario_config)
    adjustable_inputs = scenario_config["adjustable_inputs"]

    if variable_name == "Salary growth":
        base_value = adjustable_inputs["salary_growth_rate"]
        adjusted_value = base_value * (1 + change)
        adjustable_inputs["salary_growth_rate"] = clamp(adjusted_value, 0.0, 1.0)

    elif variable_name == "Rent":
        base_value = adjustable_inputs["rent_multiplier"]
        adjusted_value = base_value * (1 + change)
        adjustable_inputs["rent_multiplier"] = clamp(adjusted_value, 0.0, 10.0)

    elif variable_name == "Tuition":
        base_value = adjustable_inputs["tuition_multiplier"]
        adjusted_value = base_value * (1 + change)
        adjustable_inputs["tuition_multiplier"] = clamp(adjusted_value, 0.0, 10.0)

    elif variable_name == "Childcare":
        base_value = adjustable_inputs["childcare_multiplier"]
        adjusted_value = base_value * (1 + change)
        adjustable_inputs["childcare_multiplier"] = clamp(adjusted_value, 0.0, 10.0)

    elif variable_name == "Investment return":
        base_value = adjustable_inputs["investment_return_rate"]
        adjusted_value = base_value * (1 + change)
        adjustable_inputs["investment_return_rate"] = clamp(adjusted_value, 0.0, 1.0)

    elif variable_name == "Spouse income percentage":
        base_value = scenario_config["spouse_income_settings"]["income_percentage"]
        adjusted_value = base_value * (1 + change)
        scenario_config["spouse_income_settings"]["income_percentage"] = clamp(
            adjusted_value,
            0.0,
            1.0
        )

    elif variable_name == "Exchange rate":
        pass

    else:
        raise ValueError(f"Unknown sensitivity variable: {variable_name}")

    return scenario_config


def build_sensitivity_analysis(
    dataset: Dict[str, Any],
    base_scenario_config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:

    local_currency = get_country_currency(dataset)
    base_exchange_rate = get_exchange_rate_to_lkr(dataset)

    base_result = run_single_nav_simulation(
        dataset=dataset,
        scenario_config=base_scenario_config
    )

    base_nav_local = float(base_result["nav_summary"]["year_10_nav"])
    base_nav_lkr = base_nav_local * base_exchange_rate
    base_nav_present_value_lkr = calculate_lkr_present_value(base_nav_lkr)

    records: List[Dict[str, Any]] = []

    for variable_name in SENSITIVITY_VARIABLES:
        for change in SENSITIVITY_LEVELS:

            temp_exchange_rate = base_exchange_rate

            if variable_name == "Exchange rate":
                temp_exchange_rate = base_exchange_rate * (1 + change)
                temp_scenario_config = copy.deepcopy(base_scenario_config)
            else:
                temp_scenario_config = apply_sensitivity_change(
                    base_scenario_config=base_scenario_config,
                    variable_name=variable_name,
                    change=change
                )

            temp_result = run_single_nav_simulation(
                dataset=dataset,
                scenario_config=temp_scenario_config
            )

            temp_nav_local = float(temp_result["nav_summary"]["year_10_nav"])
            temp_nav_lkr = temp_nav_local * temp_exchange_rate
            temp_nav_present_value_lkr = calculate_lkr_present_value(temp_nav_lkr)

            delta_nav_local = temp_nav_local - base_nav_local
            delta_nav_lkr = temp_nav_lkr - base_nav_lkr
            delta_nav_present_value_lkr = (
                temp_nav_present_value_lkr - base_nav_present_value_lkr
            )

            if base_nav_local != 0:
                delta_percent_local = delta_nav_local / abs(base_nav_local)
            else:
                delta_percent_local = 0.0

            if base_nav_lkr != 0:
                delta_percent_lkr = delta_nav_lkr / abs(base_nav_lkr)
            else:
                delta_percent_lkr = 0.0

            records.append(
                {
                    "Variable": variable_name,
                    "Change": change,
                    "Change Label": "Base" if change == 0 else f"{change * 100:+.0f}%",
                    "Local Currency": local_currency,
                    "Exchange Rate to LKR": round(temp_exchange_rate, 4),

                    "Year-10 NAV Local": round(temp_nav_local, 2),
                    f"Year-10 NAV {local_currency}": round(temp_nav_local, 2),
                    "Year-10 NAV LKR": round(temp_nav_lkr, 2),
                    "Year-10 NAV Present Value LKR": round(temp_nav_present_value_lkr, 2),

                    "Base NAV Local": round(base_nav_local, 2),
                    f"Base NAV {local_currency}": round(base_nav_local, 2),
                    "Base NAV LKR": round(base_nav_lkr, 2),
                    "Base NAV Present Value LKR": round(base_nav_present_value_lkr, 2),

                    "Delta NAV Local": round(delta_nav_local, 2),
                    f"Delta NAV {local_currency}": round(delta_nav_local, 2),
                    "Delta NAV LKR": round(delta_nav_lkr, 2),
                    "Delta NAV Present Value LKR": round(delta_nav_present_value_lkr, 2),

                    "Delta % Local": round(delta_percent_local, 4),
                    "Delta % LKR": round(delta_percent_lkr, 4),
                    "Present Value Discount Rate": LKR_PRESENT_VALUE_DISCOUNT_RATE,
                    "Present Value Discount Years": 10,

                    "Impact Direction": get_impact_direction(delta_nav_lkr),
                    "Risk Interpretation": get_risk_interpretation(
                        variable_name=variable_name,
                        change=change,
                        delta_nav_lkr=delta_nav_lkr
                    )
                }
            )

    sensitivity_df = pd.DataFrame(records)

    tornado_records = []

    for variable_name in SENSITIVITY_VARIABLES:
        variable_rows = sensitivity_df[sensitivity_df["Variable"] == variable_name]

        max_impact_local = float(variable_rows["Delta NAV Local"].abs().max())
        max_impact_lkr = float(variable_rows["Delta NAV LKR"].abs().max())
        max_impact_present_value_lkr = float(
            variable_rows["Delta NAV Present Value LKR"].abs().max()
        )
        max_impact_percent = float(variable_rows["Delta % LKR"].abs().max())

        tornado_records.append(
            {
                "Variable": variable_name,
                "Local Currency": local_currency,
                "Max Impact Local": round(max_impact_local, 2),
                f"Max Impact {local_currency}": round(max_impact_local, 2),
                "Max Impact LKR": round(max_impact_lkr, 2),
                "Max Impact Present Value LKR": round(max_impact_present_value_lkr, 2),
                "Max Impact %": round(max_impact_percent, 4)
            }
        )

    tornado_df = pd.DataFrame(tornado_records)
    tornado_df["Impact Rank"] = tornado_df["Max Impact %"].rank(
        method="dense",
        ascending=False
    ).astype(int)

    tornado_df = tornado_df.sort_values(
        by="Max Impact %",
        ascending=True
    ).reset_index(drop=True)

    return {
        "sensitivity_df": sensitivity_df,
        "tornado_df": tornado_df
    }


def build_final_testing_results(
    dataset: Dict[str, Any],
    car_option_label: str,
    investment_option_label: str,
    spouse_income_case_label: str,
    salary_growth_rate: float,
    inflation_rate: float,
    investment_return_rate: float,
    rent_multiplier: float,
    tuition_multiplier: float,
    childcare_multiplier: float
) -> pd.DataFrame:

    local_currency = get_country_currency(dataset)
    exchange_rate = get_exchange_rate_to_lkr(dataset)

    test_cases = [
        {
            "Test Case": "Student visa + Single + No car",
            "Migration Path": "Student visa path",
            "Life Scenario": "Single",
            "Car Option": "No car",
            "Investment Option": investment_option_label
        },
        {
            "Test Case": "Student visa + Married one child + Car",
            "Migration Path": "Student visa path",
            "Life Scenario": "Married one child",
            "Car Option": "Buy car",
            "Investment Option": investment_option_label
        },
        {
            "Test Case": "Working visa + Single + Invest",
            "Migration Path": "Working visa path",
            "Life Scenario": "Single",
            "Car Option": "No car",
            "Investment Option": "Invest positive cash flow"
        },
        {
            "Test Case": "Working visa + Married two children + No car",
            "Migration Path": "Working visa path",
            "Life Scenario": "Married two children",
            "Car Option": "No car",
            "Investment Option": investment_option_label
        }
    ]

    records: List[Dict[str, Any]] = []

    for test_case in test_cases:
        scenario_config = build_scenario_config(
            dataset=dataset,
            migration_path_label=test_case["Migration Path"],
            life_scenario_label=test_case["Life Scenario"],
            car_option_label=test_case["Car Option"],
            investment_option_label=test_case["Investment Option"],
            spouse_income_case_label=spouse_income_case_label,
            salary_growth_rate=salary_growth_rate,
            inflation_rate=inflation_rate,
            investment_return_rate=investment_return_rate,
            rent_multiplier=rent_multiplier,
            tuition_multiplier=tuition_multiplier,
            childcare_multiplier=childcare_multiplier
        )

        result = run_single_nav_simulation(
            dataset=dataset,
            scenario_config=scenario_config
        )

        expense_df = result["expense_df"]
        nav_df = result["nav_df"]
        nav_summary = result["nav_summary"]

        final_nav_local = float(nav_summary["year_10_nav"])
        final_nav_lkr = final_nav_local * exchange_rate
        final_nav_present_value_lkr = float(
            nav_summary.get(
                "year_10_nav_present_value_lkr",
                calculate_lkr_present_value(final_nav_lkr)
            )
        )

        nav_not_blank = pd.notna(final_nav_local)

        rent_grows = expense_df.iloc[-1]["Rent"] >= expense_df.iloc[0]["Rent"]

        migration_path_key = scenario_config["selected_keys"]["migration_path"]
        study_years = scenario_config["migration_path_defaults"].get("study_years", [])

        if migration_path_key == "student_visa_path":
            tuition_ok = True

            for _, row in expense_df.iterrows():
                year = int(row["Year"])
                tuition = float(row["Tuition"])

                if year in study_years and tuition <= 0:
                    tuition_ok = False

                if year not in study_years and tuition != 0:
                    tuition_ok = False
        else:
            tuition_ok = expense_df["Tuition"].sum() == 0

        first_child_year = scenario_config["life_scenario_defaults"].get("first_child_year")

        if first_child_year is None:
            childcare_ok = expense_df["Childcare"].sum() == 0
        else:
            childcare_before_child = expense_df[
                expense_df["Year"] < first_child_year
            ]["Childcare"].sum()

            childcare_after_child = expense_df[
                expense_df["Year"] >= first_child_year
            ]["Childcare"].sum()

            childcare_ok = childcare_before_child == 0 and childcare_after_child > 0

        if test_case["Car Option"] == "No car":
            car_ok = expense_df["Car Cost"].sum() == 0
        else:
            car_purchase_rows = expense_df[
                expense_df["Car Purchase Cost"] > 0
            ]

            car_running_after_purchase = expense_df[
                expense_df["Car Running Cost"] > 0
            ]["Car Running Cost"].sum()

            car_ok = len(car_purchase_rows) == 1 and car_running_after_purchase > 0

        has_cash_shortage = nav_df["Cash Shortage"].sum() > 0

        if has_cash_shortage:
            debt_logic_ok = nav_df.iloc[-1]["Total Debt"] > 0
        else:
            debt_logic_ok = True

        final_nav_realistic = pd.notna(final_nav_local) and abs(final_nav_local) < 100_000_000_000

        all_checks_passed = all(
            [
                nav_not_blank,
                rent_grows,
                tuition_ok,
                childcare_ok,
                car_ok,
                debt_logic_ok,
                final_nav_realistic
            ]
        )

        records.append(
            {
                "Test Case": test_case["Test Case"],
                "Currency": local_currency,
                "Final NAV Local": round(final_nav_local, 2),
                f"Final NAV {local_currency}": round(final_nav_local, 2),
                "Final NAV LKR": round(final_nav_lkr, 2),
                "Final NAV Present Value LKR": round(final_nav_present_value_lkr, 2),
                "NAV Not Blank": nav_not_blank,
                "Rent Grows with Inflation": rent_grows,
                "Tuition Logic OK": tuition_ok,
                "Childcare Logic OK": childcare_ok,
                "Car Logic OK": car_ok,
                "Debt Logic OK": debt_logic_ok,
                "Final NAV Realistic": final_nav_realistic,
                "All Checks Passed": all_checks_passed
            }
        )

    testing_df = pd.DataFrame(records)
    return testing_df
