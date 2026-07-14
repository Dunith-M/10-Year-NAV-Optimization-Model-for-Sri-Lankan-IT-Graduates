from typing import Any, Dict, List
import pandas as pd

from src.currency_utils import (
    LKR_PRESENT_VALUE_DISCOUNT_RATE,
    calculate_lkr_present_value,
    convert_local_to_lkr,
    get_country_currency,
    get_exchange_rate_to_lkr
)

from src.debt_model import (
    allocate_cash_shortage_to_debt_categories,
    calculate_car_loan_debt,
    calculate_debt_interest_costs,
    calculate_education_debt,
    calculate_migration_debt,
    calculate_negative_cash_debt,
    calculate_total_liabilities,
    get_debt_interest_rates,
    repay_debts_with_positive_cash_flow
)


def get_value(dataset: Dict[str, Any], path: str) -> Any:
    current_value = dataset

    for key in path.split("."):
        current_value = current_value[key]

    return current_value


def get_inflation_factor(year: int, inflation_rate: float) -> float:
    return (1 + inflation_rate) ** (year - 1)


def get_scenario_overrides(scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    return scenario_config.get("scenario_overrides", {})


def calculate_dynamic_car_expense_and_value(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int,
    car_purchase_year: int
) -> Dict[str, float]:

    inflation_rate = float(
        scenario_config["adjustable_inputs"]["inflation_rate"]
    )

    inflation_factor = get_inflation_factor(
        year=year,
        inflation_rate=inflation_rate
    )

    base_car_purchase_price = float(
        get_value(dataset, "car.used_car_purchase_price.value")
    )
    annual_insurance = float(
        get_value(dataset, "car.annual_insurance.value")
    )
    annual_fuel = float(
        get_value(dataset, "car.annual_fuel.value")
    )
    annual_maintenance = float(
        get_value(dataset, "car.annual_maintenance.value")
    )
    depreciation_rate = float(
        get_value(dataset, "car.annual_depreciation_rate.value")
    )

    inflated_car_purchase_price = base_car_purchase_price * inflation_factor

    car_purchase_cost = 0.0

    if year == car_purchase_year:
        car_purchase_cost = inflated_car_purchase_price

    annual_running_cost = (
        annual_insurance + annual_fuel + annual_maintenance
    ) * inflation_factor

    years_since_purchase = year - car_purchase_year

    car_value = inflated_car_purchase_price * (
        (1 - depreciation_rate) ** years_since_purchase
    )

    car_total_cost = car_purchase_cost + annual_running_cost

    return {
        "car_purchase_cost": car_purchase_cost,
        "car_annual_running_cost": annual_running_cost,
        "car_total_cost": car_total_cost,
        "car_value": car_value
    }


def apply_dynamic_car_to_expense_df(
    expense_df: pd.DataFrame,
    index: int,
    car_result: Dict[str, float]
) -> None:

    expense_df.at[index, "Car Purchase Cost"] = round(
        car_result["car_purchase_cost"], 2
    )
    expense_df.at[index, "Car Running Cost"] = round(
        car_result["car_annual_running_cost"], 2
    )
    expense_df.at[index, "Car Cost"] = round(
        car_result["car_total_cost"], 2
    )
    expense_df.at[index, "Car Value"] = round(
        car_result["car_value"], 2
    )

    old_total_expenses = float(expense_df.at[index, "Total Expenses"])

    expense_df.at[index, "Total Expenses"] = round(
        old_total_expenses + car_result["car_total_cost"],
        2
    )


def apply_debt_cost_to_expense_df(
    expense_df: pd.DataFrame,
    index: int,
    interest_paid: float
) -> None:

    if "Debt Cost" not in expense_df.columns:
        expense_df["Debt Cost"] = 0.0

    old_debt_cost = float(expense_df.at[index, "Debt Cost"])
    old_total_expenses = float(expense_df.at[index, "Total Expenses"])

    expense_df.at[index, "Debt Cost"] = round(interest_paid, 2)
    expense_df.at[index, "Total Expenses"] = round(
        old_total_expenses - old_debt_cost + interest_paid,
        2
    )


def calculate_debt_to_income_ratio(
    total_debt: float,
    gross_income: float
) -> float:

    if gross_income <= 0:
        return 0.0

    return total_debt / gross_income


def calculate_weighted_debt_interest_rate(
    opening_total_debt: float,
    interest_paid: float
) -> float:

    if opening_total_debt <= 0:
        return 0.0

    return interest_paid / opening_total_debt


def calculate_nav_simulation(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Core NAV calculation engine.

    All base calculations happen in selected local currency.
    LKR equivalent columns are added using selected-country exchange rate.
    """

    local_currency = get_country_currency(dataset)
    exchange_rate_to_lkr = get_exchange_rate_to_lkr(dataset)

    savings_interest_rate = float(
        get_value(dataset, "investment_and_economy.savings_interest_rate.value")
    )

    investment_return_rate = float(
        scenario_config["adjustable_inputs"]["investment_return_rate"]
    )

    debt_interest_rates = get_debt_interest_rates(dataset)

    education_loan_interest_rate = debt_interest_rates["education_debt"]
    migration_loan_interest_rate = debt_interest_rates["migration_debt"]
    car_loan_interest_rate = debt_interest_rates["car_loan_debt"]
    negative_cash_interest_rate = debt_interest_rates["negative_cash_debt"]

    investment_settings = scenario_config.get("investment_settings", {})
    investment_method = investment_settings.get("method", "save_only")

    investment_percentage = float(
        investment_settings.get(
            "investment_percentage",
            1.0 if investment_method == "invest_positive_cash_flow" else 0.0
        )
    )

    investment_percentage = max(0.0, min(investment_percentage, 1.0))

    overrides = get_scenario_overrides(scenario_config)

    dynamic_car_enabled = bool(
        overrides.get("car_purchase_after_positive_cash_flow", False)
    )

    dynamic_car_purchased = False
    dynamic_car_purchase_year = None

    cash_savings_balance = 0.0
    investment_balance = 0.0
    superannuation_balance = 0.0

    education_debt = 0.0
    migration_debt = 0.0
    car_loan_debt = 0.0
    negative_cash_debt = 0.0

    records: List[Dict[str, Any]] = []

    for index in range(len(income_df)):
        income_row = income_df.iloc[index]
        expense_row = expense_df.iloc[index]

        year = int(income_row["Year"])

        gross_income = float(income_row.get("Gross Income", income_row["Net Income"]))
        net_income = float(income_row["Net Income"])
        yearly_superannuation = float(income_row["Superannuation"])

        total_expenses = float(expense_row["Total Expenses"])
        car_value = float(expense_row["Car Value"])

        if dynamic_car_enabled:
            pre_car_cash_flow = net_income - total_expenses

            if (
                not dynamic_car_purchased
                and pre_car_cash_flow > 0
            ):
                dynamic_car_purchased = True
                dynamic_car_purchase_year = year

            if dynamic_car_purchased and dynamic_car_purchase_year is not None:
                car_result = calculate_dynamic_car_expense_and_value(
                    dataset=dataset,
                    scenario_config=scenario_config,
                    year=year,
                    car_purchase_year=dynamic_car_purchase_year
                )

                apply_dynamic_car_to_expense_df(
                    expense_df=expense_df,
                    index=index,
                    car_result=car_result
                )

                total_expenses = float(expense_df.at[index, "Total Expenses"])
                car_value = float(expense_df.at[index, "Car Value"])
                expense_row = expense_df.iloc[index]

        debt_balances_before_interest = {
            "education_debt": education_debt,
            "migration_debt": migration_debt,
            "car_loan_debt": car_loan_debt,
            "negative_cash_debt": negative_cash_debt
        }

        opening_total_debt = calculate_total_liabilities(
            education_debt=education_debt,
            migration_debt=migration_debt,
            car_loan_debt=car_loan_debt,
            negative_cash_debt=negative_cash_debt
        )

        interest_costs, interest_paid = calculate_debt_interest_costs(
            debt_balances=debt_balances_before_interest,
            interest_rates=debt_interest_rates
        )

        apply_debt_cost_to_expense_df(
            expense_df=expense_df,
            index=index,
            interest_paid=interest_paid
        )

        total_expenses = float(expense_df.at[index, "Total Expenses"])
        expense_row = expense_df.iloc[index]

        cash_savings_balance = cash_savings_balance * (1 + savings_interest_rate)
        investment_balance = investment_balance * (1 + investment_return_rate)

        superannuation_balance += yearly_superannuation

        cash_flow = net_income - total_expenses

        positive_cash_flow = 0.0
        cash_shortage = 0.0

        amount_added_to_savings = 0.0
        amount_added_to_investment = 0.0

        education_debt_added = 0.0
        migration_debt_added = 0.0
        car_loan_debt_added = 0.0
        negative_cash_debt_added = 0.0

        education_debt_repaid = 0.0
        migration_debt_repaid = 0.0
        car_loan_debt_repaid = 0.0
        negative_cash_debt_repaid = 0.0

        debt_repayment = 0.0
        net_cash_after_debt_repayment = 0.0

        if cash_flow >= 0:
            positive_cash_flow = cash_flow

            debt_balances_after_repayment, repayments, debt_repayment, remaining_cash = (
                repay_debts_with_positive_cash_flow(
                    available_cash=positive_cash_flow,
                    debt_balances={
                        "education_debt": education_debt,
                        "migration_debt": migration_debt,
                        "car_loan_debt": car_loan_debt,
                        "negative_cash_debt": negative_cash_debt
                    },
                    interest_rates=debt_interest_rates
                )
            )

            education_debt = debt_balances_after_repayment["education_debt"]
            migration_debt = debt_balances_after_repayment["migration_debt"]
            car_loan_debt = debt_balances_after_repayment["car_loan_debt"]
            negative_cash_debt = debt_balances_after_repayment["negative_cash_debt"]

            education_debt_repaid = repayments["education_debt"]
            migration_debt_repaid = repayments["migration_debt"]
            car_loan_debt_repaid = repayments["car_loan_debt"]
            negative_cash_debt_repaid = repayments["negative_cash_debt"]

            net_cash_after_debt_repayment = remaining_cash

            amount_added_to_investment = remaining_cash * investment_percentage
            amount_added_to_savings = remaining_cash - amount_added_to_investment

            investment_balance += amount_added_to_investment
            cash_savings_balance += amount_added_to_savings

        else:
            cash_shortage = abs(cash_flow)

            debt_additions = allocate_cash_shortage_to_debt_categories(
                cash_shortage=cash_shortage,
                expense_row=expense_row
            )

            education_debt_added = debt_additions["education_debt"]
            migration_debt_added = debt_additions["migration_debt"]
            car_loan_debt_added = debt_additions["car_loan_debt"]
            negative_cash_debt_added = debt_additions["negative_cash_debt"]

            education_debt = calculate_education_debt(
                current_debt=education_debt,
                tuition_shortage=education_debt_added
            )
            migration_debt = calculate_migration_debt(
                current_debt=migration_debt,
                migration_shortage=migration_debt_added
            )
            car_loan_debt = calculate_car_loan_debt(
                current_debt=car_loan_debt,
                car_purchase_shortage=car_loan_debt_added
            )
            negative_cash_debt = calculate_negative_cash_debt(
                current_debt=negative_cash_debt,
                general_living_shortage=negative_cash_debt_added
            )

        total_assets = (
            cash_savings_balance
            + investment_balance
            + superannuation_balance
            + car_value
        )

        total_debt = calculate_total_liabilities(
            education_debt=education_debt,
            migration_debt=migration_debt,
            car_loan_debt=car_loan_debt,
            negative_cash_debt=negative_cash_debt
        )

        total_liabilities = total_debt
        nav = total_assets - total_liabilities

        lkr_cash_flow = convert_local_to_lkr(cash_flow, dataset)
        lkr_assets = convert_local_to_lkr(total_assets, dataset)
        lkr_debt = convert_local_to_lkr(total_debt, dataset)
        lkr_liabilities = convert_local_to_lkr(total_liabilities, dataset)
        lkr_nav = convert_local_to_lkr(nav, dataset)
        present_value_lkr_nav = calculate_lkr_present_value(
            future_lkr_value=lkr_nav,
            years=year
        )

        debt_to_income_ratio = calculate_debt_to_income_ratio(
            total_debt=total_debt,
            gross_income=gross_income
        )

        weighted_debt_interest_rate = calculate_weighted_debt_interest_rate(
            opening_total_debt=opening_total_debt,
            interest_paid=interest_paid
        )

        records.append(
            {
                "Year": year,
                "Currency": local_currency,
                "Exchange Rate to LKR": round(exchange_rate_to_lkr, 4),

                "Gross Income": round(gross_income, 2),
                "Net Income": round(net_income, 2),
                "Total Expenses": round(total_expenses, 2),
                "Cash Flow": round(cash_flow, 2),
                "Positive Cash Flow": round(positive_cash_flow, 2),
                "Cash Shortage": round(cash_shortage, 2),

                "Local Currency Cash Flow": round(cash_flow, 2),
                "LKR Cash Flow": round(lkr_cash_flow, 2),

                "Amount Added To Savings": round(amount_added_to_savings, 2),
                "Amount Added To Investment": round(amount_added_to_investment, 2),
                "Investment Percentage": round(investment_percentage, 4),

                "Debt Repayment": round(debt_repayment, 2),
                "Net Cash After Debt Repayment": round(net_cash_after_debt_repayment, 2),

                "Education Debt Added": round(education_debt_added, 2),
                "Migration Debt Added": round(migration_debt_added, 2),
                "Car Loan Debt Added": round(car_loan_debt_added, 2),
                "Negative Cash Debt Added": round(negative_cash_debt_added, 2),

                "Education Debt Repaid": round(education_debt_repaid, 2),
                "Migration Debt Repaid": round(migration_debt_repaid, 2),
                "Car Loan Debt Repaid": round(car_loan_debt_repaid, 2),
                "Negative Cash Debt Repaid": round(negative_cash_debt_repaid, 2),

                "Cash Savings Balance": round(cash_savings_balance, 2),
                "Investment Balance": round(investment_balance, 2),
                "Superannuation Balance": round(superannuation_balance, 2),
                "Car Value": round(car_value, 2),

                "Total Assets": round(total_assets, 2),
                "Local Currency Assets": round(total_assets, 2),
                "LKR Assets": round(lkr_assets, 2),

                "Education Debt": round(education_debt, 2),
                "Migration Debt": round(migration_debt, 2),
                "Car Loan Debt": round(car_loan_debt, 2),
                "Negative Cash Debt": round(negative_cash_debt, 2),

                "Total Debt": round(total_debt, 2),
                "Local Currency Debt": round(total_debt, 2),
                "LKR Debt": round(lkr_debt, 2),

                "Total Liabilities": round(total_liabilities, 2),
                "Local Currency Liabilities": round(total_liabilities, 2),
                "LKR Liabilities": round(lkr_liabilities, 2),

                "NAV": round(nav, 2),
                "Local Currency NAV": round(nav, 2),
                "LKR NAV": round(lkr_nav, 2),
                "Present Value LKR NAV": round(present_value_lkr_nav, 2),
                "Present Value Discount Rate": round(LKR_PRESENT_VALUE_DISCOUNT_RATE, 4),
                "Present Value Discount Years": year,

                "Interest Paid": round(interest_paid, 2),
                "Education Debt Interest": round(interest_costs["education_debt"], 2),
                "Migration Debt Interest": round(interest_costs["migration_debt"], 2),
                "Car Loan Debt Interest": round(interest_costs["car_loan_debt"], 2),
                "Negative Cash Debt Interest": round(interest_costs["negative_cash_debt"], 2),
                "Debt-to-Income Ratio": round(debt_to_income_ratio, 4),

                "Savings Interest Rate": round(savings_interest_rate, 4),
                "Investment Return Rate": round(investment_return_rate, 4),
                "Debt Interest Rate": round(weighted_debt_interest_rate, 4),
                "Education Loan Interest Rate": round(education_loan_interest_rate, 4),
                "Migration Loan Interest Rate": round(migration_loan_interest_rate, 4),
                "Car Loan Interest Rate": round(car_loan_interest_rate, 4),
                "Negative Cash Interest Rate": round(negative_cash_interest_rate, 4),

                "Dynamic Car Enabled": dynamic_car_enabled,
                "Dynamic Car Purchase Year": dynamic_car_purchase_year
            }
        )

    nav_df = pd.DataFrame(records)
    return nav_df


def get_nav_summary(nav_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create final NAV summary for dashboard metrics.
    """

    final_row = nav_df.iloc[-1]
    final_year = int(final_row.get("Year", 10))

    nav_column = "Local Currency NAV" if "Local Currency NAV" in nav_df.columns else "NAV"
    assets_column = "Local Currency Assets" if "Local Currency Assets" in nav_df.columns else "Total Assets"
    debt_column = "Local Currency Debt" if "Local Currency Debt" in nav_df.columns else "Total Debt"
    liabilities_column = (
        "Local Currency Liabilities"
        if "Local Currency Liabilities" in nav_df.columns
        else "Total Liabilities"
    )

    break_even_rows = nav_df[nav_df[nav_column] >= 0]

    if len(break_even_rows) > 0:
        break_even_year = int(break_even_rows.iloc[0]["Year"])
    else:
        break_even_year = None

    highest_debt_row = nav_df.loc[nav_df[debt_column].idxmax()]
    lowest_nav_row = nav_df.loc[nav_df[nav_column].idxmin()]

    year_10_nav_lkr = float(final_row.get("LKR NAV", final_row[nav_column]))
    year_10_nav_present_value_lkr = float(
        final_row.get(
            "Present Value LKR NAV",
            calculate_lkr_present_value(
                future_lkr_value=year_10_nav_lkr,
                years=final_year
            )
        )
    )

    return {
        "currency": str(final_row.get("Currency", "LOCAL")),
        "exchange_rate_to_lkr": float(final_row.get("Exchange Rate to LKR", 1.0)),

        "year_10_nav": float(final_row[nav_column]),
        "year_10_nav_lkr": year_10_nav_lkr,
        "year_10_nav_present_value_lkr": year_10_nav_present_value_lkr,
        "year_10_present_value_discount_rate": LKR_PRESENT_VALUE_DISCOUNT_RATE,
        "year_10_present_value_years": final_year,

        "year_10_total_assets": float(final_row[assets_column]),
        "year_10_total_assets_lkr": float(final_row.get("LKR Assets", final_row[assets_column])),

        "year_10_total_liabilities": float(final_row[liabilities_column]),
        "year_10_total_liabilities_lkr": float(
            final_row.get("LKR Liabilities", final_row[liabilities_column])
        ),

        "year_10_total_debt": float(final_row[debt_column]),
        "year_10_total_debt_lkr": float(final_row.get("LKR Debt", final_row[debt_column])),

        "year_10_cash_savings": float(final_row["Cash Savings Balance"]),
        "year_10_investment_balance": float(final_row["Investment Balance"]),
        "year_10_superannuation_balance": float(final_row["Superannuation Balance"]),
        "year_10_car_value": float(final_row["Car Value"]),

        "year_10_education_debt": float(final_row["Education Debt"]),
        "year_10_migration_debt": float(final_row["Migration Debt"]),
        "year_10_car_loan_debt": float(final_row["Car Loan Debt"]),
        "year_10_negative_cash_debt": float(final_row["Negative Cash Debt"]),
        "year_10_debt_to_income_ratio": float(final_row["Debt-to-Income Ratio"]),

        "total_positive_cash_flow": float(nav_df["Positive Cash Flow"].sum()),
        "total_cash_shortage": float(nav_df["Cash Shortage"].sum()),
        "total_interest_paid": float(nav_df["Interest Paid"].sum()),
        "total_debt_repayment": float(nav_df["Debt Repayment"].sum()),

        "break_even_year": break_even_year,

        "highest_debt_year": int(highest_debt_row["Year"]),
        "highest_debt_amount": float(highest_debt_row[debt_column]),
        "highest_debt_amount_lkr": float(
            highest_debt_row.get("LKR Debt", highest_debt_row[debt_column])
        ),
        "highest_debt_to_income_ratio": float(
            highest_debt_row["Debt-to-Income Ratio"]
        ),

        "lowest_nav_year": int(lowest_nav_row["Year"]),
        "lowest_nav_amount": float(lowest_nav_row[nav_column]),
        "lowest_nav_amount_lkr": float(
            lowest_nav_row.get("LKR NAV", lowest_nav_row[nav_column])
        )
    }


def create_final_simulation_table(
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create the final compact simulation table.
    """

    final_df = pd.DataFrame(
        {
            "Year": nav_df["Year"],
            "Currency": nav_df.get("Currency", "LOCAL"),

            "Gross Income": income_df["Gross Income"],
            "Net Income": nav_df["Net Income"],
            "Expenses": nav_df["Total Expenses"],
            "Debt Cost": expense_df["Debt Cost"],
            "Interest Paid": nav_df["Interest Paid"],

            "Local Currency Cash Flow": nav_df["Local Currency Cash Flow"],
            "LKR Cash Flow": nav_df["LKR Cash Flow"],

            "Debt Repayment": nav_df["Debt Repayment"],

            "Education Debt": nav_df["Education Debt"],
            "Migration Debt": nav_df["Migration Debt"],
            "Car Loan Debt": nav_df["Car Loan Debt"],
            "Negative Cash Debt": nav_df["Negative Cash Debt"],

            "Local Currency Debt": nav_df["Local Currency Debt"],
            "LKR Debt": nav_df["LKR Debt"],

            "Debt-to-Income Ratio": nav_df["Debt-to-Income Ratio"],

            "Local Currency Assets": nav_df["Local Currency Assets"],
            "LKR Assets": nav_df["LKR Assets"],

            "Local Currency Liabilities": nav_df["Local Currency Liabilities"],
            "LKR Liabilities": nav_df["LKR Liabilities"],

            "Local Currency NAV": nav_df["Local Currency NAV"],
            "LKR NAV": nav_df["LKR NAV"],
            "Present Value LKR NAV": nav_df["Present Value LKR NAV"],
            "Present Value Discount Rate": nav_df["Present Value Discount Rate"],
            "Present Value Discount Years": nav_df["Present Value Discount Years"]
        }
    )

    return final_df
