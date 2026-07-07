from typing import Any, Dict, Tuple


DebtBalances = Dict[str, float]
DebtInterestRates = Dict[str, float]


def get_optional_value(
    dataset: Dict[str, Any],
    path: str,
    default_value: float
) -> float:
    """
    Safely read a nested dataset value using dot notation.

    This is intentionally defensive because older dataset files may not have
    separate car loan or negative cash interest rate fields yet.
    """

    current_value: Any = dataset

    try:
        for key in path.split("."):
            current_value = current_value[key]

        return float(current_value)

    except (KeyError, TypeError, ValueError):
        return float(default_value)


def calculate_interest_for_debt(
    debt_balance: float,
    interest_rate: float
) -> float:
    """
    Calculate one year of interest cost for an opening debt balance.

    This model treats interest as a yearly debt cost. The interest is added to
    annual expenses in nav_model.py, not blindly added twice to the debt balance.
    """

    if debt_balance <= 0:
        return 0.0

    if interest_rate <= 0:
        return 0.0

    return debt_balance * interest_rate


def calculate_education_debt(
    current_debt: float,
    tuition_shortage: float = 0.0,
    repayment: float = 0.0
) -> float:
    """
    Update education debt.

    Rule:
        Tuition shortage -> Education Debt
    """

    updated_debt = current_debt + tuition_shortage - repayment
    return max(updated_debt, 0.0)


def calculate_migration_debt(
    current_debt: float,
    migration_shortage: float = 0.0,
    repayment: float = 0.0
) -> float:
    """
    Update migration debt.

    Rule:
        Visa / migration shortage -> Migration Debt
    """

    updated_debt = current_debt + migration_shortage - repayment
    return max(updated_debt, 0.0)


def calculate_car_loan_debt(
    current_debt: float,
    car_purchase_shortage: float = 0.0,
    repayment: float = 0.0
) -> float:
    """
    Update car loan debt.

    Rule:
        Car purchase shortage -> Car Loan Debt
    """

    updated_debt = current_debt + car_purchase_shortage - repayment
    return max(updated_debt, 0.0)


def calculate_negative_cash_debt(
    current_debt: float,
    general_living_shortage: float = 0.0,
    repayment: float = 0.0
) -> float:
    """
    Update negative cash debt.

    Rule:
        General living shortage -> Negative Cash Debt
    """

    updated_debt = current_debt + general_living_shortage - repayment
    return max(updated_debt, 0.0)


def calculate_total_liabilities(
    education_debt: float,
    migration_debt: float,
    car_loan_debt: float,
    negative_cash_debt: float
) -> float:
    """
    Total liabilities used in NAV.
    """

    return (
        education_debt
        + migration_debt
        + car_loan_debt
        + negative_cash_debt
    )


def get_debt_interest_rates(dataset: Dict[str, Any]) -> DebtInterestRates:
    """
    Read available debt interest rates from the dataset.

    Fallbacks are used so the app does not break if the JSON file has not yet
    been expanded with separate car loan / negative cash interest rates.
    """

    education_rate = get_optional_value(
        dataset=dataset,
        path="loans_and_debt.education_loan_interest_rate.value",
        default_value=0.0
    )

    migration_rate = get_optional_value(
        dataset=dataset,
        path="loans_and_debt.migration_loan_interest_rate.value",
        default_value=education_rate
    )

    car_loan_rate = get_optional_value(
        dataset=dataset,
        path="loans_and_debt.car_loan_interest_rate.value",
        default_value=migration_rate
    )

    negative_cash_rate = get_optional_value(
        dataset=dataset,
        path="loans_and_debt.negative_cash_interest_rate.value",
        default_value=migration_rate
    )

    return {
        "education_debt": education_rate,
        "migration_debt": migration_rate,
        "car_loan_debt": car_loan_rate,
        "negative_cash_debt": negative_cash_rate
    }


def calculate_debt_interest_costs(
    debt_balances: DebtBalances,
    interest_rates: DebtInterestRates
) -> Tuple[DebtBalances, float]:
    """
    Calculate annual interest cost for each debt category.
    """

    interest_costs = {
        "education_debt": calculate_interest_for_debt(
            debt_balance=debt_balances.get("education_debt", 0.0),
            interest_rate=interest_rates.get("education_debt", 0.0)
        ),
        "migration_debt": calculate_interest_for_debt(
            debt_balance=debt_balances.get("migration_debt", 0.0),
            interest_rate=interest_rates.get("migration_debt", 0.0)
        ),
        "car_loan_debt": calculate_interest_for_debt(
            debt_balance=debt_balances.get("car_loan_debt", 0.0),
            interest_rate=interest_rates.get("car_loan_debt", 0.0)
        ),
        "negative_cash_debt": calculate_interest_for_debt(
            debt_balance=debt_balances.get("negative_cash_debt", 0.0),
            interest_rate=interest_rates.get("negative_cash_debt", 0.0)
        )
    }

    total_interest = sum(interest_costs.values())
    return interest_costs, total_interest


def allocate_cash_shortage_to_debt_categories(
    cash_shortage: float,
    expense_row: Any
) -> Dict[str, float]:
    """
    Split a yearly cash shortage into academic debt categories.

    Simple classification rule:
        Tuition shortage -> Education Debt
        Visa / migration shortage -> Migration Debt
        Car purchase shortage -> Car Loan Debt
        Remaining shortage -> Negative Cash Debt

    This is deliberately simple. It explains why debt happened without becoming
    a full banking-grade loan amortization model.
    """

    remaining_shortage = max(cash_shortage, 0.0)

    tuition_cost = max(float(expense_row.get("Tuition", 0.0)), 0.0)
    visa_cost = max(float(expense_row.get("Visa Fees", 0.0)), 0.0)
    car_purchase_cost = max(float(expense_row.get("Car Purchase Cost", 0.0)), 0.0)

    education_shortage = min(remaining_shortage, tuition_cost)
    remaining_shortage -= education_shortage

    migration_shortage = min(remaining_shortage, visa_cost)
    remaining_shortage -= migration_shortage

    car_purchase_shortage = min(remaining_shortage, car_purchase_cost)
    remaining_shortage -= car_purchase_shortage

    negative_cash_shortage = remaining_shortage

    return {
        "education_debt": education_shortage,
        "migration_debt": migration_shortage,
        "car_loan_debt": car_purchase_shortage,
        "negative_cash_debt": negative_cash_shortage
    }


def repay_debts_with_positive_cash_flow(
    available_cash: float,
    debt_balances: DebtBalances,
    interest_rates: DebtInterestRates
) -> Tuple[DebtBalances, DebtBalances, float, float]:
    """
    Use positive cash flow to repay existing debt before saving/investing.

    Repayment priority:
        Highest interest debt first.

    This keeps the model realistic while still staying simple.
    """

    remaining_cash = max(available_cash, 0.0)

    updated_balances: DebtBalances = {
        "education_debt": max(debt_balances.get("education_debt", 0.0), 0.0),
        "migration_debt": max(debt_balances.get("migration_debt", 0.0), 0.0),
        "car_loan_debt": max(debt_balances.get("car_loan_debt", 0.0), 0.0),
        "negative_cash_debt": max(debt_balances.get("negative_cash_debt", 0.0), 0.0)
    }

    repayments: DebtBalances = {
        "education_debt": 0.0,
        "migration_debt": 0.0,
        "car_loan_debt": 0.0,
        "negative_cash_debt": 0.0
    }

    repayment_priority = sorted(
        updated_balances.keys(),
        key=lambda debt_key: interest_rates.get(debt_key, 0.0),
        reverse=True
    )

    for debt_key in repayment_priority:
        if remaining_cash <= 0:
            break

        current_debt = updated_balances[debt_key]

        if current_debt <= 0:
            continue

        repayment_amount = min(remaining_cash, current_debt)

        updated_balances[debt_key] -= repayment_amount
        repayments[debt_key] += repayment_amount
        remaining_cash -= repayment_amount

    total_repayment = sum(repayments.values())

    return updated_balances, repayments, total_repayment, remaining_cash