from typing import Any, Dict, List
import pandas as pd
from src.currency_utils import get_country_currency

from src.education_model import (
    calculate_tuition_expense,
    get_education_status_for_year
)

from src.visa_model import (
    calculate_visa_fee_expense,
    get_visa_status_for_year,
    get_pr_application_year
)

from src.family_model import (
    calculate_childcare_children_count,
    get_general_living_expense_type,
    get_general_living_multiplier,
    get_life_stage_label,
    get_number_of_children_for_year,
    get_scenario_overrides,
    is_married_or_family
)


def get_value(dataset: Dict[str, Any], path: str) -> Any:
    """
    Read a nested value from the dataset using dot notation.
    """

    current_value = dataset

    for key in path.split("."):
        current_value = current_value[key]

    return current_value


def get_inflation_factor(year: int, inflation_rate: float) -> float:
    """
    Inflation factor formula:
    (1 + inflation rate) ^ (year - 1)
    """

    return (1 + inflation_rate) ** (year - 1)


def calculate_rent_expense(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> float:
    """
    Rent logic:

    If single:
        use single monthly rent × 12

    If married or family:
        use family monthly rent × 12
    """

    rent_multiplier = float(
        scenario_config["adjustable_inputs"]["rent_multiplier"]
    )

    if is_married_or_family(
        scenario_config=scenario_config,
        year=year
    ):
        monthly_rent = float(
            get_value(dataset, "expenses.rent.family_monthly.value")
        )
    else:
        monthly_rent = float(
            get_value(dataset, "expenses.rent.single_monthly.value")
        )

    annual_rent = monthly_rent * 12
    return annual_rent * inflation_factor * rent_multiplier


def calculate_general_living_expense(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> float:
    """
    General living expense logic.

    Dataset values:
        single_monthly
        couple_monthly
        family_with_one_child_monthly

    Model assumption:
        If there are two children:
            monthly_living =
            family_with_one_child_monthly * 1.20
    """

    expense_type = get_general_living_expense_type(
        scenario_config=scenario_config,
        year=year
    )

    base_monthly_living = float(
        get_value(
            dataset,
            f"expenses.general_living_excluding_rent.{expense_type}.value"
        )
    )

    living_multiplier = get_general_living_multiplier(
        scenario_config=scenario_config,
        year=year
    )

    monthly_living = base_monthly_living * living_multiplier
    annual_living = monthly_living * 12

    return annual_living * inflation_factor


def calculate_healthcare_expense(
    dataset: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> float:
    """
    Healthcare / insurance expense.
    """

    monthly_healthcare = float(
        get_value(dataset, "expenses.healthcare_or_insurance.monthly_cost.value")
    )

    annual_healthcare = monthly_healthcare * 12
    return annual_healthcare * inflation_factor


def calculate_transport_expense(
    dataset: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> float:
    """
    Public transport expense.
    """

    monthly_transport = float(
        get_value(dataset, "expenses.transport.public_transport_monthly.value")
    )

    annual_transport = monthly_transport * 12
    return annual_transport * inflation_factor


def calculate_childcare_expense(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> float:
    """
    Childcare expense logic.

    This now costs childcare per active childcare child.

    Example:
        first child Year 7
        second child Year 9

        Year 7 -> 1 child cost
        Year 8 -> 1 child cost
        Year 9 -> 2 child cost
        Year 10 -> 2 child cost
    """

    childcare_multiplier = float(
        scenario_config["adjustable_inputs"]["childcare_multiplier"]
    )

    active_children = calculate_childcare_children_count(
        dataset=dataset,
        scenario_config=scenario_config,
        year=year
    )

    if active_children == 0:
        return 0.0

    monthly_childcare_per_child = float(
        get_value(dataset, "expenses.childcare.monthly_cost_per_child.value")
    )

    annual_childcare = monthly_childcare_per_child * 12 * active_children

    return annual_childcare * inflation_factor * childcare_multiplier


def get_static_car_purchase_year(
    scenario_config: Dict[str, Any]
) -> int:
    """
    Get fixed car purchase year from scenario override or dataset defaults.
    """

    overrides = get_scenario_overrides(scenario_config)

    if overrides.get("car_timing_override_enabled", False):
        car_purchase_year = overrides.get("car_purchase_year")

        if car_purchase_year is None:
            return 0

        return int(car_purchase_year)

    migration_path_defaults = scenario_config["migration_path_defaults"]

    return int(
        migration_path_defaults.get("car_purchase_allowed_from_year", 3)
    )


def calculate_car_expense_and_value(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int,
    inflation_factor: float
) -> Dict[str, float]:
    """
    Car logic:

    Fixed timing:
        Car purchase cost applies once.
        Annual running cost applies every year after purchase.
        Car value depreciates yearly.

    Dynamic timing:
        If user selects "Buy car after positive cash flow",
        expense_model returns zero car cost.
        nav_model later decides the purchase year because it has income data.
    """

    overrides = get_scenario_overrides(scenario_config)

    buy_car = bool(scenario_config["car_settings"]["buy_car"])

    if overrides.get("car_timing_override_enabled", False):
        buy_car = bool(overrides.get("buy_car", buy_car))

    if not buy_car:
        return {
            "car_purchase_cost": 0.0,
            "car_annual_running_cost": 0.0,
            "car_total_cost": 0.0,
            "car_value": 0.0
        }

    if overrides.get("car_purchase_after_positive_cash_flow", False):
        return {
            "car_purchase_cost": 0.0,
            "car_annual_running_cost": 0.0,
            "car_total_cost": 0.0,
            "car_value": 0.0
        }

    car_purchase_year = get_static_car_purchase_year(
        scenario_config=scenario_config
    )

    if car_purchase_year <= 0 or year < car_purchase_year:
        return {
            "car_purchase_cost": 0.0,
            "car_annual_running_cost": 0.0,
            "car_total_cost": 0.0,
            "car_value": 0.0
        }

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


def calculate_yearly_expenses(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate yearly expenses from Year 1 to Year 10.

    Delegated logic:
        Tuition timing -> src.education_model
        Visa / PR timing -> src.visa_model
        Family / childcare timing -> src.family_model

    Debt Cost:
        Initialized here as 0.0.
        Updated later inside nav_model.py after debt balances are known.
    """

    time_horizon_years = int(
        get_value(dataset, "metadata.time_horizon_years")
        
    )
    
    local_currency = get_country_currency(dataset)

    inflation_rate = float(
        scenario_config["adjustable_inputs"]["inflation_rate"]
    )

    records: List[Dict[str, Any]] = []

    for year in range(1, time_horizon_years + 1):
        inflation_factor = get_inflation_factor(
            year=year,
            inflation_rate=inflation_rate
        )

        children_count = get_number_of_children_for_year(
            scenario_config=scenario_config,
            year=year
        )

        children_in_childcare = calculate_childcare_children_count(
            dataset=dataset,
            scenario_config=scenario_config,
            year=year
        )

        rent = calculate_rent_expense(
            dataset=dataset,
            scenario_config=scenario_config,
            year=year,
            inflation_factor=inflation_factor
        )

        general_living = calculate_general_living_expense(
            dataset=dataset,
            scenario_config=scenario_config,
            year=year,
            inflation_factor=inflation_factor
        )

        tuition = calculate_tuition_expense(
            dataset=dataset,
            scenario_config=scenario_config,
            year=year,
            inflation_factor=inflation_factor
        )

        visa_fees = calculate_visa_fee_expense(
            dataset=dataset,
            scenario_config=scenario_config,
            year=year,
            inflation_factor=inflation_factor
        )

        healthcare = calculate_healthcare_expense(
            dataset=dataset,
            year=year,
            inflation_factor=inflation_factor
        )

        transport = calculate_transport_expense(
            dataset=dataset,
            year=year,
            inflation_factor=inflation_factor
        )

        childcare = calculate_childcare_expense(
            dataset=dataset,
            scenario_config=scenario_config,
            year=year,
            inflation_factor=inflation_factor
        )

        car_result = calculate_car_expense_and_value(
            dataset=dataset,
            scenario_config=scenario_config,
            year=year,
            inflation_factor=inflation_factor
        )

        # Debt Cost starts as 0.0 because yearly debt interest depends on
        # running debt balances created inside nav_model.py.
        # nav_model.py overwrites this value after calculating interest.
        debt_cost = 0.0

        total_expenses = (
            rent
            + general_living
            + tuition
            + visa_fees
            + healthcare
            + transport
            + childcare
            + car_result["car_total_cost"]
            + debt_cost
        )

        records.append(
            {
                "Year": year,
                "Currency": local_currency,
                "Inflation Factor": round(inflation_factor, 4),
                "Life Stage": get_life_stage_label(
                    year=year,
                    scenario_config=scenario_config
                ),
                "Children Count": children_count,
                "Children in Childcare": children_in_childcare,
                "Education Status": get_education_status_for_year(
                    scenario_config=scenario_config,
                    year=year
                ),
                "Visa Status": get_visa_status_for_year(
                    scenario_config=scenario_config,
                    year=year
                ),
                "PR Application Year": get_pr_application_year(
                    scenario_config=scenario_config
                ),
                "Rent": round(rent, 2),
                "General Living": round(general_living, 2),
                "Tuition": round(tuition, 2),
                "Visa Fees": round(visa_fees, 2),
                "Healthcare / Insurance": round(healthcare, 2),
                "Transport": round(transport, 2),
                "Childcare": round(childcare, 2),
                "Car Purchase Cost": round(car_result["car_purchase_cost"], 2),
                "Car Running Cost": round(car_result["car_annual_running_cost"], 2),
                "Car Cost": round(car_result["car_total_cost"], 2),
                "Debt Cost": round(debt_cost, 2),
                "Total Expenses": round(total_expenses, 2),
                "Car Value": round(car_result["car_value"], 2)
            }
        )

    expense_df = pd.DataFrame(records)
    return expense_df


def get_expense_summary(expense_df: pd.DataFrame) -> Dict[str, float]:
    """
    Create expense summary for dashboard metrics.
    """

    return {
        "total_expenses": float(expense_df["Total Expenses"].sum()),
        "total_rent": float(expense_df["Rent"].sum()),
        "total_living": float(expense_df["General Living"].sum()),
        "total_tuition": float(expense_df["Tuition"].sum()),
        "total_visa_fees": float(expense_df["Visa Fees"].sum()),
        "total_childcare": float(expense_df["Childcare"].sum()),
        "total_car_cost": float(expense_df["Car Cost"].sum()),
        "total_debt_cost": float(expense_df["Debt Cost"].sum()),
        "year_10_expenses": float(expense_df.iloc[-1]["Total Expenses"]),
        "year_10_car_value": float(expense_df.iloc[-1]["Car Value"])
    }