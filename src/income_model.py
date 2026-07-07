from typing import Any, Dict, List
import pandas as pd

from src.currency_utils import get_country_currency


def get_value(dataset: Dict[str, Any], path: str) -> Any:
    """
    Read a nested value from the dataset using dot notation.
    """

    current_value = dataset

    for key in path.split("."):
        current_value = current_value[key]

    return current_value


def get_base_salary_for_year(
    year: int,
    graduate_salary: float,
    mid_level_salary: float,
    senior_salary: float
) -> float:
    """
    Career salary stage logic.

    Year 1-3: Graduate salary
    Year 4-7: Mid-level salary
    Year 8-10: Senior salary
    """

    if 1 <= year <= 3:
        return graduate_salary

    if 4 <= year <= 7:
        return mid_level_salary

    return senior_salary


def apply_salary_growth(base_salary: float, year: int, salary_growth_rate: float) -> float:
    """
    Apply salary growth based on model year.

    Example:
        Year 1 = base salary
        Year 2 = base salary * (1 + g)
        Year 3 = base salary * (1 + g)^2
    """

    growth_factor = (1 + salary_growth_rate) ** (year - 1)
    return base_salary * growth_factor


def calculate_student_part_time_income(
    hourly_wage: float,
    legal_work_hours_per_week: float,
    working_weeks_per_year: int
) -> float:
    """
    Student part-time income.

    Formula:
        hourly wage × legal work hours per week × working weeks per year
    """

    return hourly_wage * legal_work_hours_per_week * working_weeks_per_year


def calculate_spouse_income_for_year(
    year: int,
    life_scenario_defaults: Dict[str, Any],
    base_spouse_salary: float,
    spouse_income_percentage: float,
    salary_growth_rate: float
) -> float:
    """
    Spouse income applies only after marriage year.

    If marriage_year is null, spouse income is zero.
    """

    marriage_year = life_scenario_defaults.get("marriage_year")

    if marriage_year is None:
        return 0.0

    if year < marriage_year:
        return 0.0

    grown_spouse_salary = apply_salary_growth(
        base_salary=base_spouse_salary,
        year=year,
        salary_growth_rate=salary_growth_rate
    )

    return grown_spouse_salary * spouse_income_percentage


def calculate_yearly_income(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate yearly income values from Year 1 to Year 10.

    All money values are in selected country local currency.
    """

    local_currency = get_country_currency(dataset)

    time_horizon_years = int(get_value(dataset, "metadata.time_horizon_years"))

    graduate_salary = float(
        get_value(dataset, "income.it_software_salary.graduate_annual_salary.value")
    )
    mid_level_salary = float(
        get_value(dataset, "income.it_software_salary.mid_level_annual_salary.value")
    )
    senior_salary = float(
        get_value(dataset, "income.it_software_salary.senior_annual_salary.value")
    )

    hourly_wage = float(
        get_value(dataset, "income.student_part_time_work.hourly_wage.value")
    )
    working_weeks_per_year = int(
        get_value(dataset, "income.student_part_time_work.working_weeks_per_year.value")
    )
    legal_work_hours_per_week = float(
        get_value(dataset, "visa.student_visa.legal_work_hours_per_week.value")
    )

    base_spouse_salary = float(
        get_value(dataset, "income.spouse_income.annual_salary.value")
    )

    effective_tax_rate = float(
        get_value(dataset, "tax_and_retirement.effective_income_tax_rate.value")
    )
    superannuation_rate = float(
        get_value(dataset, "tax_and_retirement.employer_superannuation_rate.value")
    )

    salary_growth_rate = float(
        scenario_config["adjustable_inputs"]["salary_growth_rate"]
    )

    migration_path_key = scenario_config["selected_keys"]["migration_path"]
    migration_path_defaults = scenario_config["migration_path_defaults"]

    life_scenario_defaults = scenario_config["life_scenario_defaults"]

    spouse_income_percentage = float(
        scenario_config["spouse_income_settings"]["income_percentage"]
    )

    records: List[Dict[str, Any]] = []

    for year in range(1, time_horizon_years + 1):

        base_salary = get_base_salary_for_year(
            year=year,
            graduate_salary=graduate_salary,
            mid_level_salary=mid_level_salary,
            senior_salary=senior_salary
        )

        grown_full_time_salary = apply_salary_growth(
            base_salary=base_salary,
            year=year,
            salary_growth_rate=salary_growth_rate
        )

        student_part_time_income = 0.0
        full_time_employment_income = 0.0
        career_stage = "Not working full-time"

        if migration_path_key == "student_visa_path":
            study_years = migration_path_defaults["study_years"]
            full_time_work_start_year = migration_path_defaults["full_time_work_start_year"]

            if year in study_years:
                student_part_time_income = calculate_student_part_time_income(
                    hourly_wage=hourly_wage,
                    legal_work_hours_per_week=legal_work_hours_per_week,
                    working_weeks_per_year=working_weeks_per_year
                )
                full_time_employment_income = 0.0
                career_stage = "Student / Part-time"

            elif year >= full_time_work_start_year:
                full_time_employment_income = grown_full_time_salary
                career_stage = get_career_stage_label(year)

        elif migration_path_key == "working_visa_path":
            full_time_work_start_year = migration_path_defaults["full_time_work_start_year"]

            if year >= full_time_work_start_year:
                full_time_employment_income = grown_full_time_salary
                career_stage = get_career_stage_label(year)

        spouse_income = calculate_spouse_income_for_year(
            year=year,
            life_scenario_defaults=life_scenario_defaults,
            base_spouse_salary=base_spouse_salary,
            spouse_income_percentage=spouse_income_percentage,
            salary_growth_rate=salary_growth_rate
        )

        gross_income = (
            student_part_time_income
            + full_time_employment_income
            + spouse_income
        )

        tax = gross_income * effective_tax_rate
        net_income = gross_income - tax

        superannuation = (
            full_time_employment_income + spouse_income
        ) * superannuation_rate

        records.append(
            {
                "Year": year,
                "Currency": local_currency,
                "Career Stage": career_stage,
                "Student Part-Time Income": round(student_part_time_income, 2),
                "Full-Time Employment Income": round(full_time_employment_income, 2),
                "Spouse Income": round(spouse_income, 2),
                "Gross Income": round(gross_income, 2),
                "Tax": round(tax, 2),
                "Net Income": round(net_income, 2),
                "Superannuation": round(superannuation, 2)
            }
        )

    income_df = pd.DataFrame(records)
    return income_df


def get_career_stage_label(year: int) -> str:
    """
    Return readable career stage label.
    """

    if 1 <= year <= 3:
        return "Graduate / Junior"

    if 4 <= year <= 7:
        return "Mid-level"

    return "Senior"


def get_income_summary(income_df: pd.DataFrame) -> Dict[str, float]:
    """
    Create total income summary for dashboard metrics.
    """

    return {
        "total_gross_income": float(income_df["Gross Income"].sum()),
        "total_tax": float(income_df["Tax"].sum()),
        "total_net_income": float(income_df["Net Income"].sum()),
        "total_superannuation": float(income_df["Superannuation"].sum()),
        "year_10_gross_income": float(income_df.iloc[-1]["Gross Income"]),
        "year_10_net_income": float(income_df.iloc[-1]["Net Income"])
    }