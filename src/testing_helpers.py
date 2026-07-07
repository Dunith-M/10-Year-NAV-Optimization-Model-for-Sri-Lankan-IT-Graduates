from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):
        return pd.DataFrame(value)

    if isinstance(value, dict):
        return pd.DataFrame([value])

    return pd.DataFrame({"Value": [value]})


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def _find_column(
    df: pd.DataFrame,
    exact_candidates: Optional[List[str]] = None,
    contains_candidates: Optional[List[str]] = None
) -> Optional[str]:
    """
    Find a column by exact name or partial text match.
    """

    if df is None or df.empty:
        return None

    exact_candidates = exact_candidates or []
    contains_candidates = contains_candidates or []

    for candidate in exact_candidates:
        if candidate in df.columns:
            return candidate

    normalized_columns = {
        _normalize_text(column): column
        for column in df.columns
    }

    for candidate in exact_candidates:
        normalized_candidate = _normalize_text(candidate)

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    for column in df.columns:
        normalized_column = _normalize_text(column)

        for candidate in contains_candidates:
            if _normalize_text(candidate) in normalized_column:
                return column

    return None


def _sum_column(df: pd.DataFrame, column: Optional[str]) -> float:
    if column is None or df is None or df.empty or column not in df.columns:
        return 0.0

    return float(
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0).sum()
    )


def _last_column_value(df: pd.DataFrame, column: Optional[str]) -> float:
    if column is None or df is None or df.empty or column not in df.columns:
        return 0.0

    return _safe_float(df[column].iloc[-1])


def _max_column_value(df: pd.DataFrame, column: Optional[str]) -> float:
    if column is None or df is None or df.empty or column not in df.columns:
        return 0.0

    return float(
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0).max()
    )


def _min_column_value(df: pd.DataFrame, column: Optional[str]) -> float:
    if column is None or df is None or df.empty or column not in df.columns:
        return 0.0

    return float(
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0).min()
    )


def _find_in_nested_dict(data: Any, possible_keys: List[str]) -> Optional[Any]:
    """
    Find a value inside nested scenario_config dictionaries.
    """

    normalized_keys = {_normalize_text(key) for key in possible_keys}

    if isinstance(data, dict):
        for key, value in data.items():
            if _normalize_text(key) in normalized_keys:
                return value

            found_value = _find_in_nested_dict(value, possible_keys)

            if found_value is not None:
                return found_value

    elif isinstance(data, list):
        for item in data:
            found_value = _find_in_nested_dict(item, possible_keys)

            if found_value is not None:
                return found_value

    return None


def _scenario_text(scenario_config: Dict[str, Any]) -> str:
    """
    Flatten scenario config into one lowercase text string.
    This keeps tests robust even if config keys differ slightly.
    """

    return _normalize_text(scenario_config)


def _build_result(
    test: str,
    status: str,
    severity: str,
    expected: str,
    actual: str,
    evidence: str,
    recommendation: str
) -> Dict[str, str]:
    """
    Standard row format for the Testing tab.
    Includes both Status and Result for compatibility with older UI code.
    """

    return {
        "Test": test,
        "Status": status,
        "Result": status,
        "Severity": severity,
        "Expected": expected,
        "Actual": actual,
        "Evidence": evidence,
        "Recommendation": recommendation
    }


def check_tuition_logic(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    expense_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Validate tuition logic for the current scenario.
    """

    expense_df = _ensure_dataframe(expense_df)

    tuition_column = _find_column(
        expense_df,
        exact_candidates=[
            "Tuition",
            "Tuition Cost",
            "Tuition Expense",
            "Education Cost"
        ],
        contains_candidates=[
            "tuition"
        ]
    )

    total_tuition = _sum_column(expense_df, tuition_column)

    scenario_text = _scenario_text(scenario_config)
    education_mode = _normalize_text(
        _find_in_nested_dict(
            scenario_config,
            [
                "education_mode_label",
                "education_mode",
                "Education mode"
            ]
        )
    )

    looks_like_no_study = (
        "no further study" in scenario_text
        or education_mode == "no further study"
    )

    looks_like_student_or_study = (
        "student visa" in scenario_text
        or "master" in scenario_text
        or "mba" in scenario_text
        or "study" in scenario_text
    ) and not looks_like_no_study

    if looks_like_no_study and total_tuition > 0:
        return _build_result(
            test="Tuition logic",
            status="FAIL",
            severity="High",
            expected="No tuition should be charged when education mode is no further study.",
            actual=f"Total tuition = AUD {total_tuition:,.0f}",
            evidence=f"Tuition column detected: {tuition_column}",
            recommendation="Check education_model.calculate_tuition_expense and scenario overrides."
        )

    if looks_like_student_or_study and total_tuition <= 0:
        return _build_result(
            test="Tuition logic",
            status="WARNING",
            severity="Medium",
            expected="A study/student scenario usually needs tuition cost.",
            actual=f"Total tuition = AUD {total_tuition:,.0f}",
            evidence=f"Tuition column detected: {tuition_column}",
            recommendation="Confirm whether this specific education mode should have zero tuition."
        )

    return _build_result(
        test="Tuition logic",
        status="PASS",
        severity="Low",
        expected="Tuition should match education mode.",
        actual=f"Total tuition = AUD {total_tuition:,.0f}",
        evidence=f"Tuition column detected: {tuition_column}",
        recommendation="No action needed."
    )


def check_childcare_logic(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    expense_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Validate childcare logic for the current scenario.
    """

    expense_df = _ensure_dataframe(expense_df)

    childcare_column = _find_column(
        expense_df,
        exact_candidates=[
            "Childcare",
            "Childcare Cost",
            "Childcare Expense"
        ],
        contains_candidates=[
            "childcare",
            "child care"
        ]
    )

    total_childcare = _sum_column(expense_df, childcare_column)

    scenario_text = _scenario_text(scenario_config)

    no_child_selected = (
        "no child" in scenario_text
        or "no children" in scenario_text
        or "no second child" in scenario_text and "first child" not in scenario_text
    )

    child_expected = (
        "1 child" in scenario_text
        or "2 children" in scenario_text
        or "two children" in scenario_text
        or "first child" in scenario_text
        or "second child" in scenario_text
        or "married_two_children" in scenario_text
        or "married_one_child" in scenario_text
    ) and not no_child_selected

    if no_child_selected and total_childcare > 0:
        return _build_result(
            test="Childcare logic",
            status="FAIL",
            severity="High",
            expected="No childcare should be charged when no child is selected.",
            actual=f"Total childcare = AUD {total_childcare:,.0f}",
            evidence=f"Childcare column detected: {childcare_column}",
            recommendation="Check family_model.get_number_of_children_for_year and expense_model childcare logic."
        )

    if child_expected and total_childcare <= 0:
        return _build_result(
            test="Childcare logic",
            status="WARNING",
            severity="Medium",
            expected="A child scenario usually needs childcare cost for eligible child years.",
            actual=f"Total childcare = AUD {total_childcare:,.0f}",
            evidence=f"Childcare column detected: {childcare_column}",
            recommendation="Confirm child timing and childcare_until_age assumptions."
        )

    return _build_result(
        test="Childcare logic",
        status="PASS",
        severity="Low",
        expected="Childcare should match selected family scenario.",
        actual=f"Total childcare = AUD {total_childcare:,.0f}",
        evidence=f"Childcare column detected: {childcare_column}",
        recommendation="No action needed."
    )


def check_car_logic(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Validate car logic for the current scenario.
    """

    expense_df = _ensure_dataframe(expense_df)
    nav_df = _ensure_dataframe(nav_df)

    car_expense_column = _find_column(
        expense_df,
        exact_candidates=[
            "Car",
            "Car Cost",
            "Car Expense",
            "Vehicle Cost"
        ],
        contains_candidates=[
            "car",
            "vehicle"
        ]
    )

    car_value_column = _find_column(
        nav_df,
        exact_candidates=[
            "Car Value",
            "Vehicle Value"
        ],
        contains_candidates=[
            "car value",
            "vehicle value"
        ]
    )

    car_loan_column = _find_column(
        nav_df,
        exact_candidates=[
            "Car Loan Debt",
            "Vehicle Loan Debt"
        ],
        contains_candidates=[
            "car loan",
            "vehicle loan"
        ]
    )

    total_car_expense = _sum_column(expense_df, car_expense_column)
    final_car_value = _last_column_value(nav_df, car_value_column)
    final_car_loan = _last_column_value(nav_df, car_loan_column)

    scenario_text = _scenario_text(scenario_config)

    no_car_selected = "no car" in scenario_text
    car_selected = (
        "buy car" in scenario_text
        or "car purchase" in scenario_text
        or "buy_car" in scenario_text
    ) and not no_car_selected

    car_activity = total_car_expense + final_car_value + final_car_loan

    if no_car_selected and car_activity > 0:
        return _build_result(
            test="Car logic",
            status="FAIL",
            severity="High",
            expected="No car cost, car value, or car loan should exist when no car is selected.",
            actual=(
                f"Car expense = AUD {total_car_expense:,.0f}; "
                f"car value = AUD {final_car_value:,.0f}; "
                f"car loan = AUD {final_car_loan:,.0f}"
            ),
            evidence=(
                f"Columns detected: expense={car_expense_column}, "
                f"value={car_value_column}, loan={car_loan_column}"
            ),
            recommendation="Check car purchase timing and debt_model.calculate_car_loan_debt."
        )

    if car_selected and car_activity <= 0:
        return _build_result(
            test="Car logic",
            status="WARNING",
            severity="Medium",
            expected="A buy-car scenario should usually create car cost, car value, or car loan.",
            actual="No car activity detected.",
            evidence=(
                f"Columns detected: expense={car_expense_column}, "
                f"value={car_value_column}, loan={car_loan_column}"
            ),
            recommendation="Confirm whether car purchase is delayed until positive cash flow."
        )

    return _build_result(
        test="Car logic",
        status="PASS",
        severity="Low",
        expected="Car costs should match car option.",
        actual=(
            f"Car expense = AUD {total_car_expense:,.0f}; "
            f"car value = AUD {final_car_value:,.0f}; "
            f"car loan = AUD {final_car_loan:,.0f}"
        ),
        evidence=(
            f"Columns detected: expense={car_expense_column}, "
            f"value={car_value_column}, loan={car_loan_column}"
        ),
        recommendation="No action needed."
    )


def check_debt_logic(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Validate debt/liability logic for the current scenario.
    """

    income_df = _ensure_dataframe(income_df)
    expense_df = _ensure_dataframe(expense_df)
    nav_df = _ensure_dataframe(nav_df)

    total_debt_column = _find_column(
        nav_df,
        exact_candidates=[
            "Total Debt",
            "Debt",
            "Total Liabilities",
            "Liabilities"
        ],
        contains_candidates=[
            "total debt",
            "total liabilities"
        ]
    )

    interest_column = _find_column(
        nav_df,
        exact_candidates=[
            "Interest Paid",
            "Debt Interest",
            "Interest"
        ],
        contains_candidates=[
            "interest"
        ]
    )

    income_column = _find_column(
        income_df,
        exact_candidates=[
            "Net Income",
            "Gross Income",
            "Total Income"
        ],
        contains_candidates=[
            "net income",
            "gross income",
            "total income"
        ]
    )

    final_debt = _last_column_value(nav_df, total_debt_column)
    max_debt = _max_column_value(nav_df, total_debt_column)
    min_debt = _min_column_value(nav_df, total_debt_column)
    total_interest = _sum_column(nav_df, interest_column)
    total_income = _sum_column(income_df, income_column)

    debt_to_income_ratio = 0.0

    if total_income > 0:
        debt_to_income_ratio = max_debt / total_income

    if min_debt < 0:
        return _build_result(
            test="Debt logic",
            status="FAIL",
            severity="High",
            expected="Debt/liability values should not become negative.",
            actual=f"Minimum debt/liability value = AUD {min_debt:,.0f}",
            evidence=f"Debt column detected: {total_debt_column}",
            recommendation="Check debt_model liability calculations and sign conventions."
        )

    if max_debt > 0 and total_interest <= 0:
        return _build_result(
            test="Debt logic",
            status="WARNING",
            severity="Medium",
            expected="If debt exists, interest should usually be charged unless intentionally disabled.",
            actual=(
                f"Maximum debt = AUD {max_debt:,.0f}; "
                f"total interest = AUD {total_interest:,.0f}"
            ),
            evidence=(
                f"Debt column detected: {total_debt_column}; "
                f"interest column detected: {interest_column}"
            ),
            recommendation="Check calculate_interest_for_debt or confirm zero-interest assumption."
        )

    if debt_to_income_ratio > 0.5:
        return _build_result(
            test="Debt logic",
            status="WARNING",
            severity="Medium",
            expected="Debt should stay manageable relative to total income.",
            actual=(
                f"Maximum debt = AUD {max_debt:,.0f}; "
                f"debt-to-total-income ratio = {debt_to_income_ratio:.2f}"
            ),
            evidence=f"Income column detected: {income_column}",
            recommendation="This is not a code bug, but the scenario has meaningful financial risk."
        )

    return _build_result(
        test="Debt logic",
        status="PASS",
        severity="Low",
        expected="Debt should remain non-negative and financially explainable.",
        actual=(
            f"Final debt = AUD {final_debt:,.0f}; "
            f"maximum debt = AUD {max_debt:,.0f}; "
            f"total interest = AUD {total_interest:,.0f}"
        ),
        evidence=(
            f"Debt column detected: {total_debt_column}; "
            f"interest column detected: {interest_column}"
        ),
        recommendation="No action needed."
    )


def check_nav_output_logic(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    nav_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Basic NAV output validation.
    """

    nav_df = _ensure_dataframe(nav_df)

    nav_column = _find_column(
        nav_df,
        exact_candidates=[
            "NAV",
            "Net Asset Value",
            "Net Asset Value AUD"
        ],
        contains_candidates=[
            "nav",
            "net asset"
        ]
    )

    if nav_df.empty:
        return _build_result(
            test="NAV output",
            status="FAIL",
            severity="High",
            expected="NAV simulation should return a non-empty yearly table.",
            actual="NAV table is empty.",
            evidence="No rows found.",
            recommendation="Check nav_model.calculate_nav_simulation."
        )

    if nav_column is None:
        return _build_result(
            test="NAV output",
            status="FAIL",
            severity="High",
            expected="NAV table should contain a NAV or Net Asset Value column.",
            actual=f"Available columns: {list(nav_df.columns)}",
            evidence="NAV column not found.",
            recommendation="Check nav_model output column names."
        )

    final_nav = _last_column_value(nav_df, nav_column)

    return _build_result(
        test="NAV output",
        status="PASS",
        severity="Low",
        expected="NAV table should exist and contain final NAV.",
        actual=f"Final NAV = AUD {final_nav:,.0f}",
        evidence=f"NAV column detected: {nav_column}",
        recommendation="No action needed."
    )


def check_data_completeness(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Check whether core model tables are present.
    """

    income_df = _ensure_dataframe(income_df)
    expense_df = _ensure_dataframe(expense_df)
    nav_df = _ensure_dataframe(nav_df)

    empty_tables = []

    if income_df.empty:
        empty_tables.append("Income")

    if expense_df.empty:
        empty_tables.append("Expenses")

    if nav_df.empty:
        empty_tables.append("NAV")

    if empty_tables:
        return _build_result(
            test="Data completeness",
            status="FAIL",
            severity="High",
            expected="Income, Expenses, and NAV tables should all be generated.",
            actual=f"Empty tables: {', '.join(empty_tables)}",
            evidence="One or more model outputs are empty.",
            recommendation="Check the model pipeline before using this scenario in viva."
        )

    return _build_result(
        test="Data completeness",
        status="PASS",
        severity="Low",
        expected="Income, Expenses, and NAV tables should all be generated.",
        actual=(
            f"Income rows = {len(income_df)}, "
            f"Expense rows = {len(expense_df)}, "
            f"NAV rows = {len(nav_df)}"
        ),
        evidence="All core output tables exist.",
        recommendation="No action needed."
    )


def run_model_validation_tests(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    nav_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run final model validation tests.

    This replaces the old practice of keeping final testing logic inside
    sensitivity.py. Sensitivity and testing are separate responsibilities.
    """

    results = [
        check_data_completeness(
            dataset=dataset,
            scenario_config=scenario_config,
            income_df=income_df,
            expense_df=expense_df,
            nav_df=nav_df
        ),
        check_nav_output_logic(
            dataset=dataset,
            scenario_config=scenario_config,
            nav_df=nav_df
        ),
        check_tuition_logic(
            dataset=dataset,
            scenario_config=scenario_config,
            expense_df=expense_df
        ),
        check_childcare_logic(
            dataset=dataset,
            scenario_config=scenario_config,
            expense_df=expense_df
        ),
        check_car_logic(
            dataset=dataset,
            scenario_config=scenario_config,
            expense_df=expense_df,
            nav_df=nav_df
        ),
        check_debt_logic(
            dataset=dataset,
            scenario_config=scenario_config,
            income_df=income_df,
            expense_df=expense_df,
            nav_df=nav_df
        )
    ]

    return pd.DataFrame(results)