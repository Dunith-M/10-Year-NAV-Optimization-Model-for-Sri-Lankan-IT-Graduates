from typing import Any, Dict, List, Optional
import copy


FAMILY_WITH_TWO_CHILDREN_LIVING_MULTIPLIER = 1.20


def get_value(dataset: Dict[str, Any], path: str) -> Any:
    """
    Read a nested value from the dataset using dot notation.
    """

    current_value = dataset

    for key in path.split("."):
        current_value = current_value[key]

    return current_value


def normalize_year(value: Any) -> Optional[int]:
    """
    Convert a year-like value into a clean integer.

    Rules:
        None, 0, negative values, and invalid values become None.
    """

    if value is None:
        return None

    try:
        year = int(value)
    except (TypeError, ValueError):
        return None

    if year <= 0:
        return None

    return year


def get_scenario_overrides(scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely read scenario overrides.

    This keeps older scenario_config objects backward-compatible.
    """

    return scenario_config.get("scenario_overrides", {})


def get_effective_life_scenario_defaults(
    scenario_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply family and child timing overrides without changing the dataset.

    The dataset stays as the base truth.
    Advanced UI selections only change the effective model logic.

    Important rule:
        A second child cannot exist if the first child is disabled.
    """

    original_defaults = scenario_config.get("life_scenario_defaults", {})
    effective_defaults = copy.deepcopy(original_defaults)

    overrides = get_scenario_overrides(scenario_config)

    if overrides.get("first_child_override_enabled", False):
        first_child_year = normalize_year(
            overrides.get("first_child_year_override")
        )

        effective_defaults["first_child_year"] = first_child_year

        if first_child_year is None:
            effective_defaults["second_child_year"] = None

    if overrides.get("second_child_override_enabled", False):
        second_child_year = normalize_year(
            overrides.get("second_child_year_override")
        )

        effective_defaults["second_child_year"] = second_child_year

    marriage_year = normalize_year(effective_defaults.get("marriage_year"))
    first_child_year = normalize_year(effective_defaults.get("first_child_year"))
    second_child_year = normalize_year(effective_defaults.get("second_child_year"))

    effective_defaults["marriage_year"] = marriage_year
    effective_defaults["first_child_year"] = first_child_year
    effective_defaults["second_child_year"] = second_child_year

    if first_child_year is None:
        effective_defaults["second_child_year"] = None

    if (
        first_child_year is not None
        and second_child_year is not None
        and second_child_year < first_child_year
    ):
        effective_defaults["second_child_year"] = None

    return effective_defaults


def get_marriage_year(scenario_config: Dict[str, Any]) -> Optional[int]:
    """
    Return the effective marriage year after scenario overrides.
    """

    life_defaults = get_effective_life_scenario_defaults(scenario_config)
    return normalize_year(life_defaults.get("marriage_year"))


def get_first_child_year(scenario_config: Dict[str, Any]) -> Optional[int]:
    """
    Return the effective first child birth year after scenario overrides.
    """

    life_defaults = get_effective_life_scenario_defaults(scenario_config)
    return normalize_year(life_defaults.get("first_child_year"))


def get_second_child_year(scenario_config: Dict[str, Any]) -> Optional[int]:
    """
    Return the effective second child birth year after scenario overrides.
    """

    life_defaults = get_effective_life_scenario_defaults(scenario_config)
    return normalize_year(life_defaults.get("second_child_year"))


def get_child_birth_years(
    scenario_config: Dict[str, Any]
) -> List[Optional[int]]:
    """
    Return child birth years in model order.
    """

    return [
        get_first_child_year(scenario_config),
        get_second_child_year(scenario_config)
    ]


def get_number_of_children_for_year(
    scenario_config: Dict[str, Any],
    year: int
) -> int:
    """
    Count how many children exist in a given simulation year.

    Example:
        first_child_year = 7
        second_child_year = 9

        Year 6 -> 0
        Year 7 -> 1
        Year 8 -> 1
        Year 9 -> 2
    """

    children_count = 0

    for child_birth_year in get_child_birth_years(scenario_config):
        if child_birth_year is None:
            continue

        if year >= child_birth_year:
            children_count += 1

    return children_count


def calculate_childcare_children_count(
    dataset: Dict[str, Any],
    scenario_config: Dict[str, Any],
    year: int
) -> int:
    """
    Count how many children generate childcare cost in a given year.

    Childcare rule:
        For each child:
            if child exists and child_age <= childcare_until_age,
            childcare applies.

    Child age is modelled as:
        age = current_year - child_birth_year

    Example:
        first_child_year = 7
        second_child_year = 9
        childcare_until_age = 4

        Year 7  -> first child age 0 -> childcare applies
        Year 8  -> first child age 1 -> childcare applies
        Year 9  -> first child age 2, second child age 0 -> both apply
        Year 10 -> first child age 3, second child age 1 -> both apply
    """

    childcare_until_age = int(
        get_value(dataset, "expenses.childcare.applies_until_child_age.value")
    )

    active_childcare_children = 0

    for child_birth_year in get_child_birth_years(scenario_config):
        if child_birth_year is None:
            continue

        child_age = year - child_birth_year

        if child_age < 0:
            continue

        if child_age <= childcare_until_age:
            active_childcare_children += 1

    return active_childcare_children


def is_married_or_family(
    scenario_config: Dict[str, Any],
    year: int
) -> bool:
    """
    Return True when the person is married or has at least one child.
    """

    marriage_year = get_marriage_year(scenario_config)

    if marriage_year is not None and year >= marriage_year:
        return True

    if get_number_of_children_for_year(scenario_config, year) > 0:
        return True

    return False


def has_child(
    scenario_config: Dict[str, Any],
    year: int
) -> bool:
    """
    Return True when at least one child exists in the given year.
    """

    return get_number_of_children_for_year(scenario_config, year) > 0


def get_general_living_expense_type(
    scenario_config: Dict[str, Any],
    year: int
) -> str:
    """
    Select the dataset general living expense category.

    Dataset-supported categories:
        single_monthly
        couple_monthly
        family_with_one_child_monthly

    There is no dataset field for two children.
    Two-child uplift is handled separately using a model assumption.
    """

    life_scenario_key = scenario_config["selected_keys"]["life_scenario"]
    marriage_year = get_marriage_year(scenario_config)
    children_count = get_number_of_children_for_year(
        scenario_config=scenario_config,
        year=year
    )

    if life_scenario_key == "single" and children_count == 0:
        return "single_monthly"

    if marriage_year is not None and year < marriage_year and children_count == 0:
        return "single_monthly"

    if children_count >= 1:
        return "family_with_one_child_monthly"

    if marriage_year is not None and year >= marriage_year:
        return "couple_monthly"

    return "single_monthly"


def get_general_living_multiplier(
    scenario_config: Dict[str, Any],
    year: int
) -> float:
    """
    Return the model-only living cost multiplier.

    Dataset has:
        family_with_one_child_monthly

    Model assumption:
        family_with_two_children_living =
        family_with_one_child_monthly * 1.20
    """

    children_count = get_number_of_children_for_year(
        scenario_config=scenario_config,
        year=year
    )

    if children_count < 2:
        return 1.0

    overrides = get_scenario_overrides(scenario_config)

    return float(
        overrides.get(
            "family_with_two_children_living_multiplier",
            FAMILY_WITH_TWO_CHILDREN_LIVING_MULTIPLIER
        )
    )


def get_life_stage_label(
    year: int,
    scenario_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Return a readable life-stage label for tables and dashboards.

    The first argument is year so it can still be called as:
        get_life_stage_label(year)

    For full scenario-aware output, call:
        get_life_stage_label(year, scenario_config)
    """

    if scenario_config is None:
        return f"Year {year}"

    marriage_year = get_marriage_year(scenario_config)
    children_count = get_number_of_children_for_year(
        scenario_config=scenario_config,
        year=year
    )

    if children_count >= 2:
        return "Family with 2 children"

    if children_count == 1:
        return "Family with 1 child"

    if marriage_year is not None and year >= marriage_year:
        return "Married / couple"

    return "Single"