from typing import Any, Dict, Optional


LKR_PRESENT_VALUE_DISCOUNT_RATE = 0.055
LKR_PRESENT_VALUE_YEARS = 10


CURRENCY_SYMBOLS = {
    "AUD": "$",
    "LKR": "Rs.",
    "EUR": "€",
    "JPY": "¥",
    "USD": "$",
    "CAD": "$",
    "NZD": "$",
    "SGD": "$",
    "AED": "د.إ"
}


def get_nested_value(dataset: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely read a nested value from the dataset using dot notation.
    """

    current_value: Any = dataset

    for key in path.split("."):
        if not isinstance(current_value, dict):
            return default

        if key not in current_value:
            return default

        current_value = current_value[key]

    return current_value


def get_country_currency(dataset: Dict[str, Any]) -> str:
    """
    Return selected country currency from metadata.
    """

    currency = get_nested_value(dataset, "metadata.currency")

    if not currency:
        raise ValueError("Dataset metadata.currency is missing.")

    return str(currency).upper().strip()


def get_currency_symbol(currency: str) -> str:
    """
    Return currency symbol for known currencies.
    """

    currency_code = str(currency).upper().strip()
    return CURRENCY_SYMBOLS.get(currency_code, currency_code)


def get_exchange_rate_key(dataset: Dict[str, Any]) -> str:
    """
    Return the selected country's exchange-rate key.

    Examples:
        AUD -> aud_to_lkr_exchange_rate
        LKR -> lkr_to_lkr_exchange_rate
        EUR -> eur_to_lkr_exchange_rate
        JPY -> jpy_to_lkr_exchange_rate
    """

    currency = get_country_currency(dataset)
    expected_key = f"{currency.lower()}_to_lkr_exchange_rate"

    economy_section = dataset.get("investment_and_economy", {})

    if expected_key in economy_section:
        return expected_key

    if currency == "LKR":
        return "lkr_to_lkr_exchange_rate"

    for key in economy_section.keys():
        normalized_key = str(key).lower().strip()

        if normalized_key.endswith("_to_lkr_exchange_rate"):
            return str(key)

    raise KeyError(
        f"Exchange-rate field not found. Expected: "
        f"investment_and_economy.{expected_key}.value"
    )


def get_exchange_rate_path(dataset: Dict[str, Any]) -> str:
    """
    Return full dot path to selected country's LKR exchange rate.
    """

    exchange_rate_key = get_exchange_rate_key(dataset)
    return f"investment_and_economy.{exchange_rate_key}.value"


def get_exchange_rate_to_lkr(dataset: Dict[str, Any]) -> float:
    """
    Return exchange rate from selected local currency to LKR.
    """

    currency = get_country_currency(dataset)
    exchange_rate_path = get_exchange_rate_path(dataset)

    exchange_rate = get_nested_value(dataset, exchange_rate_path)

    if exchange_rate is None and currency == "LKR":
        return 1.0

    try:
        exchange_rate_float = float(exchange_rate)
    except Exception as error:
        raise ValueError(
            f"Invalid exchange-rate value at {exchange_rate_path}: {exchange_rate}"
        ) from error

    if exchange_rate_float <= 0:
        raise ValueError(
            f"Exchange rate must be greater than zero at {exchange_rate_path}."
        )

    return exchange_rate_float


def convert_local_to_lkr(
    amount: float,
    dataset: Dict[str, Any],
    exchange_rate: Optional[float] = None
) -> float:
    """
    Convert selected-country local currency amount into LKR.
    """

    if exchange_rate is None:
        exchange_rate = get_exchange_rate_to_lkr(dataset)

    return float(amount) * float(exchange_rate)


def calculate_present_value(
    future_value: Any,
    discount_rate: float = LKR_PRESENT_VALUE_DISCOUNT_RATE,
    years: int = LKR_PRESENT_VALUE_YEARS
) -> float:
    """
    Discount a future value back to today's money value.

    Formula:
        present value = future value / (1 + discount rate) ** years
    """

    try:
        amount = float(future_value)
    except Exception:
        amount = 0.0

    safe_years = max(int(years), 0)
    safe_discount_rate = float(discount_rate)

    return amount / ((1 + safe_discount_rate) ** safe_years)


def calculate_lkr_present_value(
    future_lkr_value: Any,
    years: int = LKR_PRESENT_VALUE_YEARS,
    discount_rate: float = LKR_PRESENT_VALUE_DISCOUNT_RATE
) -> float:
    """
    Discount a future LKR value using Sri Lanka inflation as the LKR discount rate.
    """

    return calculate_present_value(
        future_value=future_lkr_value,
        discount_rate=discount_rate,
        years=years
    )


def format_local_currency(
    value: Any,
    dataset: Optional[Dict[str, Any]] = None,
    currency: Optional[str] = None,
    decimals: int = 0
) -> str:
    """
    Format selected-country local currency.

    Example:
        AUD $10,000
        EUR €10,000
        JPY ¥10,000
        LKR Rs. 10,000
    """

    if currency is None:
        if dataset is None:
            currency = "LOCAL"
        else:
            currency = get_country_currency(dataset)

    currency_code = str(currency).upper().strip()
    symbol = get_currency_symbol(currency_code)

    try:
        amount = float(value)
    except Exception:
        amount = 0.0

    formatted_amount = f"{amount:,.{decimals}f}"

    if symbol == currency_code:
        return f"{currency_code} {formatted_amount}"

    return f"{currency_code} {symbol}{formatted_amount}"


def format_lkr(value: Any, decimals: int = 0) -> str:
    """
    Format LKR amount.
    """

    try:
        amount = float(value)
    except Exception:
        amount = 0.0

    return f"LKR Rs. {amount:,.{decimals}f}"


def format_lkr_equivalent(
    local_value: Any,
    dataset: Dict[str, Any],
    decimals: int = 0
) -> str:
    """
    Convert local value to LKR and format it.
    """

    lkr_value = convert_local_to_lkr(
        amount=float(local_value),
        dataset=dataset
    )

    return format_lkr(lkr_value, decimals=decimals)


def format_currency_pair(
    local_value: Any,
    dataset: Dict[str, Any],
    decimals: int = 0
) -> str:
    """
    Format local currency with LKR equivalent.

    Example:
        AUD $10,000 | LKR Rs. 2,000,000
    """

    local_text = format_local_currency(
        value=local_value,
        dataset=dataset,
        decimals=decimals
    )

    lkr_text = format_lkr_equivalent(
        local_value=local_value,
        dataset=dataset,
        decimals=decimals
    )

    return f"{local_text} | {lkr_text}"
