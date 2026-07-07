from typing import Any, Dict
import re

import pandas as pd
import streamlit as st

from src.data_loader import get_dataset_source_records


SOURCE_COLUMNS = [
    "Country",
    "Variable",
    "Source",
    "Source type",
    "Source category",
    "Year",
    "Notes",
    "Reliability"
]


OFFICIAL_SOURCE_MARKERS = [
    ".gov",
    "gov.",
    "gov.au",
    ".gov.au",
    "gov.lk",
    ".gov.lk",
    "gov.de",
    ".gov.de",
    "go.jp",
    ".go.jp",
    "homeaffairs.gov.au",
    "ato.gov.au",
    "abs.gov.au",
    "fairwork.gov.au",
    "servicesaustralia.gov.au",
    "studyinaustralia.gov.au",
    "immi.homeaffairs.gov.au",
    "austrade.gov.au",
    "ird.gov.lk",
    "statistics.gov.lk",
    "immigration.gov.lk",
    "destatis.de",
    "make-it-in-germany.com",
    "mhlw.go.jp",
    "stat.go.jp",
    "jasso.go.jp"
]


PLACEHOLDER_SOURCE_MARKERS = [
    ":contentreference",
    "contentreference[",
    "turn",
    "filecite",
    "source needed",
    "placeholder"
]


SOURCE_CATEGORY_OPTIONS = [
    "Official sources",
    "Salary sources",
    "Cost of living sources",
    "Education sources",
    "Economic sources",
    "Assumptions only"
]


def normalize_text(value: Any) -> str:
    """
    Convert any value into safe lowercase text.
    """

    if value is None:
        return ""

    return str(value).strip().lower()


def source_exists(source: Any) -> bool:
    """
    Check whether a source field exists.
    """

    return bool(str(source or "").strip())


def is_clean_url(source: Any) -> bool:
    """
    Check whether the source looks like a clean URL.
    """

    source_text = str(source or "").strip()

    return bool(
        re.search(
            r"https?://[^\s\]\)]+",
            source_text
        )
    )


def is_placeholder_source(source: Any) -> bool:
    """
    Detect placeholder-style source references.

    Example:
    :contentReference[...]
    """

    source_text = normalize_text(source)

    return any(
        marker in source_text
        for marker in PLACEHOLDER_SOURCE_MARKERS
    )


def is_official_source(source: Any) -> bool:
    """
    Detect whether a source appears to be official/government.
    """

    source_text = normalize_text(source)

    return any(
        marker in source_text
        for marker in OFFICIAL_SOURCE_MARKERS
    )


def notes_include_assumption(notes: Any) -> bool:
    """
    Detect assumption-based values from notes.
    """

    notes_text = normalize_text(notes)

    assumption_markers = [
        "assumed",
        "assumption",
        "estimate",
        "estimated",
        "approx",
        "approximate"
    ]

    return any(
        marker in notes_text
        for marker in assumption_markers
    )


def classify_source_type(source: Any) -> str:
    """
    Classify source type for transparency.
    """

    if not source_exists(source):
        return "Missing"

    if is_placeholder_source(source):
        return "Placeholder / not clean URL"

    if not is_clean_url(source):
        return "Text reference / not URL"

    if is_official_source(source):
        return "Official / government"

    return "General web source"


def calculate_source_reliability(source: Any, notes: Any = "") -> str:
    """
    Calculate simple reliability label.

    Rules:
        Source exists and looks official -> High
        Source exists but general estimate -> Medium
        Source empty -> Low
        Notes include assumed -> Low/Medium
    """

    if not source_exists(source):
        return "Low"

    if is_placeholder_source(source):
        return "Low"

    if notes_include_assumption(notes):
        if is_official_source(source):
            return "Medium"
        return "Low/Medium"

    if is_clean_url(source) and is_official_source(source):
        return "High"

    if is_clean_url(source):
        return "Medium"

    return "Medium"


def classify_source_category(
    variable: Any,
    source: Any,
    notes: Any = ""
) -> str:
    """
    Classify source into user-facing filter categories.
    """

    combined_text = " ".join(
        [
            normalize_text(variable),
            normalize_text(source),
            normalize_text(notes)
        ]
    )

    if notes_include_assumption(notes) or not source_exists(source):
        return "Assumptions only"

    if is_official_source(source):
        return "Official sources"

    salary_markers = [
        "salary",
        "income",
        "wage",
        "earnings",
        "graduate",
        "spouse"
    ]

    cost_markers = [
        "rent",
        "living",
        "cost of living",
        "transport",
        "food",
        "healthcare",
        "childcare",
        "car",
        "expense"
    ]

    education_markers = [
        "tuition",
        "education",
        "university",
        "study",
        "master",
        "mba",
        "student"
    ]

    economic_markers = [
        "inflation",
        "exchange",
        "interest",
        "investment",
        "return",
        "tax",
        "economy",
        "superannuation",
        "retirement"
    ]

    if any(marker in combined_text for marker in salary_markers):
        return "Salary sources"

    if any(marker in combined_text for marker in cost_markers):
        return "Cost of living sources"

    if any(marker in combined_text for marker in education_markers):
        return "Education sources"

    if any(marker in combined_text for marker in economic_markers):
        return "Economic sources"

    return "Assumptions only"


def get_dataset_country_name(dataset: Dict[str, Any], fallback: str = "Selected Country") -> str:
    metadata = dataset.get("metadata", {})

    return str(
        metadata.get("country")
        or metadata.get("country_name")
        or dataset.get("country")
        or fallback
    )


def has_unclean_source_references(dataset: Dict[str, Any]) -> bool:
    """
    Check whether the dataset has placeholder-style or non-URL source fields.
    """

    records = get_dataset_source_records(dataset)

    for record in records:
        source = record.get("source", "")

        if source_exists(source):
            if is_placeholder_source(source) or not is_clean_url(source):
                return True

    return False


def build_source_table(
    dataset: Dict[str, Any],
    country_name: str | None = None
) -> pd.DataFrame:
    """
    Build source reliability table from the selected country's dataset.

    Output columns:
        Country
        Variable
        Source
        Source type
        Source category
        Year
        Notes
        Reliability
    """

    selected_country = country_name or get_dataset_country_name(dataset)

    records = get_dataset_source_records(dataset)

    rows = []

    for record in records:
        section = record.get("section", "")
        variable = record.get("variable", "")
        source = record.get("source", "")
        notes = record.get("notes", "")

        full_variable_name = f"{section} → {variable}"

        rows.append(
            {
                "Country": selected_country,
                "Variable": full_variable_name,
                "Source": source,
                "Source type": classify_source_type(source),
                "Source category": classify_source_category(
                    variable=full_variable_name,
                    source=source,
                    notes=notes
                ),
                "Year": record.get("year", ""),
                "Notes": notes,
                "Reliability": calculate_source_reliability(
                    source=source,
                    notes=notes
                )
            }
        )

    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def filter_source_table(
    source_df: pd.DataFrame,
    selected_reliability: list,
    selected_source_types: list,
    selected_source_categories: list,
    search_text: str
) -> pd.DataFrame:
    """
    Apply filters to the selected country's source reliability table.
    """

    filtered_df = source_df.copy()

    if selected_reliability:
        filtered_df = filtered_df[
            filtered_df["Reliability"].isin(selected_reliability)
        ]

    if selected_source_types:
        filtered_df = filtered_df[
            filtered_df["Source type"].isin(selected_source_types)
        ]

    if selected_source_categories and "Source category" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Source category"].isin(selected_source_categories)
        ]

    if search_text:
        search_text = search_text.lower().strip()

        filtered_df = filtered_df[
            filtered_df.astype(str)
            .apply(
                lambda row: row.str.lower().str.contains(
                    search_text,
                    regex=False
                ).any(),
                axis=1
            )
        ]

    return filtered_df


def render_source_explorer(
    source_df: pd.DataFrame,
    show_unclean_source_warning: bool = False,
    selected_country: str | None = None
) -> None:
    """
    Render source reliability page inside Streamlit.
    """

    title = "Source Reliability Explorer"

    if selected_country:
        title = f"Source Reliability Explorer - {selected_country}"

    st.subheader(title)

    st.caption(
        "This table checks selected-country source quality without changing the JSON file."
    )

    if show_unclean_source_warning:
        st.warning(
            "Some source fields are not clean URLs. Replace them later for final submission."
        )

    if source_df.empty:
        st.info("No source records found in the selected country dataset.")
        return

    high_count = int((source_df["Reliability"] == "High").sum())
    medium_count = int(
        source_df["Reliability"].isin(["Medium", "Low/Medium"]).sum()
    )
    low_count = int((source_df["Reliability"] == "Low").sum())

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    with metric_col_1:
        st.metric("Total source rows", len(source_df))

    with metric_col_2:
        st.metric("High reliability", high_count)

    with metric_col_3:
        st.metric("Medium reliability", medium_count)

    with metric_col_4:
        st.metric("Low reliability", low_count)

    reliability_options = sorted(
        source_df["Reliability"].dropna().astype(str).unique().tolist()
    )

    source_type_options = sorted(
        source_df["Source type"].dropna().astype(str).unique().tolist()
    )

    source_category_options = [
        option
        for option in SOURCE_CATEGORY_OPTIONS
        if option in source_df["Source category"].dropna().astype(str).unique().tolist()
    ]

    filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns([1, 1, 1, 2])

    with filter_col_1:
        selected_reliability = st.multiselect(
            "Reliability",
            reliability_options,
            default=[]
        )

    with filter_col_2:
        selected_source_types = st.multiselect(
            "Source type",
            source_type_options,
            default=[]
        )

    with filter_col_3:
        selected_source_categories = st.multiselect(
            "Source category",
            source_category_options,
            default=[]
        )

    with filter_col_4:
        search_text = st.text_input(
            "Search sources",
            placeholder="Example: visa, salary, rent, official, assumed"
        )

    filtered_df = filter_source_table(
        source_df=source_df,
        selected_reliability=selected_reliability,
        selected_source_types=selected_source_types,
        selected_source_categories=selected_source_categories,
        search_text=search_text
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
