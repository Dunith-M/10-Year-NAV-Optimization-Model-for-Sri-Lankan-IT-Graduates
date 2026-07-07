from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.data_loader import get_dataset_assumption_records
from src.source_explorer import calculate_source_reliability


ASSUMPTION_COLUMNS = [
    "Country",
    "Section",
    "Variable",
    "Value",
    "Currency / Unit",
    "Year",
    "Source",
    "Notes",
    "Reliability"
]


def get_dataset_country_name(dataset: Dict[str, Any], fallback: str = "Selected Country") -> str:
    metadata = dataset.get("metadata", {})

    return str(
        metadata.get("country")
        or metadata.get("country_name")
        or dataset.get("country")
        or fallback
    )


def build_assumption_table(
    dataset: Dict[str, Any],
    country_name: str | None = None
) -> pd.DataFrame:
    """
    Build a clean assumption table from the selected country's nested dataset only.

    Output columns:
        Country
        Section
        Variable
        Value
        Currency / Unit
        Year
        Source
        Notes
        Reliability
    """

    selected_country = country_name or get_dataset_country_name(dataset)

    records = get_dataset_assumption_records(dataset)

    rows = []

    for record in records:
        source = record.get("source", "")
        notes = record.get("notes", "")

        rows.append(
            {
                "Country": selected_country,
                "Section": record.get("section", ""),
                "Variable": record.get("variable", ""),
                "Value": record.get("value", ""),
                "Currency / Unit": record.get("currency_or_unit", ""),
                "Year": record.get("year", ""),
                "Source": source,
                "Notes": notes,
                "Reliability": calculate_source_reliability(
                    source=source,
                    notes=notes
                )
            }
        )

    return pd.DataFrame(rows, columns=ASSUMPTION_COLUMNS)


def filter_assumption_table(
    assumption_df: pd.DataFrame,
    selected_sections: list,
    selected_reliability: list,
    search_text: str
) -> pd.DataFrame:
    """
    Apply section, reliability, and text filters to the selected country's assumption table.
    """

    filtered_df = assumption_df.copy()

    if selected_sections and "Section" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Section"].isin(selected_sections)
        ]

    if selected_reliability and "Reliability" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Reliability"].isin(selected_reliability)
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


def render_assumption_explorer(
    assumption_df: pd.DataFrame,
    selected_country: str | None = None
) -> None:
    """
    Render the selected country's assumption explorer inside Streamlit.
    """

    title = "Full Assumption Explorer"

    if selected_country:
        title = f"Full Assumption Explorer - {selected_country}"

    st.subheader(title)

    st.caption(
        "This table exposes the selected country's important dataset assumptions without changing the JSON file."
    )

    if assumption_df.empty:
        st.info("No assumptions found in the selected country dataset.")
        return

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

    with metric_col_1:
        st.metric("Total assumption rows", len(assumption_df))

    with metric_col_2:
        st.metric("Dataset sections", assumption_df["Section"].nunique())

    with metric_col_3:
        if "Reliability" in assumption_df.columns:
            low_count = int((assumption_df["Reliability"] == "Low").sum())
            st.metric("Low reliability rows", low_count)
        else:
            st.metric("Low reliability rows", "N/A")

    section_options = sorted(
        assumption_df["Section"].dropna().astype(str).unique().tolist()
    )

    reliability_options = sorted(
        assumption_df["Reliability"].dropna().astype(str).unique().tolist()
    ) if "Reliability" in assumption_df.columns else []

    filter_col_1, filter_col_2, filter_col_3 = st.columns([1, 1, 2])

    with filter_col_1:
        selected_sections = st.multiselect(
            "Filter by section",
            section_options,
            default=[]
        )

    with filter_col_2:
        selected_reliability = st.multiselect(
            "Filter by reliability",
            reliability_options,
            default=[]
        )

    with filter_col_3:
        search_text = st.text_input(
            "Search assumptions",
            placeholder="Example: salary, rent, tuition, visa, inflation"
        )

    filtered_df = filter_assumption_table(
        assumption_df=assumption_df,
        selected_sections=selected_sections,
        selected_reliability=selected_reliability,
        search_text=search_text
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
