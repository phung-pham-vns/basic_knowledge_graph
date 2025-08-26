import pandas as pd


def load_excel(
    file_path: str, sheet_name: str, ignored_column_names: list[str] = None
) -> list[str]:
    """Load Excel file and convert rows to text format."""
    if ignored_column_names is None:
        ignored_column_names = []

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    row_infos = []
    for _, row in df.iterrows():
        text_parts = []

        column_id = 0
        for column in df.columns:
            if column in ignored_column_names:
                continue

            if pd.notna(row[column]) and str(row[column]).strip():
                column_id += 1
                text_parts.append(f"{column_id}. `{column}`: {row[column]}")

        full_text = "\n".join(text_parts)

        row_infos.append(full_text)

    return row_infos
