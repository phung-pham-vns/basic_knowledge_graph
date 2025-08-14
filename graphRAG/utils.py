import hashlib
import json


def get_hash(text):
    # Convert text to bytes if it's a string
    if isinstance(text, str):
        text = text.encode("utf-8")

    # MD5 is one of the fastest algorithms
    return hashlib.md5(text).hexdigest()


def parse_extraction_output(output_str, record_delimiter=None, tuple_delimiter=None):
    """
    Parse a structured output string containing "entity" and "relationship" records into a list of dictionaries.

    The expected format for each record is:

        ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

    or

        ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

    Records are separated by a record delimiter. The output string may end with a completion marker
    (for example, "{completion_delimiter}") which will be removed.

    If not provided, this function attempts to auto-detect:
      - record_delimiter: looks for "{record_delimiter}" then "|" then falls back to newlines.
      - tuple_delimiter: looks for "{tuple_delimiter}" then ";" then falls back to a tab.

    Parameters:
        output_str (str): The complete string output.
        record_delimiter (str, optional): The delimiter that separates records.
        tuple_delimiter (str, optional): The delimiter that separates fields within a record.

    Returns:
        List[dict]: A list of dictionaries where each dictionary represents an entity or relationship.

        For an "entity", the dictionary has the keys:
            - record_type (always "entity")
            - entity_name
            - entity_type
            - entity_description

        For a "relationship", the dictionary has the keys:
            - record_type (always "relationship")
            - source_entity
            - target_entity
            - relationship_description
            - relationship_strength (as an int or float)
    """
    # Remove the completion delimiter if present.
    completion_marker = "{completion_delimiter}"
    if completion_marker in output_str:
        output_str = output_str.replace(completion_marker, "")
    output_str = output_str.strip()

    # Determine the record delimiter if not provided.
    if record_delimiter is None:
        if "{record_delimiter}" in output_str:
            record_delimiter = "{record_delimiter}"
        elif "|" in output_str:
            record_delimiter = "|"
        else:
            # Fallback: split on newlines
            record_delimiter = "\n"

    # Determine the tuple delimiter if not provided.
    if tuple_delimiter is None:
        if "{tuple_delimiter}" in output_str:
            tuple_delimiter = "{tuple_delimiter}"
        elif ";" in output_str:
            tuple_delimiter = ";"
        else:
            tuple_delimiter = "\t"

    # Split the output into individual record strings.
    raw_records = [r.strip() for r in output_str.split(record_delimiter)]

    parsed_records = []
    for rec in raw_records:
        if not rec:
            continue  # skip empty strings

        # Remove leading/trailing parentheses if present.
        if rec.startswith("(") and rec.endswith(")"):
            rec = rec[1:-1]
        rec = rec.strip()

        # Split the record into tokens using the tuple delimiter.
        tokens = [token.strip() for token in rec.split(tuple_delimiter)]
        if not tokens:
            continue

        # The first token should be either "entity" or "relationship".
        rec_type = tokens[0].strip(" \"'").lower()

        if rec_type == "entity":
            if len(tokens) != 4:
                # Optionally log or raise an error for malformed records.
                continue
            record = {
                "record_type": "entity",
                "entity_name": tokens[1],
                "entity_type": tokens[2],
                "entity_description": tokens[3],
            }
            parsed_records.append(record)
        elif rec_type == "relationship":
            if len(tokens) != 5:
                continue
            # Attempt to convert relationship_strength to a number.
            try:
                strength = float(tokens[4])
                # Convert to int if it has no fractional part.
                if strength.is_integer():
                    strength = int(strength)
            except ValueError:
                strength = tokens[4]
            record = {
                "record_type": "relationship",
                "source_entity": tokens[1],
                "target_entity": tokens[2],
                "relationship_description": tokens[3],
                "relationship_strength": strength,
            }
            parsed_records.append(record)
        else:
            # Unknown record type; skip it or handle accordingly.
            continue
    nodes = [el for el in parsed_records if el.get("record_type") == "entity"]
    relationships = [
        el for el in parsed_records if el.get("record_type") == "relationship"
    ]
    return nodes, relationships


def extract_json(input: str):
    return json.loads(input.removeprefix("```json").removesuffix("```").strip())
