def extract_json(input: str):
    return input.removeprefix("```json").removesuffix("```").strip()
