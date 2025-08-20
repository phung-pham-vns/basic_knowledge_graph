from .local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from .reduce_system_prompt import REDUCE_SYSTEM_PROMPT
from .map_system_prompt import MAP_SYSTEM_PROMPT
from .summarize_prompt import SUMMARIZE_PROMPT
from .graph_extraction_prompt import GRAPH_EXTRACTION_PROMPT
from .community_report_prompt import COMMUNITY_REPORT_PROMPT


def create_extraction_prompt(entity_types, input_text, tuple_delimiter=";"):
    prompt = GRAPH_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        input_text=input_text,
        tuple_delimiter=tuple_delimiter,
        record_delimiter="|",
        completion_delimiter="\n\n",
    )
    return prompt


def get_summarize_prompt(entity_name, description_list):
    return SUMMARIZE_PROMPT.format(
        entity_name=entity_name, description_list=description_list
    )


def get_map_system_prompt(context):
    return MAP_SYSTEM_PROMPT.format(context_data=context)


def get_reduce_system_prompt(report_data, response_type: str = "multiple paragraphs"):
    return REDUCE_SYSTEM_PROMPT.format(
        report_data=report_data, response_type=response_type
    )


def get_local_system_prompt(report_data, response_type: str = "multiple paragraphs"):
    return LOCAL_SEARCH_SYSTEM_PROMPT.format(
        context_data=report_data, response_type=response_type
    )


def get_summarize_community_prompt(nodes, relationships):
    input_text = f"""Entities

    {nodes}

    Relationships

    {relationships}
    """
    return COMMUNITY_REPORT_PROMPT.format(
        input_text=input_text,
    )
