from typing import List, Optional, TypedDict, Annotated
import operator
from langchain_core.messages import HumanMessage, AIMessage


# State definition for the agent
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    context: Optional[str]
    original_query: Optional[str]
    current_query: Optional[str]
    retrieval_attempts: int
    max_retrieval_attempts: int
    reflection_feedback: Optional[str]
    should_refine: bool
    is_sufficient: bool
    is_complete: bool
    final_response: Optional[str]
