#!/usr/bin/env python3
"""
Refactored Agentic Graph RAG with Reflection and Generation
"""

import asyncio
import json
from typing import List, Optional, Dict, Any, TypedDict, Annotated
from dataclasses import dataclass
import operator

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import organized modules
from src.deps.models import ModelConfig, get_llm
from src.core.config import SearchResults, SearchComponentConfig, RerankerType, create_search_config
from src.core.types import AgentState
from src.core.nodes import agent_node, search_node, generation_node, refinement_node
from src.prompts.generation_prompts import get_system_prompt


class AgenticGraphRAG:
    """Main class for Agentic Graph RAG with reflection and generation."""

    def __init__(self, model_config: ModelConfig = None):
        self.model_config = model_config or ModelConfig.get_default_config()
        self.app = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow."""

        # Create workflow nodes with model config
        async def agent(state: AgentState) -> AgentState:
            return await agent_node(state, self.model_config)

        async def search(state: AgentState) -> AgentState:
            return await search_node(state, self.model_config)

        async def generate(state: AgentState) -> AgentState:
            return await generation_node(state, self.model_config)

        async def refine(state: AgentState) -> AgentState:
            return await refinement_node(state)

        async def generate_final(state: AgentState) -> AgentState:
            """Node that triggers final answer generation."""
            # Import here to avoid circular dependency
            from src.core.tools import generate_final_answer

            # Get the context and query from state
            context = state.get("context", "")
            original_query = state.get("original_query", "")
            attempts = state.get("retrieval_attempts", 1)
            reflection_feedback = state.get("reflection_feedback", "")

            # Extract query from first human message if not in state
            if not original_query and state.get("messages"):
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage):
                        original_query = msg.content
                        break

            # Generate final answer
            try:
                final_response = await generate_final_answer.ainvoke(
                    {
                        "query": original_query,
                        "context": context,
                        "attempts": attempts,
                        "reflection_feedback": reflection_feedback,
                        "llm_config": self.model_config,
                    }
                )

                return {
                    "final_response": final_response,
                    "is_complete": True,
                    "context": state.get("context", ""),
                    "retrieval_attempts": state.get("retrieval_attempts", 1),
                    "is_sufficient": state.get("is_sufficient", True),
                    "reflection_feedback": state.get("reflection_feedback", ""),
                }
            except Exception as e:
                # Fallback if generation fails
                return {
                    "final_response": f"Based on the retrieved information: {context[:500]}...",
                    "is_complete": True,
                    "context": state.get("context", ""),
                    "retrieval_attempts": state.get("retrieval_attempts", 1),
                    "is_sufficient": state.get("is_sufficient", True),
                    "reflection_feedback": state.get("reflection_feedback", ""),
                }

        # Router function
        def router(state: AgentState) -> str:
            last_message = state["messages"][-1]

            # If there are tool calls, determine which tool
            if last_message.tool_calls:
                tool_name = last_message.tool_calls[0]["name"]
                if tool_name == "search_knowledge_graph":
                    return "search"
                elif tool_name == "generate_final_answer":
                    return "generate"

            # Check if we should refine and retry
            should_refine = state.get("should_refine", False)
            is_sufficient = state.get("is_sufficient", True)
            is_complete = state.get("is_complete", False)

            if is_complete:
                return "end"
            elif should_refine and not is_sufficient:
                return "refine"
            else:
                return "end"

        # Router for search node - decides next step after search
        def search_router(state: AgentState) -> str:
            should_refine = state.get("should_refine", False)
            is_sufficient = state.get("is_sufficient", True)

            if should_refine and not is_sufficient:
                return "refine"
            elif is_sufficient:
                return "generate_final"
            else:
                return "end"

        # Build the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent)
        workflow.add_node("search", search)
        workflow.add_node("generate", generate)
        workflow.add_node("generate_final", generate_final)
        workflow.add_node("refine", refine)

        workflow.add_edge("agent", "search")
        workflow.add_conditional_edges(
            "search", search_router, {"generate_final": "generate_final", "refine": "refine", "end": END}
        )
        workflow.add_conditional_edges("generate", router, {"end": END})
        workflow.add_edge("generate_final", END)
        workflow.add_edge("refine", "agent")
        workflow.set_entry_point("agent")

        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def query(
        self,
        query: str,
        max_retrieval_attempts: int = 3,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process a query through the agentic RAG system.

        Args:
            query: User's question
            max_retrieval_attempts: Maximum number of retrieval attempts
            thread_id: Thread ID for conversation memory

        Returns:
            Dict containing response and metadata
        """
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "context": None,
            "original_query": None,
            "current_query": None,
            "retrieval_attempts": 0,
            "max_retrieval_attempts": max_retrieval_attempts,
            "reflection_feedback": None,
            "should_refine": False,
            "is_sufficient": False,
            "is_complete": False,
            "final_response": None,
        }

        final_state = None
        all_contexts = []

        async for event in self.app.astream(initial_state, config):
            final_state = list(event.values())[-1]
            if final_state.get("context"):
                all_contexts.append(final_state["context"])

        # Extract final response
        final_response = ""
        if final_state:
            if final_state.get("final_response"):
                final_response = final_state["final_response"]
            elif final_state.get("messages"):
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
                        final_response = msg.content
                        break

        return {
            "response": final_response,
            "total_attempts": final_state.get("retrieval_attempts", 0) if final_state else 0,
            "final_context": final_state.get("context", "") if final_state else "",
            "all_contexts": all_contexts,
            "reflection_feedback": final_state.get("reflection_feedback", "") if final_state else "",
            "was_sufficient": final_state.get("is_sufficient", False) if final_state else False,
        }


# Global instance for backward compatibility
_default_rag = None


def get_default_rag() -> AgenticGraphRAG:
    """Get the default RAG instance."""
    global _default_rag
    if _default_rag is None:
        _default_rag = AgenticGraphRAG()
    return _default_rag


async def agentic_graph_rag_query(
    query: str, max_retrieval_attempts: int = 3, thread_id: str = "default", model_config: ModelConfig = None
) -> Dict[str, Any]:
    """
    Convenience function for backward compatibility.
    """
    if model_config:
        rag = AgenticGraphRAG(model_config)
        return await rag.query(query, max_retrieval_attempts, thread_id)
    else:
        return await get_default_rag().query(query, max_retrieval_attempts, thread_id)


# Demo function
async def demo():
    """Demo the refactored system."""
    print("🧠 Starting Refactored Agentic Graph RAG with Generation Node...")

    # Create RAG instance with custom config
    config = ModelConfig.get_default_config()
    rag = AgenticGraphRAG(config)

    query = "What causes durian leaf curling and how to treat it?"
    print(f"Query: {query}")
    print("-" * 80)

    result = await rag.query(query=query, max_retrieval_attempts=3, thread_id="demo_refactored")

    print(f"✅ Response: {result['response']}")
    print(f"📊 Total attempts: {result['total_attempts']}")
    print(f"🎯 Was sufficient: {result['was_sufficient']}")
    if result["reflection_feedback"]:
        print(f"🤔 Reflection: {result['reflection_feedback'][:100]}...")


if __name__ == "__main__":
    asyncio.run(demo())
