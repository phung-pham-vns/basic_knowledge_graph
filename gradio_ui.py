#!/usr/bin/env python3
"""
Gradio UI for Refactored Agentic Graph RAG with Generation Node
"""

import asyncio
import gradio as gr
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.agentic_rag import AgenticGraphRAG
from src.deps.models import ModelConfig, LLMType


class RefactoredRAGChatbot:
    """Chatbot interface for the refactored Agentic Graph RAG system."""

    def __init__(self):
        self.conversation_history = []
        self.session_stats = {"total_queries": 0, "total_attempts": 0, "avg_attempts": 0, "successful_queries": 0}
        self.current_rag = None
        self.current_config = None

    def _get_rag_instance(self, config_type: str) -> AgenticGraphRAG:
        """Get RAG instance with specified configuration."""
        config_map = {
            "default": ModelConfig.get_default_config(),
            "high_quality": ModelConfig.get_high_quality_config(),
            "fast": ModelConfig.get_fast_config(),
        }

        new_config = config_map.get(config_type, ModelConfig.get_default_config())

        # Create new instance if config changed
        if self.current_config != config_type:
            self.current_rag = AgenticGraphRAG(new_config)
            self.current_config = config_type

        return self.current_rag

    async def process_query(
        self, query: str, max_attempts: int = 3, config_type: str = "default", show_details: bool = False
    ) -> Tuple[str, str, str, str]:
        """Process a query through the refactored agentic RAG system."""

        if not query.strip():
            return "Please enter a valid query.", "", "", self._format_history()

        start_time = time.time()

        try:
            # Get RAG instance with specified config
            rag = self._get_rag_instance(config_type)

            # Generate unique thread ID
            thread_id = f"chat_{int(time.time())}"

            # Process the query
            result = await rag.query(query=query, max_retrieval_attempts=max_attempts, thread_id=thread_id)

            end_time = time.time()
            processing_time = end_time - start_time

            # Update conversation history
            self.conversation_history.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "response": result["response"],
                    "attempts": result["total_attempts"],
                    "was_sufficient": result["was_sufficient"],
                    "processing_time": processing_time,
                    "config": config_type,
                }
            )

            # Update session stats
            self._update_stats(result["total_attempts"], result["was_sufficient"])

            # Format response
            response = result["response"] or "I couldn't find relevant information to answer your query."

            # Format details if requested
            details = self._format_details(result, processing_time, config_type) if show_details else ""

            # Format current stats
            stats = self._format_stats()

            # Format conversation history
            history = self._format_history()

            return response, details, stats, history

        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            self.conversation_history.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "response": error_msg,
                    "attempts": 0,
                    "was_sufficient": False,
                    "processing_time": time.time() - start_time,
                    "config": config_type,
                }
            )
            return error_msg, "", self._format_stats(), self._format_history()

    def _update_stats(self, attempts: int, was_sufficient: bool):
        """Update session statistics."""
        self.session_stats["total_queries"] += 1
        self.session_stats["total_attempts"] += attempts

        if was_sufficient:
            self.session_stats["successful_queries"] += 1

        if self.session_stats["total_queries"] > 0:
            self.session_stats["avg_attempts"] = (
                self.session_stats["total_attempts"] / self.session_stats["total_queries"]
            )

    def _format_details(self, result: Dict[str, Any], processing_time: float, config_type: str) -> str:
        """Format detailed information about the query processing."""
        details = f"""
## 🔍 Query Processing Details

**Processing Time:** {processing_time:.2f} seconds
**Model Configuration:** {config_type.title()}
**Total Retrieval Attempts:** {result['total_attempts']}
**Results Sufficient:** {'✅ Yes' if result['was_sufficient'] else '❌ No'}

### 🤔 Final Reflection
{result['reflection_feedback'] or 'No reflection feedback available'}

### 📄 Retrieved Context Length
{len(result['final_context'])} characters

### 🔄 All Search Attempts
"""

        for i, context in enumerate(result["all_contexts"], 1):
            details += f"\n**Attempt {i}:** {len(context)} characters"

        details += f"""

### ⚙️ System Architecture
- **Organized Structure**: Prompts and models properly separated
- **Generation Node**: Specialized final answer creation  
- **Reflection Tools**: Quality assessment and query refinement
- **Model Factory**: Configurable LLM selection
"""

        return details

    def _format_stats(self) -> str:
        """Format session statistics."""
        stats = self.session_stats
        success_rate = (stats["successful_queries"] / stats["total_queries"] * 100) if stats["total_queries"] > 0 else 0

        return f"""
## 📊 Session Statistics

- **Total Queries:** {stats['total_queries']}
- **Successful Queries:** {stats['successful_queries']}
- **Success Rate:** {success_rate:.1f}%
- **Total Retrieval Attempts:** {stats['total_attempts']}
- **Average Attempts per Query:** {stats['avg_attempts']:.1f}
- **Current Configuration:** {self.current_config or 'None'}
"""

    def _format_history(self) -> str:
        """Format conversation history."""
        if not self.conversation_history:
            return "No conversation history yet."

        history = "## 💬 Conversation History\n\n"

        for i, entry in enumerate(reversed(self.conversation_history[-10:]), 1):  # Show last 10
            status_icon = "✅" if entry["was_sufficient"] else "⚠️"
            config_badge = f"[{entry.get('config', 'default').upper()}]"
            history += f"""
### {status_icon} Query {len(self.conversation_history) - i + 1} - {entry['timestamp']} {config_badge}
**Q:** {entry['query'][:100]}{'...' if len(entry['query']) > 100 else ''}
**A:** {entry['response'][:150]}{'...' if len(entry['response']) > 150 else ''}
**Attempts:** {entry['attempts']} | **Time:** {entry['processing_time']:.1f}s

---
"""

        return history

    def clear_history(self):
        """Clear conversation history and stats."""
        self.conversation_history = []
        self.session_stats = {"total_queries": 0, "total_attempts": 0, "avg_attempts": 0, "successful_queries": 0}
        return "", "", self._format_stats(), "Conversation history cleared."


# Initialize the chatbot
chatbot = RefactoredRAGChatbot()


def sync_process_query(
    query: str, max_attempts: int, config_type: str, show_details: bool
) -> Tuple[str, str, str, str]:
    """Synchronous wrapper for the async query processing."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(chatbot.process_query(query, max_attempts, config_type, show_details))


def clear_conversation():
    """Clear conversation wrapper."""
    return chatbot.clear_history()


def create_refactored_interface():
    """Create the refactored Gradio interface."""

    with gr.Blocks(
        title="🧠 Refactored Agentic Graph RAG",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 30px; }
        .query-box { font-size: 16px; }
        .response-box { font-size: 14px; line-height: 1.6; }
        .details-box { font-size: 12px; }
        .stats-box { font-size: 13px; }
        .config-info { background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0; }
        """,
    ) as interface:

        # Header
        gr.HTML(
            """
        <div class="main-header">
            <h1>🧠 Refactored Agentic Graph RAG with Generation Node</h1>
            <p>Enhanced system with organized architecture, specialized generation, and configurable models</p>
            <div class="config-info">
                <strong>✨ New Features:</strong> Generation Node • Organized Prompts • Model Factory • Enhanced Structure
            </div>
        </div>
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Query input
                query_input = gr.Textbox(
                    label="🔍 Your Question",
                    placeholder="e.g., What causes durian leaf curling and how to treat it?",
                    lines=3,
                    elem_classes=["query-box"],
                )

                with gr.Row():
                    # Configuration options
                    max_attempts = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Max Retrieval Attempts",
                        info="Maximum number of search and refinement attempts",
                    )

                    config_type = gr.Dropdown(
                        choices=["default", "high_quality", "fast"],
                        value="default",
                        label="Model Configuration",
                        info="Choose model quality vs speed tradeoff",
                    )

                with gr.Row():
                    show_details = gr.Checkbox(
                        label="Show Processing Details",
                        value=False,
                        info="Display reflection feedback and system architecture details",
                    )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Ask Question", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear History", variant="secondary")

            with gr.Column(scale=1):
                # Model configuration info
                gr.HTML(
                    """
                <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                    <h4>⚙️ Model Configurations</h4>
                    <p><strong>Default:</strong> Balanced performance</p>
                    <p><strong>High Quality:</strong> Best results (slower)</p>
                    <p><strong>Fast:</strong> Quick responses (lower quality)</p>
                </div>
                """
                )

                # Session statistics
                stats_display = gr.Markdown(
                    value=chatbot._format_stats(), label="📊 Session Stats", elem_classes=["stats-box"]
                )

        # Response area
        with gr.Row():
            with gr.Column():
                response_output = gr.Markdown(label="🤖 Generated Response", elem_classes=["response-box"])

        # Details area (collapsible)
        with gr.Accordion("🔍 Processing Details & Architecture", open=False):
            details_output = gr.Markdown(elem_classes=["details-box"])

        # Conversation history
        with gr.Accordion("💬 Conversation History", open=False):
            history_output = gr.Markdown(value=chatbot._format_history())

        # Architecture information
        with gr.Accordion("🏗️ System Architecture", open=False):
            gr.HTML(
                """
            <div style="padding: 15px;">
                <h4>Enhanced Architecture Components:</h4>
                <ul>
                    <li><strong>Generation Node:</strong> Specialized final answer creation with comprehensive prompts</li>
                    <li><strong>Organized Prompts:</strong> Separated reflection and generation prompts in <code>graphiti/prompts/</code></li>
                    <li><strong>Model Factory:</strong> Configurable LLM management in <code>graphiti/deps/models/</code></li>
                    <li><strong>Modular Tools:</strong> Search, generation, and reflection tools in <code>graphiti/core/tools/</code></li>
                    <li><strong>Workflow Nodes:</strong> Agent, search, generation, and refinement nodes in <code>graphiti/core/nodes/</code></li>
                </ul>
                <h4>Benefits:</h4>
                <ul>
                    <li>✅ Better code organization and maintainability</li>
                    <li>✅ Specialized models for different tasks</li>
                    <li>✅ Enhanced answer quality with generation node</li>
                    <li>✅ Configurable performance vs quality tradeoffs</li>
                    <li>✅ Improved error handling and debugging</li>
                </ul>
            </div>
            """
            )

        # Example queries
        with gr.Accordion("💡 Example Queries", open=False):
            examples = gr.Examples(
                examples=[
                    "What causes durian leaf curling and how to treat it?",
                    "How do I prevent pest infestations in durian trees?",
                    "What are the symptoms of durian root rot disease?",
                    "How to identify and treat durian fruit fly damage?",
                    "What fungicides are effective for durian leaf spot?",
                    "How to manage durian tree nutrition deficiencies?",
                ],
                inputs=[query_input],
            )

        # Event handlers
        submit_btn.click(
            fn=sync_process_query,
            inputs=[query_input, max_attempts, config_type, show_details],
            outputs=[response_output, details_output, stats_display, history_output],
            show_progress=True,
        )

        clear_btn.click(fn=clear_conversation, outputs=[response_output, details_output, stats_display, history_output])

        # Allow Enter key to submit
        query_input.submit(
            fn=sync_process_query,
            inputs=[query_input, max_attempts, config_type, show_details],
            outputs=[response_output, details_output, stats_display, history_output],
            show_progress=True,
        )

    return interface


def main():
    """Main function to launch the refactored Gradio interface."""
    print("🚀 Starting Refactored Agentic Graph RAG Chatbot...")
    print("🏗️ New Architecture: Generation Node + Organized Structure")
    print("📊 Initializing enhanced system components...")

    # Create and launch the interface
    interface = create_refactored_interface()

    print("✅ Refactored system ready!")
    print("🌐 Launching enhanced web interface...")

    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7862,  # Different port for refactored version
        share=False,  # Set to True for public sharing
        show_error=True,  # Show detailed errors
        quiet=False,  # Show startup logs
        inbrowser=True,  # Open browser automatically
        favicon_path=None,  # Custom favicon
        ssl_verify=False,  # For development
    )


if __name__ == "__main__":
    main()
