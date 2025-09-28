import asyncio
import gradio as gr
from typing import List, Dict, Any, AsyncGenerator, Tuple
from pprint import pformat
from src.core.workflow import initialize_workflow
from langgraph.graph import END


async def run_traced_workflow(inputs: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the workflow with streaming trace events.
    Yields trace messages and generation as they become available.
    """
    workflow, graphiti = await initialize_workflow()
    inputs["graphiti"] = graphiti
    trace: List[str] = []

    try:
        async for event in workflow.astream_events(inputs, version="v2"):
            kind = event["event"]
            update: Dict[str, Any] = {"trace": None, "generation": None}

            if kind == "on_chain_start":
                name = event["name"]
                # if name not in ["route_question", "decide_to_generate", "grade_generation_vs_documents_and_question"]:
                trace.append(f"🔧 **Step Started**: {name}")
                update["trace"] = "\n".join(trace)

            elif kind == "on_chain_end":
                name = event["name"]
                output = event["data"].get("output", "No output")
                trace.append(f"📋 **Step Completed**: {name}")
                update["trace"] = "\n".join(trace)

                if name == "answer_generation" and isinstance(output, dict):
                    update["generation"] = output.get("generation", "")

            elif kind == "on_chain_error":
                error_msg = f"❌ **Error in Step**: {event['name']}\n{str(event['error'])}"
                trace.append(error_msg)
                update["trace"] = "\n".join(trace)

            elif event["name"] == "__end__":
                update["generation"] = event["data"].get("generation", "No answer generated.")
                update["trace"] = "\n".join(trace)

            if update["trace"] or update["generation"]:
                yield update

    except Exception as e:
        trace.append(f"❌ **Workflow Error**: {str(e)}")
        yield {"trace": "\n".join(trace), "generation": "An error occurred during processing."}

    finally:
        await graphiti.close()


async def chatbot_response(
    message: str, history: List[List[str]]
) -> AsyncGenerator[Tuple[List[List[str]], str, str], None]:
    """
    Gradio chatbot handler with streaming trace for each event.
    Yields updated chat history, trace output, and clears input box.
    """
    if not message.strip():
        yield history, "Please enter a question.", ""
        return

    inputs = {
        "question": message,
        "n_documents": 3,
        "node_contents": [],
        "edge_contents": [],
    }

    new_history = history + [[message, ""]]
    current_response = ""
    current_trace = "Processing your question..."

    async for update in run_traced_workflow(inputs):
        trace = update.get("trace", current_trace)
        generation = update.get("generation", None)

        if generation:
            current_response = generation
            new_history[-1][1] = current_response

        current_trace = trace
        yield new_history, current_trace, ""

    new_history[-1][1] = current_response or "No response generated."
    yield new_history, current_trace or "No trace available.", ""


# Gradio Interface
with gr.Blocks(title="Durian Pest & Disease Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🥭 Durian Pest & Disease Assistant
        Ask questions about durian leaves, pests, or diseases, and see the workflow in action with real-time tracing!
        """
    )

    chatbot = gr.Chatbot(
        height=400,
        show_label=False,
        avatar_images=("user.png", "bot.png"),
        render_markdown=True,
        value=[],
        # streaming=True,  # Requires Gradio 4.0+
    )

    msg = gr.Textbox(
        placeholder="e.g., My young durian leaves are curling and look scorched at the edges—could that be leafhopper damage?",
        show_label=False,
        container=True,
    )

    with gr.Row():
        clear_btn = gr.Button("Clear", variant="secondary")
        submit_btn = gr.Button("Send", variant="primary")

    trace_output = gr.Markdown(
        label="Workflow Trace & Tools Used",
        value="**Trace will appear here as the workflow processes your question.**",
    )

    # Event handlers
    submit_btn.click(
        fn=chatbot_response,
        inputs=[msg, chatbot],
        outputs=[chatbot, trace_output, msg],
    )
    msg.submit(
        fn=chatbot_response,
        inputs=[msg, chatbot],
        outputs=[chatbot, trace_output, msg],
    )
    clear_btn.click(
        fn=lambda: ([], "**Trace cleared!**"),
        outputs=[chatbot, trace_output],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
