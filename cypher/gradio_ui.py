import sys
import gradio as gr
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from retrieve import chain, graph_client
from settings import settings


def gradio_qa_interface():
    """Create and return a Gradio interface for Q&A."""

    def answer_question(question):
        """Function to answer questions using the knowledge graph."""
        if not question.strip():
            return "⚠️ Please enter a question."

        try:
            # Get response from the chain
            response = chain.invoke({"query": question})
            return response["result"]
        except Exception as e:
            return f"❌ Error occurred: {str(e)}\n\nPlease check your database connection and API keys."

    # Example questions for quick access
    examples = [
        "Which disease in Thailand affects the most durian varieties?",
        "Which disease appears in more than two seasons in a year, and which seasons are they?",
        "Which disease usually appears on durian trees during the rainy season?",
        "If my tree has yellowing leaves, what disease could it be?",
        "Which disease appears on the most parts of the durian tree, and which parts are they?",
        "Bệnh nào ảnh hưởng đến nhiều loại giống cây trồng nhất?",
    ]

    # Create the Gradio interface
    with gr.Blocks(
        title="🌳 Durian Knowledge Graph Q&A",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            color: #1f77b4;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .header p {
            color: #666;
            font-size: 1.2rem;
        }
        """,
    ) as interface:

        # Header
        with gr.Row():
            gr.HTML(
                """
                <div class="header">
                    <h1>🌳 Durian Knowledge Graph Q&A</h1>
                    <p>Ask questions about durian diseases, symptoms, and crop management</p>
                </div>
            """
            )

        # Main Q&A section
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="❓ Ask Your Question",
                    placeholder="e.g., Which disease affects durian leaves during rainy season?",
                    lines=3,
                    max_lines=5,
                )

                submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")

                answer_output = gr.Textbox(label="💡 Answer", lines=8, max_lines=15, interactive=False)

            with gr.Column(scale=1):
                # Connection status
                try:
                    schema = graph_client.schema
                    status_html = f"""
                        <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 1rem; margin-bottom: 1rem;">
                            <h4>✅ Connected to Neo4j Database</h4>
                            <p><strong>Database:</strong> {settings.graph_db.graph_db_url}</p>
                            <p><strong>Model:</strong> {settings.llm.llm_model}</p>
                            <p><strong>Temperature:</strong> {settings.llm.llm_temperature}</p>
                        </div>
                    """
                except Exception as e:
                    status_html = f"""
                        <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 1rem; margin-bottom: 1rem;">
                            <h4>❌ Database Connection Failed</h4>
                            <p><strong>Error:</strong> {str(e)}</p>
                        </div>
                    """

                gr.HTML(status_html)

                # Example questions
                gr.HTML("<h4>💡 Example Questions</h4>")
                for i, example in enumerate(examples):
                    gr.HTML(
                        f"""
                        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 0.5rem; margin: 0.5rem 0; cursor: pointer;" 
                              onclick="document.querySelector('textarea[data-testid=\\'question_input\\']').value = '{example.replace("'", "\\'")}'">
                            {example}
                        </div>
                    """
                    )

        # Database schema section
        with gr.Row():
            with gr.Accordion("📊 Database Schema", open=False):
                try:
                    gr.Code(schema, language="cypher", label="Neo4j Schema")
                except:
                    gr.HTML("<p>Unable to load database schema</p>")

        # Footer
        gr.HTML(
            """
            <div style="text-align: center; color: #666; margin-top: 2rem; padding: 1rem; border-top: 1px solid #eee;">
                <p>Built with ❤️ using Gradio, LangChain, and Neo4j</p>
                <p>Durian Knowledge Graph Q&A System</p>
            </div>
        """
        )

        # Connect the submit button
        submit_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

        # Allow Enter key to submit
        question_input.submit(fn=answer_question, inputs=question_input, outputs=answer_output)

    return interface


def main():
    interface = gradio_qa_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, quiet=False)


if __name__ == "__main__":
    main()
