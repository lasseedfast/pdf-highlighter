import asyncio
import io
from highlight_pdf import Highlighter

# User input/question
user_input = "What are the main findings?"

# Answer received from LLM based on text in a PDF
llm_answer = "The main findings are that the treatment was effective in 70% of cases."

# PDF filename
pdf_filename = "example_pdf_document.pdf"

# Pages to consider (optional, can be None)
pages = [1, 2]

# Initialize the Highlighter
highlighter = Highlighter(
    model='llama3.1',
    comment=True  # Enable comments to understand the context
)

# Define the main asynchronous function to highlight the PDF
async def main():
    highlighted_pdf_buffer = await highlighter.highlight(
        user_input=user_input,
        data=[{"text": llm_answer, "pdf_filename": pdf_filename, "pages": pages}]
    )
    
    # Save the highlighted PDF to a new file
    with open("highlighted_example_pdf_document.pdf", "wb") as f:
        f.write(highlighted_pdf_buffer.getbuffer())

# Run the main function using asyncio
asyncio.run(main())