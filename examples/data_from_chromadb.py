import asyncio
from highlight_pdf import Highlighter
import chromadb
import ollama

# Initialize ChromaDB client
client = chromadb.Client()

# Define the query to fetch relevant text snippets and metadata from ChromaDB
query = "What is said about climate?"
model = "llama3.1"

# Perform the query on ChromaDB
results = client.query(query)

# Results might look like this:
# results = [
#     {
#         "metadatas": [[
#             {
#                 "pdf_filename": "example_pdf_document.pdf",
#                 "pages": [1]
#             }]],
#         "documents": [["<Text extracted from the PDF page>"]],
#         "ids": ["<ID of the document>"]
#     },
#     {
#         "metadatas": [[
#             {
#                 "pdf_filename": "another_pdf_document.pdf",
#                 "pages": [2, 3]
#             }]],
#         "documents": [["<Another text extracted from the PDF pages>"]],
#         "ids": ["<ID of another document>"]
#     }
# ]

# Ask a LLM a question about the text snippets
documents_string = "\n".join(results[0]["documents"])
answer = ollama.chat(
    query=f"{query}\Only use information from the texts below when answering the question!\n\nTexts:\n{documents_string}",
    model=model,
    options={"temperature": 0},
)["message"]["content"]

# Now you want to highlight relevant information in the PDFs to understand what the LLM is using!

# Each result from ChromaDB contains the PDF filename and the pages where the text is found
data = [
    {
        "user_input": query,
        "pdf_filename": result["metadatas"][0]["pdf_filename"],
        "pages": result["metadatas"][0].get("pages"),
    }
    for result in results
]

# Initialize the Highlighter
highlighter = Highlighter(
    model="llama3.1",
    comment=True,  # Enable comments to understand the context
)


# Define the main asynchronous function to highlight the PDFs
async def highlight_pdf():
    # Use the highlight method to highlight the relevant sentences in the PDFs
    highlighted_pdf_buffer = await highlighter.highlight(
        data=data, zero_indexed_pages=True  # Pages are zero-based (e.g., 0, 1, 2, ...)
    )

    # Save the highlighted PDF to a new file
    with open("highlighted_combined_documents.pdf", "wb") as f:
        f.write(highlighted_pdf_buffer.getbuffer())


# Run the main function using asyncio
asyncio.run(highlight_pdf())
