# PDF Highlighter

A library for highlighting and annotating sentences in PDF documents using Large Language Models (LLM). It's made to help users identify and emphasize relevant sentences in PDF documents. Compatible with both OpenAI and Ollama libraries.

## Use cases

- **Finding Relevant Information**:
   - Highlight specific sentences in a PDF that are relevant to a user's question or input. For example, if a user asks, "What are the main findings?", the tool will highlight sentences in the PDF that answer this question.

- **Reviewing LLM-Generated Answers**:
   - If a user has received an answer from an LLM based on information in a PDF, they can use this tool to highlight the exact text in the PDF that supports the LLM's answer. This helps in verifying and understanding the context of the LLM's response.

## Features

- Highlight sentences in PDF documents based on user input.
- Optionally add comments to highlighted sentences.
- Supports both OpenAI and Ollama language models.
- Combine multiple PDFs into a single document with highlights and comments.
- Classes and methods are asynchronous, allowing for non-blocking operations.

## Requirements

- Python 3.7+ (tested with 3.10.13)
- Required Python packages (see [`requirements.txt`](requirements.txt))

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/lasseedfast/pdf-highlighter.git
    cd pdf-highlighter
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Add your OpenAI API key and/or LLM model details to the `.env` file:
        ```
        OPENAI_API_KEY=your_openai_api_key
        LLM_MODEL=your_llm_model
        ```
        You can also set the LLM model name when initializing the `LLM` or `Highlighter` class using the `model` parameter.

5. _If using Ollama_, make sure to install the [Ollama server](https://ollama.com) and download the model you want to use. Follow the instructions in the [Ollama documentation](https://github.com/ollama/ollama) for more details.

## Usage

### Command-Line Interface

You can use the command-line interface to highlight sentences in a PDF document.

```sh
python highlight_pdf.py --user_input "Your question or input text" --pdf_filename "path/to/your/document.pdf" --openai_key "your_openai_api_key" --comment
```

#### Arguments

- `--user_input`: The text input from the user to highlight in the PDFs.
- `--pdf_filename`: The PDF filename to process.
- `--silent`: Suppress warnings (optional).
- `--openai_key`: OpenAI API key (optional if set in `.env`).
- `--comment`: Include comments in the highlighted PDF (optional).
- `--data`: Data in JSON format (fields: text, pdf_filename, pages) (optional).

#### Example

```sh
python highlight_pdf.py --user_input "What is said about climate?" --pdf_filename "example_pdf_document.pdf" --comment --llm_model llama3.1
```

### Note on Long PDFs

If the PDF is long, the result will be better if the user provides the data containing filename, user_input, and pages. This helps the tool focus on specific parts of the document, improving the accuracy and relevance of the highlights.

#### Example using the data argument

```sh
python highlight_pdf.py --data '[{"user_input": "What is said about climate?", "pdf_filename": "example_pdf_document.pdf", "pages": [1, 2]}]'
```

#### Output

The highlighted PDF will be saved with `_highlighted` appended to the original filename.

### Use in Python Code

This [example](examples/single_pdf.py) demonstrates how to use the highlight tool to understand what text in the PDF is relevant for the original user input/question. 

### Use in Python Code with ChromaDB
If the user has previously used ChromaDB to query for relevant texts, they can use the tool to highlight the relevant text in the PDFs based on the user input/question.  
This [example](examples/data_from_chromadb.py) assumes that there is a ChromaDB instance with information, and that the filenames and pages where the text is found are stored as metadata in ChromaDB.


## Streamlit Example

A Streamlit example is provided in `example_streamlit_app.py` to demonstrate how to use the PDF highlighter tool in a web application.

### Running the Streamlit App

1. Ensure you have installed the required packages and set up the environment variables as described in the Installation section.
2. Install streamlit:
    ```sh
    pip install streamlit
    ```
3. Run the Streamlit app:
    ```sh
    streamlit run example_streamlit_app.py
    ```

#### Streamlit App Features

- Enter your question or input text.
- Upload a PDF file.
- Optionally, choose to add comments to the highlighted text.
- Click the "Highlight PDF" button to process the PDF.
- Preview the highlighted PDF in the sidebar.
- Download the highlighted PDF.

## API

### Highlighter Class

#### Methods

- `__init__(self, silent=False, openai_key=None, comment=False, llm_model=None, llm_temperature=0, llm_system_prompt=None, llm_num_ctx=None, llm_memory=True, llm_keep_alive=3600)`: Initializes the Highlighter class with the given parameters.
- `async highlight(self, user_input, docs=None, data=None, pdf_filename=None)`: Highlights sentences in the provided PDF documents based on the user input.
- `async get_sentences_with_llm(self, text, user_input)`: Uses the LLM to generate sentences from the text that should be highlighted based on the user input.
- `async annotate_pdf(self, user_input: str, filename: str, pages: list = None, extend_pages: bool = False)`: Annotates the PDF with highlighted sentences and optional comments.

### LLM Class

#### Methods

- `__init__(self, openai_key=False, model=None, temperature=0, system_prompt=None, num_ctx=None, memory=True, keep_alive=3600)`: Initializes the LLM class with the provided parameters. 
- `use_openai(self, key, model)`: Configures the class to use OpenAI for generating responses.
- `use_ollama(self, model)`: Configures the class to use Ollama for generating responses.
- `async generate(self, prompt)`: Asynchronously generates a response based on the provided prompt.

**Note:** The `num_ctx` parameter is set to 20000 by default, which may not be sufficient for all use cases. Adjust this value based on your specific requirements.

## Default Prompts

The default LLM prompts are stored in the [`prompts.yaml`](prompts.yaml) file. You can view and edit the prompts directly in this file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.