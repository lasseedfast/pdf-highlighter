import re
import warnings
import pymupdf
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import io
import dotenv
import os
import asyncio
import aiofiles
import yaml

# Check if 'punkt_tab' tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    import logging

    logging.info("Downloading 'punkt_tab' tokenizer data for NLTK.")
    nltk.download("punkt_tab")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the prompts.yaml file
prompts_path = os.path.join(script_dir, "prompts.yaml")

# Load prompts from configuration file
with open(prompts_path, "r") as file:
    prompts = yaml.safe_load(file)

CUSTOM_SYSTEM_PROMPT = prompts["CUSTOM_SYSTEM_PROMPT"]
GET_SENTENCES_PROMPT = prompts["GET_SENTENCES_PROMPT"]
EXPLANATION_PROMPT = prompts["EXPLANATION_PROMPT"]


class LLM:
    """
    LLM class for interacting with language models from OpenAI or Ollama.

    Attributes:
        model (str): The model to be used for generating responses.
        num_ctx (int): The number of context tokens to be used. Defaults to 20000.
        temperature (float): The temperature setting for the model's response generation.
        keep_alive (int): The keep-alive duration for the connection.
        options (dict): Options for the model's response generation.
        memory (bool): Whether to retain conversation history.
        messages (list): List of messages in the conversation.
        openai (bool): Flag indicating if OpenAI is being used.
        ollama (bool): Flag indicating if Ollama is being used.
        client (object): The client object for OpenAI.
        llm (object): The client object for the language model.

    Methods:
        __init__(openai_key=False, model=None, temperature=0, system_prompt=None, num_ctx=None, memory=True, keep_alive=3600):
            Initializes the LLM class with the provided parameters.
        use_openai(key, model):
            Configures the class to use OpenAI for generating responses.
        use_ollama(model):
            Configures the class to use Ollama for generating responses.
        generate(prompt):
            Asynchronously generates a response based on the provided prompt.
    """

    def __init__(
        self,
        num_ctx=20000,
        openai_key=False,
        model=None,
        temperature=0,
        system_prompt=None,
        memory=True,
        keep_alive=3600,
    ):
        """
        Initialize the highlight_pdf class.

        Parameters:
        openai_key (str or bool): API key for OpenAI. If False, Ollama will be used.
        model (str, optional): The model to be used. Defaults to None.
        temperature (float, optional): Sampling temperature for the model. Defaults to 0.
        system_prompt (str, optional): Initial system prompt for the model. Defaults to None.
        context_window (int, optional): Number of context tokens. Defaults to None.
        memory (bool, optional): Whether to use memory. Defaults to True.
        keep_alive (int, optional): Keep-alive duration in seconds. Defaults to 3600.
        """
        
        if model:
            self.model = model
        else:
            self.model = os.getenv("LLM_MODEL")
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.options = {"temperature": self.temperature, num_ctx: num_ctx}
        self.memory = memory
        if system_prompt:
            self.messages = [{"role": "system", "content": system_prompt}]
        else:
            self.messages = [{"role": "system", "content": CUSTOM_SYSTEM_PROMPT}]

        # Check if OpenAI key is provided
        if openai_key:  # Use OpenAI
            self.use_openai(openai_key, model)
        elif os.getenv("OPENAI_API_KEY") != '':  # Use OpenAI
            self.use_openai(os.getenv("OPENAI_API_KEY"), model)
        else:  # Use Ollama
            self.use_ollama(model)

    def use_openai(self, key, model):
        """
        Configures the instance to use OpenAI's API for language model operations.

        Args:
            key (str): The API key for authenticating with OpenAI.
            model (str): The specific model to use. If not provided, it will default to the value of the "OPENAI_MODEL" environment variable.

        Attributes:
            llm (module): The OpenAI module.
            client (openai.AsyncOpenAI): The OpenAI client initialized with the provided API key.
            openai (bool): Flag indicating that OpenAI is being used.
            ollama (bool): Flag indicating that Ollama is not being used.
            model (str): The model to be used for OpenAI operations.
        """
        import openai

        self.llm = openai
        self.client = openai.AsyncOpenAI(api_key=key)
        self.openai = True
        self.ollama = False
        if model:
            self.model = model
        else:
            self.model = os.getenv("LLM_MODEL")

    def use_ollama(self, model):
        """
        Configures the instance to use the Ollama LLM (Language Learning Model) service.

        This method initializes an asynchronous Ollama client and sets the appropriate flags
        to indicate that Ollama is being used instead of OpenAI. It also sets the model to be
        used for the LLM, either from the provided argument or from an environment variable.

        Args:
            model (str): The name of the model to be used. If not provided, the model name
                         will be fetched from the environment variable 'LLM_MODEL'.
        """
        import ollama

        self.llm = ollama.AsyncClient()
        self.ollama = True
        self.openai = False
        if model:
            self.model = model
        else:
            self.model = os.getenv("LLM_MODEL")

    async def generate(self, prompt):
        """
        Generates a response based on the provided prompt using either OpenAI or Ollama.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response.

        Notes:
            - The prompt is stripped of leading whitespace on each line.
        """
        prompt = re.sub(r"^\s+", "", prompt, flags=re.MULTILINE)
        self.messages.append({"role": "user", "content": prompt})
        if self.openai:
            chat_completion = await self.client.chat.completions.create(
                messages=self.messages, model=self.model, temperature=0
            )
            answer = chat_completion.choices[0].message.content
            return answer
        elif self.ollama:
            response = await self.llm.chat(
                messages=self.messages,
                model=self.model,
                options=self.options,
                keep_alive=self.keep_alive,
            )
            answer = response["message"]["content"]

        self.messages.append({"role": "assistant", "content": answer})
        if not self.memory:
            self.messages = self.messages[0]
        return answer


class Highlighter:
    """
    Highlighter class for annotating and highlighting sentences in PDF documents using an LLM (Large Language Model).
    Attributes:
        silent (bool): Flag to suppress warnings.
        comment (bool): Flag to add comments to highlighted sentences.
        llm_params (dict): Parameters for the LLM.
    Methods:
        __init__(self, silent=False, openai_key=None, comments=False, llm_model=None, llm_temperature=0, llm_system_prompt=None, llm_num_ctx=None, llm_memory=True, llm_keep_alive=3600):
            Initializes the Highlighter class with the given parameters.
        async highlight(self, user_input, docs=None, data=None, pdf_filename=None):
            Highlights sentences in the provided PDF documents based on the user input.
        async get_sentences_with_llm(self, text, user_input):
            Uses the LLM to generate sentences from the text that should be highlighted based on the user input.
        async annotate_pdf(self, user_input: str, filename: str, pages: list = None, extend_pages: bool = False):
            Annotates the PDF with highlighted sentences and optional comments.
            Fixes the filename by replacing special characters with their ASCII equivalents.
    """

    def __init__(
        self,
        silent=False,
        openai_key=None,
        comment=False,
        llm_model=None,
        llm_temperature=0,
        llm_system_prompt=None,
        llm_num_ctx=None,
        llm_memory=True,
        llm_keep_alive=3600,
    ):
        """
        Initialize the class with the given parameters.

        Parameters:
        silent (bool): Flag to suppress output.
        openai_key (str or None): API key for OpenAI.
        comment (bool): Flag to enable or disable comments.
        llm_model (str or None): The model name for the language model.
        llm_temperature (float): The temperature setting for the language model.
        llm_system_prompt (str or None): The system prompt for the language model.
        llm_num_ctx (int or None): The number of context tokens for the language model.
        llm_memory (bool): Flag to enable or disable memory for the language model.
        llm_keep_alive (int): The keep-alive duration for the language model in seconds.
        """
        dotenv.load_dotenv()
        
        # Ensure both model are provided or set in the environment
        assert llm_model or os.getenv("LLM_MODEL"), "LLM_MODEL must be provided as argument or set in the environment."

        self.silent = silent
        self.comment = comment
        self.llm_params = {
            "openai_key": openai_key,
            "model": llm_model,
            "temperature": llm_temperature,
            "system_prompt": llm_system_prompt,
            "num_ctx": llm_num_ctx,
            "memory": llm_memory,
            "keep_alive": llm_keep_alive,
        }


    async def highlight(
        self,
        user_input=None,
        docs=None,
        data=None,
        pdf_filename=None,
        pages=None,
        zero_indexed_pages=False,
        pdf_buffer=None
    ):
        """
        Highlights text in one or more PDF documents based on user input.
        Args:
            user_input (str): The text input from the user to highlight in the PDFs. Defaults to None.
            docs (list, optional): A list of PDF filenames to process. Defaults to None.
            data (list, optional): Data in JSON format to process. Should be on the format: [{"user_input": "text", "pdf_filename": "filename", "pages": [1, 2, 3]}]. Defaults to None.
            pdf_filename (str, optional): A single PDF filename to process. Defaults to None.
            pages (list, optional): A list of page numbers to process. Defaults to None.
            zero_indexed_pages (bool, optional): Flag to indicate if the page numbers are zero-indexed. Defaults to False.
            pdf_buffer (io.BytesIO, optional): A buffer containing the PDF that should be highlighted. #! 
        Returns:
            io.BytesIO: A buffer containing the combined PDF with highlights.
        Raises:
            AssertionError: If none of `data`, `pdf_filename`, or `docs` are provided.
        """
        pdf_buffers = []
        assert any(
            [data, pdf_filename, docs, pdf_buffer]
        ), "You need to provide either a PDF filename, a list of filenames or data in JSON format."

        if data:
            user_input = [item["user_input"] for item in data]
            docs = [item["pdf_filename"] for item in data]
            pages = [item.get("pages") for item in data]
            if not zero_indexed_pages:
                pages = [[p - 1 for p in page] for page in pages]


        if not docs and any([pdf_filename, pdf_buffer]):
            user_input = [user_input]
            docs = [pdf_filename if pdf_filename else pdf_buffer]
            pages = [pages]

        tasks = [
            self.annotate_pdf(ui, doc, pages=pg)
            for ui, doc, pg in zip(user_input, docs, pages or [pages] * len(docs))
        ]
        pdf_buffers = await asyncio.gather(*tasks)

        combined_pdf = pymupdf.open()
        new_toc = []

        for buffer in pdf_buffers:
            if not buffer:
                continue
            pdf = pymupdf.open(stream=buffer, filetype="pdf")
            length = len(combined_pdf)
            combined_pdf.insert_pdf(pdf)
            new_toc.append([1, f"Document {length + 1}", length + 1])

        combined_pdf.set_toc(new_toc)
        pdf_buffer = io.BytesIO()
        combined_pdf.save(pdf_buffer)
        pdf_buffer.seek(0)

        return pdf_buffer

    async def get_sentences_with_llm(self, text, user_input):
        prompt = GET_SENTENCES_PROMPT.format(text=text, user_input=user_input)

        answer = await self.llm.generate(prompt)
        return answer.split("\n")

    async def annotate_pdf(
        self,
        user_input: str,
        pdf_file: str,
        pages: list = None,
        extend_pages: bool = False,
    ):
        self.llm = LLM(**self.llm_params)

        if not isinstance(pdf_file, io.BytesIO):
            pdf = pymupdf.open(pdf_file)
        else:
            pdf = pymupdf.open(stream=pdf_file, filetype="pdf")
        output_pdf = pymupdf.open()
        vectorizer = TfidfVectorizer()

        if pages is not None:
            new_pdf = pymupdf.open()
            pdf_pages = pdf.pages(pages[0], pages[-1] + 1)
            pdf_text = ""
            for page in pdf_pages:
                pdf_text += f'\n{page.get_text("text")}'
                new_pdf.insert_pdf(pdf, from_page=page.number, to_page=page.number)
        else:
            pdf_text = "\n".join([page.get_text("text") for page in pdf])
            new_pdf = pymupdf.open()
            new_pdf.insert_pdf(pdf)

        pdf_sentences = nltk.sent_tokenize(pdf_text)
        tfidf_text = vectorizer.fit_transform(pdf_sentences)
        sentences = await self.get_sentences_with_llm(pdf_text, user_input)
        highlight_sentences = []
        for sentence in sentences:
            if sentence == "None" or len(sentence) < 5:
                continue

            sentence = sentence.replace('"', "").strip()
            if sentence in pdf_text:
                highlight_sentences.append(sentence)
            else:
                tfidf_sentence = vectorizer.transform([sentence])
                cosine_similarities = linear_kernel(
                    tfidf_sentence, tfidf_text
                ).flatten()
                most_similar_index = cosine_similarities.argmax()
                most_similar_sentence = pdf_sentences[most_similar_index]
                highlight_sentences.append(most_similar_sentence)

        relevant_pages = set()

        for sentence in highlight_sentences:
            found = False
            if self.comment:
                explanation = await self.llm.generate(
                    EXPLANATION_PROMPT.format(sentence=sentence, user_input=user_input)
                )
            for page in new_pdf:
                rects = page.search_for(sentence)
                if not rects:
                    continue
                found = True
                p1 = rects[0].tl
                p2 = rects[-1].br
                highlight = page.add_highlight_annot(start=p1, stop=p2)
                if self.comment:
                    highlight.set_info(content=explanation)
                relevant_pages.add(page.number)
                new_pdf.reload_page(page)

            if not found and not self.silent:
                warnings.warn(f"Sentence not found: {sentence}", category=UserWarning)

        extended_pages = []
        if extend_pages:
            for p in relevant_pages:
                extended_pages.append(p)
                if p - 1 not in extended_pages and p - 1 != -1:
                    extended_pages.append(p - 1)
                if p + 1 not in extended_pages:
                    extended_pages.append(p + 1)
            relevant_pages = extended_pages
        for p in relevant_pages:
            output_pdf.insert_pdf(new_pdf, from_page=p, to_page=p)

        if len(output_pdf) != 0:
            buffer = io.BytesIO()
            new_pdf.save(buffer)
            buffer.seek(0)
            return buffer
        else:
            if not self.silent:
                warnings.warn("No relevant sentences found", category=UserWarning)
            return None


async def save_pdf_to_file(pdf_buffer, filename):
    async with aiofiles.open(filename, "wb") as f:
        await f.write(pdf_buffer.getbuffer())


if __name__ == "__main__":
    import argparse
    import json

    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(
        description=(
            "Highlight sentences in PDF documents using an LLM.\n\n"
            "For more information, visit: https://github.com/lasseedfast/pdf-highlighter/blob/main/README.md"
        )
    )
    parser.add_argument(
        "--user_input",
        type=str,
        help="The text input from the user to highlight in the PDFs.",
    )
    parser.add_argument("--pdf_filename", type=str, help="The PDF filename to process.")
    parser.add_argument("--silent", action="store_true", help="Suppress warnings.")
    parser.add_argument("--openai_key", type=str, help="API key for OpenAI.")
    parser.add_argument("--llm_model", type=str, help="The model name for the language model.")
    parser.add_argument(
        "--comment",
        action="store_true",
        help="Include comments in the highlighted PDF.",
    )
    parser.add_argument(
        "--data",
        type=json.loads,
        help="Data in JSON format (fields: user_input, pdf_filename, list_of_pages).",
    )

    args = parser.parse_args()

    # Initialize the Highlighter class with the provided arguments
    highlighter = Highlighter(
        silent=args.silent,
        openai_key=args.openai_key,
        comment=args.comment,
        llm_model=args.llm_model,
    )

    # Define the main asynchronous function to highlight the PDF
    async def main():
        highlighted_pdf = await highlighter.highlight(
            user_input=args.user_input,
            pdf_filename=args.pdf_filename,
            data=args.data,
        )
        # Save the highlighted PDF to a new file
        if not args.pdf_filename:
            # If no specific PDF filename is provided
            if args.data and len(args.data) == 1:
                # If data is provided and contains exactly one item, use its filename
                filename = args.data[0]["pdf_filename"].replace(".pdf", "_highlighted.pdf")
            else:
                # If no specific filename and data contains multiple items, generate a timestamped filename
                from datetime import datetime
                filename = f"highlighted_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        else:
            # If a specific PDF filename is provided, append '_highlighted' to its name
            filename = args.pdf_filename.replace(".pdf", "_highlighted.pdf")
        await save_pdf_to_file(
            highlighted_pdf, filename
        )
        # Print the clickable file path
        print(f'''Highlighted PDF saved to "file://{filename.replace(' ', '%20')}"''')

    # Run the main function using asyncio
    asyncio.run(main())
