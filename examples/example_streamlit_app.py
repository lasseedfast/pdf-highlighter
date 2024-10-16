import streamlit as st
from highlight_pdf import Highlighter
import asyncio
import io
import base64

async def highlight_pdf(user_input, pdf_file, make_comments):
    highlighter = Highlighter(comment=make_comments)
    pdf_buffer = io.BytesIO(pdf_file.read())
    highlighted_pdf_buffer = await highlighter.highlight(user_input, pdf_filename=pdf_buffer)
    return highlighted_pdf_buffer

def main():

    with st.sidebar:
        st.write('This is a demo of a PDF highlighter tool that highlights relevant sentences in a PDF document based on user input.')
    st.title("PDF Highlighter Demo")

    user_input = st.text_input("Enter your question or input text:")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    make_comments = st.checkbox("Make comments to the highlighted text (takes a bit longer)")

    if st.button("Highlight PDF"):
        if user_input and pdf_file:
            with st.spinner("Processing..."):
                highlighted_pdf_buffer = asyncio.run(highlight_pdf(user_input, pdf_file, make_comments))
                if highlighted_pdf_buffer:
                    # Encode the PDF buffer to base64
                    base64_pdf = base64.b64encode(highlighted_pdf_buffer.getvalue()).decode('utf-8')

                    # Embed PDF in HTML
                    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="300" height="700" type="application/pdf"></iframe>'

                    with st.sidebar:
                        # Display file
                        st.markdown("_Preview of highlighted PDF:_")
                        st.markdown(pdf_display, unsafe_allow_html=True)
                
                    st.download_button(
                        label="Download Highlighted PDF",
                        data=highlighted_pdf_buffer,
                        file_name="highlighted_document.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("No relevant sentences found to highlight.")
        else:
            st.error("Please provide both user input and a PDF file.")

if __name__ == "__main__":
    main()