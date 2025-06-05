import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import PyPDF2
from docx import Document

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        document = Document(file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
    return text

def word_cloud_app():
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>😶‍🌫️ Word Cloud Generator</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("Upload a PDF, DOCX, or Text file to generate a word cloud.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type, "filesize":uploaded_file.size}
        st.write(file_details)

        text_content = ""
        if uploaded_file.type == "application/pdf":
            text_content = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_content = extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text_content = uploaded_file.getvalue().decode("utf-8")

        if text_content:
            st.subheader("Generated Word Cloud")
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_content)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

                # Option to download the word cloud image
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label="Download Word Cloud as PNG",
                    data=buf.getvalue(),
                    file_name="wordcloud.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Error generating word cloud: {e}. Please ensure the file contains sufficient text.")
        else:
            st.warning("Could not extract text from the uploaded file or the file is empty.")

# To run this as a standalone page when called by the main app
def app():
    word_cloud_app()

if __name__ == "__main__":
    st.set_page_config("Word Cloud Generator", layout='wide')
    word_cloud_app()