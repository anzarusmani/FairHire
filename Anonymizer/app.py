#version1
import streamlit as st
import fitz  # PyMuPDF
import re
from transformers import pipeline


# Define the CSS for the background image
background_image = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1529400971008-f566de0e6dfc?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(background_image, unsafe_allow_html=True)

# Define the company name and style
company_name_html = """
<div style='
    position: absolute;
    top: -1000;
    left: -100;
    padding: 10px;
    font-size: 24px;
    font-weight: bold;
    color: #333;
'>
    Fair Hire Kaleido
</div>
"""

# Inject the HTML and CSS into the Streamlit app
st.markdown(company_name_html, unsafe_allow_html=True)


# Initialize the NER pipeline for Hugging Face Transformers
ner_hf = pipeline("ner", grouped_entities=True)

# Mapping of Hugging Face entity labels to anonymized terms
label_to_anonymized_hf = {
    "PER": "[NAME]",       # Person
    "LOC": "[ADDRESS]",    # Location
}

# Regex patterns for email, phone number, and age
email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
phone_pattern = re.compile(r'\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}|\d{3}-\d{3}-\d{4}')
age_pattern = re.compile(r'\b\d{1,3}\s*(years|year|yrs|yr)\b')

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    text_list = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        text_list.append(text)
    return text_list

def anonymize_text_with_hf(text_list):
    """Replaces specific NER-identified terms with anonymized terms using Hugging Face Transformers."""
    anonymized_list = []
    for text in text_list:
        ner_results = ner_hf(text)
        anonymized_text = text
        for entity in ner_results:
            label = entity["entity_group"]
            word = entity["word"]
            if label in label_to_anonymized_hf:
                anonymized_text = anonymized_text.replace(word, label_to_anonymized_hf[label])
        anonymized_list.append(anonymized_text)
    return anonymized_list

def anonymize_text_with_regex(text_list):
    """Replaces email, phone numbers, and age with anonymized terms using regex."""
    anonymized_list = []
    for text in text_list:
        # Anonymize emails
        text = email_pattern.sub("[EMAIL]", text)
        # Anonymize phone numbers
        text = phone_pattern.sub("[PHONE]", text)
        # Anonymize ages
        text = age_pattern.sub("[AGE]", text)
        anonymized_list.append(text)
    return anonymized_list

def create_new_pdf_with_anonymized_text(output_pdf_path, anonymized_text_list):
    """Creates a new PDF with the anonymized text on clean pages."""
    doc = fitz.open()  # Create a new PDF document
    for anonymized_text in anonymized_text_list:
        page = doc.new_page()  # Add a new blank page
        lines = anonymized_text.split('\n')
        y = 72  # Start y position for text insertion
        for line in lines:
            text_rect = fitz.Rect(72, y, 540, y + 20)
            page.insert_textbox(text_rect, line, fontsize=11, fontname="helv")
            y += 15  # Move y position for next line
    doc.save(output_pdf_path)

def process_resume(input_pdf, output_pdf):
    """Processes the resume PDF to anonymize specific fields and create a new PDF."""
    # Extract text from the input PDF
    original_text_list = extract_text_from_pdf(input_pdf)
    
    # Anonymize text using Hugging Face Transformers
    anonymized_text_hf = anonymize_text_with_hf(original_text_list)
    
    # Anonymize emails, phone numbers, and age using regex
    anonymized_text_regex = anonymize_text_with_regex(anonymized_text_hf)
    
    # Create a new PDF with the anonymized text
    create_new_pdf_with_anonymized_text(output_pdf, anonymized_text_regex)


col1, col2 = st.columns(2)

with col1:  

    # Streamlit app
    st.title("Resume Anonymizer")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file_uploader")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("input.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

    if st.button("Anonymize PDF", key="anonymize_button"):
        output_pdf_path = "output.pdf"
        process_resume("input.pdf", output_pdf_path)

        # Provide a download button for the anonymized PDF
        with open(output_pdf_path, "rb") as f:
            st.download_button(
                label="Download Anonymized PDF",
                data=f,
                file_name="anonymized_output.pdf",
                mime="application/pdf",
                key="download_button"
            )

#############################################################################################################################################

