import streamlit as st
import spacy
import plotly.graph_objects as go

# Load the larger spaCy model with word vectors
nlp = spacy.load("en_core_web_lg")

# Function to calculate compatibility index based on contextual similarity
def calculate_compatibility(job_description, employee_skills):
    jd_doc = nlp(job_description.lower())  # Lowercase for case insensitivity
    es_doc = nlp(employee_skills.lower())
    similarity_score = jd_doc.similarity(es_doc)
    return similarity_score

# Function to create a gauge chart
def create_gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 10,  # Scaling to 0-10 range
        title = {'text': "Compatibility Index"},
        gauge = {
            'axis': {'range': [0, 10]},
            'steps': [
                {'range': [0, 3.3], 'color': "red"},
                {'range': [3.3, 6.6], 'color': "yellow"},
                {'range': [6.6, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value * 10
            }
        }
    ))
    return fig

# Streamlit App
st.title('Job Compatibility Checker')

# Input for job description
job_description = st.text_input('Enter Job Description:', 'javascript,python')

# Input for employee skills
employee_skills = st.text_input('Enter Employee Skills:', 'reactJS,development')

# Button to calculate compatibility
if st.button('Calculate Compatibility'):
    # Calculate compatibility index
    compatibility_index = calculate_compatibility(job_description, employee_skills)
    st.write(f"Compatibility Index: {compatibility_index:.2f}")

    # Display gauge chart
    fig = create_gauge_chart(compatibility_index)
    st.plotly_chart(fig)
