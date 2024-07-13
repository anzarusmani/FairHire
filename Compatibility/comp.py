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
def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 10,  # Scaling to 0-10 range
        title={'text': title},
        gauge={
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

# Add custom CSS for background image and text opacity
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover;
        background-position: center;
    }
    .stMarkdown p, .stTextInput label, .stButton button {
        background-color: rgba(0, 0, 0, 1.0);  /* Dark background with 100% opacity */
        padding: 0.5em;
        border-radius: 0.5em;
        color: white;  /* Ensure text is visible on dark background */
    }
    .stMarkdown h2 {
        background-color: rgba(0, 0, 0, 1.0);  /* Dark background with 100% opacity */
        padding: 0.5em;
        border-radius: 0.5em;
        color: white;  /* Ensure text is visible on dark background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title('Job Compatibility Checker')

# Predefined job descriptions
job_descriptions = {
    "Software Developer": "Proficient in JavaScript, Python, and ReactJS. Experience with web development and software design.",
    "Data Scientist": "Skilled in Python, machine learning, data analysis, and statistical modeling. Familiar with data visualization tools.",
    "Project Manager": "Experienced in project management, Agile methodologies, and team coordination. Proficient in project planning and execution."
}

# Display predefined job descriptions
st.subheader("Predefined Job Descriptions:")
for job_title, job_description in job_descriptions.items():
    st.markdown(f"**{job_title}:** {job_description}")

# Input for employee skills
employee_skills = st.text_input('Enter Employee Skills:', 'reactJS, development')

# Button to calculate compatibility
if st.button('Calculate Compatibility'):
    for job_title, job_description in job_descriptions.items():
        # Calculate compatibility index
        compatibility_index = calculate_compatibility(job_description, employee_skills)
        st.write(f"Compatibility Index for {job_title}: {compatibility_index:.2f}")

        # Create and display gauge chart
        gauge_chart = create_gauge_chart(compatibility_index, f"{job_title} Compatibility Index")
        st.plotly_chart(gauge_chart)
