import streamlit as st
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Function to calculate compatibility index using Sentence-BERT
def calculate_compatibility_sentence_bert(job_desc, skills):
    job_desc_embedding = model.encode(job_desc, convert_to_tensor=True)
    skills_embedding = model.encode(skills, convert_to_tensor=True)

    # Calculate cosine similarity between job description and skills embeddings
    compatibility_index = util.pytorch_cos_sim(job_desc_embedding, skills_embedding)[0][0].item()

    return compatibility_index

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
    "Graphic Designer": "Expert in visual communication and design principles, Proficient in Adobe Creative Suite, including Photoshop, Illustrator, and InDesign. Experience in creating branding and marketing materials.",
    "Mechanical Engineer": "Specializes in the design, analysis, and manufacturing of mechanical systems. Proficient in CAD software and engineering analysis tools. Experience in product development and materials selection.",
    "Content Writer": "Skilled in creating engaging and SEO-friendly content for various platforms. Proficient in research, copywriting, and editing. Experience in blog writing and social media management."
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
        # Calculate compatibility index using Sentence-BERT
        compatibility_index = calculate_compatibility_sentence_bert(job_description, employee_skills)
        st.write(f"Compatibility Index for {job_title}: {compatibility_index:.2f}")

        # Create and display gauge chart
        gauge_chart = create_gauge_chart(compatibility_index, f"{job_title} Compatibility Index")
        st.plotly_chart(gauge_chart)
