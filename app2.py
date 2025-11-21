import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import base64

st.set_page_config(page_title="Data Science Final Project",
                   layout="wide",
                   initial_sidebar_state="expanded")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# -------------------------------------------------------------------
# ✔ Correct BACKGROUNDS dict (no duplicates, no invalid quotes)
# -------------------------------------------------------------------
BACKGROUNDS = {
    "Introduction": "time_series/BackgroundImage1.jpeg",
    "Final Project Intro": "https://images.unsplash.com/photo-1552664730-d307ca884978?w=1800&h=1000&fit=crop",
    "Project Goal": "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=1800&h=1000&fit=crop",
    "Data": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1800&h=1000&fit=crop",
    "ML DL Models": "https://images.unsplash.com/photo-1555949519-3d98e70d9612?w=1800&h=1000&fit=crop",
    "Production Model": "https://images.unsplash.com/photo-1488229297570-58520851e868?w=1800&h=1000&fit=crop",
    "Results": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1800&h=1000&fit=crop",
    "Resources & References": "https://images.unsplash.com/photo-1507842217343-583f20270319?w=1800&h=1000&fit=crop",
}

# -------------------------------------------------------------------
# Background CSS function
# -------------------------------------------------------------------
def set_background(image_url: str):
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url('{image_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        .block-container {{
            background: rgba(255, 255, 255, 0.88);
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------------
# Custom styling
# -------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    .big-font {
        font-size: 36px;
        font-weight: 700;
    }
    .medium-font {
        font-size: 20px;
        color: #333333;
    }
    .card {
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        background: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Navigation
# -------------------------------------------------------------------
PAGES = [
    "Introduction",
    "Final Project Intro",
    "Project Goal",
    "Data",
    "ML DL Models",
    "Production Model",
    "Results",
    "Resources & References",
]

if 'current_page_idx' not in st.session_state:
    st.session_state.current_page_idx = 0

sidebar_selection = st.sidebar.selectbox("Jump to page", PAGES, index=st.session_state.current_page_idx)

st.session_state.current_page_idx = PAGES.index(sidebar_selection)

selection = PAGES[st.session_state.current_page_idx]

# -------------------------------------------------------------------
# Helper: Card-style header
# -------------------------------------------------------------------
def header(title, subtitle=None):
    st.markdown(f"<div class='card'><div class='big-font'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='medium-font'>{subtitle}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------
def intro():
    header("Data Science Final Project, January 2025 BIU_DS20",
           "Final project Data Science - Machine Learning & Deep Learning Algorithms.")
    st.write("\n")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Presenter
        - Ofer Tzvi  
        - Technology, Data Science, Risk Management  

        ### Objectives  
        - Present core concepts  
        - Discuss project phases  
        - Show model results
        """)
    with col2:
        st.markdown("""
        ### Duration  
        - 25–30 minutes  

        ### Audience  
        - Data practitioners  
        - Engineers  
        - Academic staff  
        """)
    st.markdown("---")


def project_intro():
    header("Agenda")
    st.markdown("""
    1. What is Data Science?  
    2. Data Preparation  
    3. EDA  
    4. ML Basics & Example  
    5. Unsupervised Learning  
    6. Model Evaluation  
    7. Deep Learning  
    8. Computer Vision  
    9. Deployment  
    10. Resources  
    """)


def project_goal():
    header("What is Data Science?", "Definitions and components")
    st.write("""
    Data science sits at the intersection of statistics, software engineering, 
    and domain knowledge.
    """)


def data():
    header("Data Preparation", "Why it matters")
    iris = datasets.load_iris(as_frame=True).frame
    demo = iris.copy()
    demo.loc[demo.sample(frac=0.06).index, 'sepal length (cm)'] = np.nan
    st.dataframe(demo.head(10))
    st.write(demo.isna().sum())


def ml_dl_models():
    header("Exploratory Data Analysis (EDA)")
    df = datasets.load_wine(as_frame=True).frame
    st.dataframe(df.head(8))

    sns.set(style="whitegrid")
    pair_df = df.iloc[:, :6].sample(150, random_state=1)
    fig = sns.pairplot(pair_df)
    st.pyplot(fig.fig)


def production_models():
    header("Machine Learning Basics")
    st.markdown("Supervised learning vs Unsupervised learning...")


def results():
    header("Model Example: RandomForest on Iris")
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, stratify=iris.target
    )
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")


def resources():
    header("Resources & References")
    st.markdown("Useful links, books, datasets...")

# -------------------------------------------------------------------
# Page router
# -------------------------------------------------------------------
PAGE_DISPATCH = {
    "Introduction": intro,
    "Final Project Intro": project_intro,
    "Project Goal": project_goal,
    "Data": data,
    "ML DL Models": ml_dl_models,
    "Production Model": production_models,
    "Results": results,
    "Resources & References": resources,
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    bg = BACKGROUNDS.get(selection)
    if bg:
        set_background(bg)

    PAGE_DISPATCH[selection]()
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made by Streamlit")

if __name__ == "__main__":
    main()
