# app2.py
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
import os
import base64

st.set_page_config(page_title="Data Science Final Project",
                   layout="wide"
                   )

# Display a fixed logo in the top-left corner
# Display fixed logos in the top-left corner (side-by-side)

LOGO_PATH_0 = "C:/Users/HP/Documents/GitHub/FinalProject_biu_ds20/src/pictures/streamlit_logo.jpeg"
LOGO_PATH_1 = "C:/Users/HP/Documents/GitHub/FinalProject_biu_ds20/src/pictures/github.png"
LOGO_PATH_2 = "C:/Users/HP/Documents/GitHub/FinalProject_biu_ds20/src/pictures/psf-logo.png"

def get_base64_image(local_path):
    with open(local_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data


try:
    logo_base64_0 = get_base64_image(LOGO_PATH_0)
    logo_base64_1 = get_base64_image(LOGO_PATH_1)
    logo_base64_2 = get_base64_image(LOGO_PATH_2)

    st.markdown(
        f"""
        <style>
        /* Container for both logos */
        .fixed-logos-container {{
            position: fixed;
            top: 70px;
            left: 10rem;
            display: flex;
            gap: 12px;
            z-index: 9999;
        }}

        /* Individual logo styling */
        .fixed-logo {{
            width: 80px;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        </style>
        <div class="fixed-logos-container">
            <img src="data:image/png;base64,{logo_base64_0}" class="fixed-logo">
            <img src="data:image/png;base64,{logo_base64_1}" class="fixed-logo">
            <img src="data:image/png;base64,{logo_base64_2}" class="fixed-logo">
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.warning(f"Logo not found: {e}")





# ---------------------------
# Helper: load local image as data URL or fallback to remote URL
# ---------------------------
def get_image_data_url(local_path: str, fallback_url: str) -> str:
    """
    If local_path exists, return a base64 data URL for embedding as CSS background.
    Otherwise return the fallback_url (remote).
    """
    try:
        if local_path and os.path.exists(local_path):
            with open(local_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
                # try to guess mime type by extension
                ext = os.path.splitext(local_path)[1].lower()
                mime = "image/jpeg"
                if ext in [".png"]:
                    mime = "image/png"
                elif ext in [".webp"]:
                    mime = "image/webp"
                return f"data:{mime};base64,{data}"
    except Exception as e:
        print(f"Error loading image {local_path}: {e}")
    return fallback_url

# ---------------------------
# Per-page backgrounds (local path first, then fallback remote image)
# ---------------------------
BASE_ASSETS_DIR = os.path.join(os.getcwd(), "assets")
BACKGROUNDS = {
    "Introduction": get_image_data_url("C:/Users/HP/Documents/GitHub/FinalProject_biu_ds20/src/pictures/12690.jpg",
                                      "https://images.unsplash.com/photo-1552664730-d307ca884978?w=1800&h=1000&fit=crop"),
    "Final Project Intro": get_image_data_url(os.path.join(BASE_ASSETS_DIR, "intro.jpg"),
                                              "https://images.unsplash.com/photo-1552664730-d307ca884978?w=1800&h=1000&fit=crop"),
    "Project Goal": get_image_data_url(os.path.join(BASE_ASSETS_DIR, "ml.jpg"),
                                      "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=1800&h=1000&fit=crop"),
    "Data": get_image_data_url(os.path.join(BASE_ASSETS_DIR, "data.jpg"),
                              "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1800&h=1000&fit=crop"),
    "ML DL Models": get_image_data_url(os.path.join(BASE_ASSETS_DIR, "ml.jpg"),
                                      "https://images.unsplash.com/photo-1555949519-3d98e70d9612?w=1800&h=1000&fit=crop"),
    "Production Model": get_image_data_url(os.path.join(BASE_ASSETS_DIR, "ml.jpg"),
                                          "https://images.unsplash.com/photo-1488229297570-58520851e868?w=1800&h=1000&fit=crop"),
    "Results": get_image_data_url(os.path.join(BASE_ASSETS_DIR, "results.jpg"),
                                 "https://images.unsplash.com/photo-1535223289827-42f1e9919769?w=1800&h=1000&fit=crop"),
    "Resources & References": get_image_data_url(os.path.join(BASE_ASSETS_DIR, "refs.jpg"),
                                                "https://images.unsplash.com/photo-1507842217343-583f20270319?w=1800&h=1000&fit=crop"),
}
# ---------------------------
# Animated binary overlay + minimal global CSS
# ---------------------------
BINARY_PARTICLES_CSS_JS = r"""
<style>
[data-testid="stAppViewContainer"] {
  position: relative;
  overflow: hidden;
}
.center-header {
    text-align: center !important;
}
/* Binary matrix overlay */
#binaryCanvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 1;
  opacity: 0.22;
  mix-blend-mode: screen;
}

/* Glassmorphic container */
.block-container {
  position: relative;
  z-index: 2;
  background: rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 2rem 10rem;
  border-radius: 16px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
}

/* Text styling */
.stApp {
  color: #111;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
}

.card {
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 6px 25px rgba(0,0,0,0.12);
  background: rgba(255, 255, 255, 0.4);
}

.big-font {
  font-size: 36px;
  font-weight: 700;
  color: #111;
  margin-bottom: 0.3rem;
}

.medium-font {
  font-size: 20px;
  color: #333;
  margin-bottom: 0.5rem;
}

/* ------------------------------
   Presenter section (adjusted)
   ------------------------------ */

/* NEW: Move full presenter block right */
.presenter-block {
  margin-left: 5rem;   /* adjust this to move even more/less right */
}

/* Clean individual text lines */
.presenter {
  font-size: 20px;
  margin-left: 0;
  margin-bottom: 0.2rem;
}

.presenter-name {
  font-size: 22px;
  margin-top: -0.5rem;
  letter-spacing: 0.5px;
}

.present-core, .present-res {
  font-size: 20px;
  margin-top: -0.3rem;
  letter-spacing: 0.3px;
}

.present-other {
  font-size: 20px;
  margin-top: 0rem;
  letter-spacing: 0.3px;
}

/* Optional: smaller spacing between sections */
section {
  margin-bottom: 1.5rem;
}
</style>

<canvas id="binaryCanvas"></canvas>

<script>
(() => {
  const canvas = document.getElementById('binaryCanvas');
  const ctx = canvas.getContext('2d');

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.scale(dpr, dpr);
  }
  resizeCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
    initDrops();
  });

  const CHARS = ['0','1'];
  let drops = [];

  function initDrops() {
    drops = [];
    const columns = Math.floor(window.innerWidth / 24);
    for (let i = 0; i < columns; i++) {
      const x = i * 24 + Math.random() * 12;
      const y = Math.random() * -window.innerHeight;
      const speed = 0.4 + Math.random() * 1.6;
      const size = 10 + Math.random() * 18;
      const alpha = 0.15 + Math.random() * 0.6;
      drops.push({x, y, speed, size, alpha, char: CHARS[Math.floor(Math.random()*2)]});
    }
  }
  initDrops();

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < drops.length; i++) {
      const d = drops[i];
      ctx.font = `${d.size}px monospace`;
      ctx.shadowColor = 'rgba(80, 240, 200, 0.8)';
      ctx.shadowBlur = 6;
      ctx.fillStyle = `rgba(100, 255, 210, ${d.alpha})`;
      ctx.fillText(d.char, d.x, d.y);

      d.y += d.speed * 1.6;
      d.x += Math.sin(d.y * 0.01 + i) * 0.3;
      if (Math.random() < 0.005) {
        d.char = CHARS[Math.floor(Math.random()*2)];
      }
      if (d.y > window.innerHeight + 20) {
        d.y = -20 - Math.random() * 200;
        d.x = Math.random() * window.innerWidth;
      }
    }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
})();
</script>
"""



st.markdown(BINARY_PARTICLES_CSS_JS, unsafe_allow_html=True)

# ---------------------------
# Page and navigation setup
# ---------------------------
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

# top-right navigation buttons
col1, col2, col3 = st.columns([10, 1, 1])
with col2:
    if st.session_state.current_page_idx > 0:
        if st.button("⬅", key="prev_btn"):
            st.session_state.current_page_idx -= 1
            st.rerun()
with col3:
    if st.session_state.current_page_idx < len(PAGES) - 1:
        if st.button("➡", key="next_btn"):
            st.session_state.current_page_idx += 1
            st.rerun()

selection = PAGES[st.session_state.current_page_idx]

# ---------------------------
# Utility: header
# ---------------------------
def header(title, subtitle=None):
    st.markdown(f"<div class='card'><div class='big-font'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='medium-font'>{subtitle}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Pages implementations
# ---------------------------
def intro():
    header("Data Science Final Project - BIU DS20")
    st.write("\n")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="presenter-block">   <!-- NEW WRAPPER -->

        ### Presenter
        <div class="presenter-name">Ofer Tzvi</div> 

        ### Objectives 
        <div class="present-core">Present core concepts</div> 
        <div class="present-res">Present results</div>

        </div>  <!-- END presenter-block -->
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        ### Duration  
        <div class="present-other">25–30 minutes</div>   

        ### Audience  
        <div class="present-other">BIU DS20</div>  
        """, unsafe_allow_html=True)

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
    header("Project Goal", "Definitions and components")
    st.write("""
    Data science sits at the intersection of statistics, software engineering, 
    and domain knowledge.
    """)

def data_page():
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
        iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
    st.text(classification_report(y_test, preds, target_names=iris.target_names))

def resources():
    header("Resources & References")
    st.markdown("""
    - Scikit-learn  
    - TensorFlow / Keras  
    - Hands-On ML books  
    - UCI, Kaggle datasets  
    """)

# Map page names to functions
PAGE_DISPATCH = {
    "Introduction": intro,
    "Final Project Intro": project_intro,
    "Project Goal": project_goal,
    "Data": data_page,
    "ML DL Models": ml_dl_models,
    "Production Model": production_models,
    "Results": results,
    "Resources & References": resources,
}

# ---------------------------
# Set the page background (static image per page)
# ---------------------------
def set_background(image_url: str):
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# Main
# ---------------------------
def main():
    bg = BACKGROUNDS.get(selection)
    if bg:
        set_background(bg)

    PAGE_DISPATCH[selection]()

if __name__ == "__main__":
    main()