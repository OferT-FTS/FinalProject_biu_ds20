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

st.set_page_config(page_title="Data Science & ML Presentation",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- Custom styling for professional look ---
st.markdown(
    "<style>
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }
    .big-font { font-size:36px; font-weight:700; }
    .medium-font { font-size:20px; color: #333333; }
    .card { padding: 18px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); background: white; }
    </style>", unsafe_allow_html=True)

# --- Navigation ---
PAGES = [
    "Home",
    "Agenda",
    "What is Data Science?",
    "Data Preparation",
    "Exploratory Data Analysis",
    "Machine Learning Basics",
    "Supervised Learning (Example)",
    "Unsupervised Learning",
    "Model Evaluation & Validation",
    "Deep Learning Overview",
    "Neural Network Example (Keras)",
    "Computer Vision & CNNs (Overview)",
    "Deployment & MLOps",
    "Resources & References",
]

selection = st.sidebar.selectbox("Jump to page", PAGES)

# --- Helper utilities ---

def header(title, subtitle=None):
    st.markdown(f"<div class='card'><div class='big-font'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='medium-font'>{subtitle}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)


# --- Pages ---

def page_home():
    header("Data Science & Machine Learning — Professional Overview",
           "A concise, example-driven presentation covering data science, ML and deep learning.")
    st.write("\n")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Presenter\n- Your name here\n- Your role / affiliation\n\n### Objectives\n- Explain core concepts of data science, ML, and DL\n- Walk through data preparation and EDA best practices\n- Demonstrate ML and DL examples with code and visuals\n- Discuss deployment and MLOps considerations")
    with col2:
        st.markdown("### Duration\n- ~45–60 minutes (adjustable)\n\n### Audience\n- Data practitioners, engineers & decision makers")
    st.markdown("---")


def page_agenda():
    header("Agenda")
    st.markdown("\n1. What is Data Science?\n2. Data Preparation\n3. EDA\n4. ML Basics & Supervised Learning Example\n5. Unsupervised Learning\n6. Model Evaluation\n7. Deep Learning Overview & Keras Example\n8. Computer Vision overview\n9. Deployment & MLOps\n10. Resources")


def page_what_is_ds():
    header("What is Data Science?", "Definitions and components")
    st.write("Data science sits at the intersection of statistics, software engineering, and domain knowledge.")
    st.markdown("**Typical workflow:**\n- Problem definition\n- Data acquisition\n- Data cleaning & feature engineering\n- Modeling & validation\n- Deployment & monitoring")


def page_data_prep():
    header("Data Preparation", "Why it matters and best practices")
    st.markdown("**Key steps:**\n- Data understanding and profiling\n- Handling missing values and outliers\n- Encoding categorical variables\n- Scaling / normalization\n- Train/validation/test splitting & leakage prevention")
    st.write("### Quick demo: load a small dataset and show missing values")
    iris = datasets.load_iris(as_frame=True).frame
    # artificially add some missing values for demo
    demo = iris.copy()
    demo.loc[demo.sample(frac=0.06).index, 'sepal length (cm)'] = np.nan
    st.dataframe(demo.head(10))
    st.markdown("Missing values summary:")
    st.write(demo.isna().sum())


def page_eda():
    header("Exploratory Data Analysis (EDA)")
    st.write("Use descriptive stats, visualizations and correlations to learn the data before modeling.")
    df = datasets.load_wine(as_frame=True).frame
    st.markdown("**Dataset example: Wine** — first 8 rows")
    st.dataframe(df.head(8))

    st.write("### Pairplot sample (small) — beware of heavy plots on large data")
    # small pairplot using seaborn
    sns.set(style="whitegrid")
    pair_df = df.iloc[:, :6].sample(150, random_state=1)
    fig = sns.pairplot(pair_df)
    st.pyplot(fig.fig)


def page_ml_basics():
    header("Machine Learning Basics", "Supervised vs unsupervised, features, labels")
    st.markdown("**Supervised learning** — labeled data: regression / classification\n**Unsupervised learning** — clustering, dimensionality reduction")
    st.markdown("**Feature engineering** examples: scaling, interactions, datetime features, embeddings")


def page_supervised_example():
    header("Supervised Learning — Example: RandomForest on Iris")
    st.write("This small demo trains a RandomForestClassifier on the Iris dataset and shows metrics.")

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    n_estimators = st.slider("n_estimators", 10, 200, 100)
    max_depth = st.slider("max_depth", 1, 20, 5)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    with st.spinner("Training RandomForest..."):
        clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.metric("Accuracy", f"{acc:.3f}")
    st.text("Classification report:")
    st.text(classification_report(y_test, preds, target_names=iris.target_names))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


def page_unsupervised():
    header("Unsupervised Learning", "Clustering and dimensionality reduction")
    st.markdown("Short descriptions: KMeans, Hierarchical clustering, PCA, t-SNE, UMAP")
    st.write("Use-case examples: customer segmentation, anomaly detection, visualization of high-dimensional data.")


def page_evaluation():
    header("Model Evaluation & Validation")
    st.markdown("**Key ideas:**\n- Train/dev/test splits\n- Cross-validation\n- Metrics: precision, recall, F1, ROC-AUC, PR-AUC\n- Calibration & reliability diagrams\n- Confusion matrix and class imbalance handling")
    st.write("### Quick example: demonstrate cross-validation concept visually")
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/K-fold_cross_validation_EN.svg", caption="K-Fold cross-validation")
    st.markdown("(Image hosted remotely — replace with internal asset for offline presentations.)")


def page_dl_overview():
    header("Deep Learning Overview", "When to use, strengths and challenges")
    st.markdown("**Deep learning**: large models, representation learning — excels with images, audio, text.\nChallenges: data hunger, interpretability, compute cost.")


def page_keras_example():
    header("Neural Network Example (Keras)", "Train a small MLP on the sklearn digits dataset")
    st.write("We will train a compact neural network on the `digits` dataset (handwritten 8x8 images). This runs quickly for demo purposes.")

    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    # flatten
    X = X.reshape((X.shape[0], -1)).astype('float32') / 16.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # build model
    num_classes = len(np.unique(y))
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    epochs = st.slider('epochs', 5, 50, 12)
    batch_size = st.selectbox('batch_size', [16, 32, 64], index=1)

    # show model summary
    st.text('Model summary:')
    model.summary(print_fn=lambda x: st.text(x))

    if st.button('Train Neural Network'):
        with st.spinner('Training the Keras model — this may take a moment'):
            history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        st.metric('Test accuracy', f"{test_acc:.3f}")

        # plot training curves
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='train_loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        st.success('Training completed — examine tradeoffs and consider regularization, more data, or architecture changes')


def page_cnn_overview():
    header("Computer Vision & Convolutional Neural Networks", "Core ideas & popular architectures")
    st.markdown("**Convolutional layers** detect local patterns; pooling reduces spatial size; modern CV uses CNN backbones like ResNet, EfficientNet.")
    st.markdown("**Applications:** image classification, segmentation, object detection, generative models.")


def page_deployment():
    header("Deployment & MLOps", "Putting models into production")
    st.markdown("**Topics to consider:**\n- Model packaging (ONNX, SavedModel)\n- Serving (FastAPI, TensorFlow Serving, TorchServe)\n- Monitoring & drift detection\n- CI/CD for models and data\- Reproducibility and pipelines (Airflow, Prefect)")
    st.write("Consider security, scaling, latency and cost when selecting an architecture for serving models.")


def page_resources():
    header("Resources & References")
    st.markdown("**Suggested reading & links**:\n- " + "Scikit-Learn, TensorFlow/Keras, Hands-On ML books\n- Papers and blogs for deep dives\n- Public datasets: UCI, Kaggle, Open Images")
    st.markdown("**Suggested next steps for learners:**\n- Build small projects end-to-end\n- Participate in Kaggle competitions\n- Learn deployment and reproducibility best practices")


# --- Router ---
PAGE_DISPATCH = {
    "Home": page_home,
    "Agenda": page_agenda,
    "What is Data Science?": page_what_is_ds,
    "Data Preparation": page_data_prep,
    "Exploratory Data Analysis": page_eda,
    "Machine Learning Basics": page_ml_basics,
    "Supervised Learning (Example)": page_supervised_example,
    "Unsupervised Learning": page_unsupervised,
    "Model Evaluation & Validation": page_evaluation,
    "Deep Learning Overview": page_dl_overview,
    "Neural Network Example (Keras)": page_keras_example,
    "Computer Vision & CNNs (Overview)": page_cnn_overview,
    "Deployment & MLOps": page_deployment,
    "Resources & References": page_resources,
}


def main():
    fn = PAGE_DISPATCH.get(selection, page_home)
    fn()
    st.sidebar.markdown('---')
    st.sidebar.markdown('Made with ❤️  •  Professional template • Customize for your talk')


if __name__ == '__main__':
    main()
