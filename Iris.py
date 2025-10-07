import base64
import streamlit as st
import pandas as pd
from joblib import load


# Load the saved model, scaler, and target names
svm_linear, scaler, target_names = load("svm_model.pkl")

# Streamlit App

def set_bg(image_file):
    import base64
    import streamlit as st

    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    /* Set background for slider container */
    section[data-testid="stSlider"] {{
    background-color: rgba(255, 255, 255, 0.8) !important;
    padding: 10px;
    border-radius: 10px;
    }}
    label, section[data-testid="stSlider"] label {{
    color: #000000 !important;
    font-weight: bold;
    }}
    div[data-testid="stSlider"] div[role="slider"] + div {{
    color: #000000 !important;
    font-weight: bold;
    }}
    header, footer {{visibility: hidden;}}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {{
        color: #1a1a1a !important;
    }}
    /* Slider track */
    div[data-baseweb="slider"] > div > div > div > div > div[role="progressbar"] {{
        background-color: #000000 !important;
    }}
    /* Slider handle */
    div[data-baseweb="slider"] input[type="range"]::-webkit-slider-thumb {{
        background-color: #000000 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

set_bg("iriss.jpg")

# ---------------- App title ----------------
st.title("🌸 Iris Flower Classification App")
st.write("Enter sepal and petal measurements to predict the iris species.")

st.header("Input Features")
def user_input_features():
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.5)
    sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
    petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    return pd.DataFrame(data, index=[0])


# Get user input
input_df = user_input_features()

input_df['petal_area'] = input_df['petal_length'] * input_df['petal_width']
input_df['sepal_area'] = input_df['sepal_length'] * input_df['sepal_width']

feature_order = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'petal_area', 'sepal_area']
input_df = input_df[feature_order]

scaled_input = scaler.transform(input_df)

# Make prediction
prediction = svm_linear.predict(scaled_input)
prediction_proba = svm_linear.predict_proba(scaled_input)

# Display results
st.subheader("Prediction")
st.markdown(f"<p style='color: #689a75; font-size:24px;'>Predicted Species:{target_names[prediction[0]]}</p>", unsafe_allow_html=True)

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)

import plotly.graph_objects as go

fig = go.Figure(
    go.Bar(
        x=target_names,
        y=prediction_proba[0],
        marker_color="#689a75"
    ))
fig.update_layout(
    plot_bgcolor="#EDE2C2",      
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#000000"),  # Set all text to black
    xaxis=dict(
        title="Species",
        tickfont=dict(color="#000000"),
        titlefont=dict(color="#000000")
    ),
    yaxis=dict(
        title="Probability",
        tickfont=dict(color="#000000"),
        titlefont=dict(color="#000000")
    )
)

st.plotly_chart(fig)