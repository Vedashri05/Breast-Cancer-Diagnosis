import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#Load the dataset
BASE_DIR = Path(__file__).resolve().parent.parent
data = pd.read_csv(BASE_DIR / "data" / "breast_cancer.csv")

#Load the ml model
artifacts = joblib.load(BASE_DIR / "artifacts" / "breast-cancer-model.pkl")
model=artifacts["model"]
scaler=artifacts["scaler"]
features=artifacts["features"]

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict={}

    for feature in features:
        input_dict[feature]=st.sidebar.slider(
            label=feature,
            min_value=float(0),
            max_value=float(data[feature].max()),
            value=float(data[feature].mean())
        )
    return input_dict


def minmax_scale(feature_list,input_data):
    scaled = []
    for f in feature_list:
        min_val = data[f].min()
        max_val = data[f].max()
        if max_val == min_val:
            scaled.append(0.0)
        else:
            scaled.append((input_data[f] - min_val) / (max_val - min_val))
    return scaled

def get_radar_chart(input_data):
    # Feature groups
    mean_features = [
        "radius_mean","texture_mean","perimeter_mean","area_mean",
        "smoothness_mean","compactness_mean","concavity_mean",
        "concave points_mean","symmetry_mean","fractal_dimension_mean"
    ]

    se_features = [
        "radius_se","texture_se","perimeter_se","area_se",
        "smoothness_se","compactness_se","concavity_se",
        "concave points_se","symmetry_se","fractal_dimension_se"
    ]

    worst_features = [
        "radius_worst","texture_worst","perimeter_worst","area_worst",
        "smoothness_worst","compactness_worst","concavity_worst",
        "concave points_worst","symmetry_worst","fractal_dimension_worst"
    ]

    categories = [
        "Radius","Texture","Perimeter","Area",
        "Smoothness","Compactness","Concavity",
        "Concave Points","Symmetry","Fractal Dimension"
    ]

    mean_scaled = minmax_scale(mean_features,input_data)
    se_scaled = minmax_scale(se_features,input_data)
    worst_scaled = minmax_scale(worst_features,input_data)

    # Radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=mean_scaled,
        theta=categories,
        fill="toself",
        name="Mean"
    ))

    fig.add_trace(go.Scatterpolar(
        r=se_scaled,
        theta=categories,
        fill="toself",
        name="Standard Error"
    ))

    fig.add_trace(go.Scatterpolar(
        r=worst_scaled,
        theta=categories,
        fill="toself",
        name="Worst"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True
    )

    return fig


def main():
    st.set_page_config(
        layout="wide",
        page_title="Breast cancer diagnosis",
        page_icon=":female-doctor:",
        initial_sidebar_state="expanded"
    )

    with open(BASE_DIR/"assets"/"style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)

    input_data=add_sidebar()

    with st.container():
        st.header("Breast cancer Predictor")
        st.write("""
            This application predicts whether a breast tumor is **Benign** or **Malignant**
            using a machine learning model trained on clinical diagnostic features.The model analyzes input features provided by the user and returns a prediction
            along with a confidence score, helping support early risk assessment.
                """)
        
        col1,col2=st.columns([3,1])
        with col1:
            radar_chart=get_radar_chart(input_data)
            st.plotly_chart(radar_chart)

        with col2:
            st.subheader("Cell cluster prediction")
            st.write("The cell cluster is:")

            #-------- Prediction of input data ---------
            input_df=pd.DataFrame([input_data],columns=features)
            #Scale using trained standardScaler
            input_scaled=scaler.transform(input_df)
            prediction=model.predict(input_scaled)[0]
           
            if prediction==0:
                st.write("<span class='diagnosis benign'>Benign</span>",unsafe_allow_html=True)
            else:
                st.write("<span class='diagnosis maligant'>Maligant</span>",unsafe_allow_html=True)

            #prediction probablity for confidence
            st.write("Probablity of being benign: ",model.predict_proba(input_scaled)[0][0])
            st.write("Probablity of being maligant: ",model.predict_proba(input_scaled)[0][1])

            st.write("""⚠️ *This tool is intended for educational purposes only and should not be used
            as a substitute for professional medical diagnosis.*""")
        

if __name__=="__main__":
    main()