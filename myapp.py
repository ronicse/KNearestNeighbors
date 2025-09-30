import streamlit as st
import pandas as pd
import numpy as np

st.title("🔎 K-Nearest Neighbors - Build & Deploy (Streamlit)")
st.sidebar.header("Dataset & Preprocessing")

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
def load_sample(name):
  if name == "Sample dataset (Iris)":
    df = load_iris(as_frame=True)
  elif name == "Sample dataset (Wine)":
    df = load_wine(as_frame=True)
  elif name == "Sample dataset (Breast Cancer)":
    df = load_breast_cancer(as_frame=True)
  else:
    return None
  df = pd.concat([df.frame.reset_index(drop=True)], axis=1)
  return df

data_source = st.sidebar.selectbox("Data Source", ["Upload CSV", "Sample dataset (Iris)", "Sample dataset (Wine)", "Sample dataset (Breast Cancer)"])

if data_source == "Upload CSV":
  uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv", "txt"])
  if uploaded is not None:
    try:
      df = pd.read_csv(uploaded)
      df = df.dropna()
      st.success("Loaded sample data.")
    except Exception as e:
      st.sidebar.error(f"Couldn't read file: {e}")
      st.stop()
  else:
    st.info("Upload a CSV on the left or cloose a sample dataset to get started.")
    st.stop()
else:
  df = load_sample(data_source)  

st.write("### Dataset Preview")
st.write(df.head())


st.subheader("Data Preprocessing")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
  st.error("Need at least two numeric columns for KNearest Neighbors")
  st.stop()

target_col = st.selectbox("Select output target variable", numeric_cols)
features = [c for c in numeric_cols if c != target_col]
fatures = st.multiselect("Features (numeric)", options=features, default=features)

X = df[features].copy()
y = df[target_col].copy()

st.sidebar.header("Preprocessing & Model")
scale_method = st.sidebar.selectbox("Scaling",["None", "StandardScaler", "MinMaxScaler"])
use_pca = st.sidebar.checkbox("Project to 2 components with PCA for visualization", value=True)

test_size = st.sidebar.slider("Test set size (%)", min_value=5, max_value=50, value=20)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.subheader("KNN hyperparameters")
k = st.sidebar.slider("K (neighbors)", min_value=1, max_value=50, value=5)
st.sidebar.selectbox("Weight fucntion",["Uniform", "Distance"])
st.sidebar.selectbox("Distance metrics", ["minkowski", "euclidean", "manhattan"])









