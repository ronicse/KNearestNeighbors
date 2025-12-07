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
features = st.multiselect("Features (numeric)", options=features, default=features)

X = df[features].copy()
y = df[target_col].copy()

st.sidebar.header("Preprocessing & Model")
scale_method = st.sidebar.selectbox("Scaling",["None", "StandardScaler", "MinMaxScaler"])
use_pca = st.sidebar.checkbox("Project to 2 components with PCA for visualization", value=True)

test_size = st.sidebar.slider("Test set size (%)", min_value=5, max_value=50, value=20)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.subheader("KNN hyperparameters")
k = st.sidebar.slider("K (neighbors)", min_value=1, max_value=50, value=5)
weights = st.sidebar.selectbox("Weight fucntion",["uniform", "distance"])
metric = st.sidebar.selectbox("Distance metrics", ["minkowski", "euclidean", "manhattan"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))


from sklearn.preprocessing import StandardScaler, MinMaxScaler
if scale_method == "StandardScaler":
  scaler = StandardScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
  X_test = pd.DataFrame(scaler.transform(X_test), columns = features)
  st.write("### Standard Scaling")
  st.write(X_train.head())
elif scale_method == "MinMaxScaler":
  scaler = MinMaxScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
  X_test = pd.DataFrame(scaler.transform(X_test), columns = features)
  st.write("### Min-Max Scling")
  st.write(X_train.head())
else:
  scaler = None

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=int(k), weights=weights, metric=metric)
clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

import matplotlib.pyplot as plt
st.write("## Model Evaluation")
col1, col2 = st.columns([1, 1])
with col1:
  st.metric("Accuracy", f"{acc:3f}")
  st.write("### Classification report")
  st.dataframe(pd.DataFrame(report).transpose())
with col2:
  st.write("### Confusion Matrix")
  fig, ax = plt.subplots()
  im = ax.matshow(cm)
  for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, int(val), ha='center', va='center')
  ax.set_xlabel('Predicted')
  ax.set_ylabel("Actual")
  st.pyplot(fig)

  


























