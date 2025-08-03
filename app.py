import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

# Load the dataset
heart_df = pd.read_csv("Heart_Disease_Prediction_Dataset.csv")

# Feature and target split
X = heart_df.drop("target", axis=1)
Y = heart_df["target"]

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, stratify=Y, random_state=2
)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy scores
training_data_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(model.predict(X_test), Y_test)

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction Model", layout="centered")
st.title("💓 Heart Disease Predictor App")

# Image in center
img = Image.open("heart_image.jpg")
resized_img = img.resize((250, 250))  # Width x Height in pixels

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(resized_img)

# User Input
st.subheader("🔢 Enter Patient Data below")
input_text = st.text_input("Provide comma-separated values (13 total features)")
    
# Predict button
if st.button("Predict"):
    if not input_text.strip():
        st.warning("⚠️ Please enter values before clicking Predict.")
    else:
        try:
            sprted_input = [float(i.strip()) for i in input_text.split(",") if i.strip()]
            expected_features = 13

            if len(sprted_input) != expected_features:
                st.warning(f"⚠️ Please enter exactly {expected_features} comma-separated values.")
            else:
                np_df = np.asarray(sprted_input).reshape(1, -1)
                prediction = model.predict(np_df)

                if prediction[0] == 0:
                    st.success("✅ This person does **not** have heart disease.")
                else:
                    st.error("⚠️ This person **has** heart disease.")

        except ValueError:
            st.error("❌ Invalid input! Please enter only **numbers** separated by commas.")

# Feature explanation
with st.expander("🧾 Click to see the meaning of each input feature"):
    st.markdown(
        """
| No. | Feature Name | Meaning |
|-----|--------------|---------|
| 1️⃣ | `age` | Age in years |
| 2️⃣ | `sex` | Sex (1 = male, 0 = female) |
| 3️⃣ | `cp` | Chest pain type (0–3) |
| 4️⃣ | `trestbps` | Resting blood pressure (mm Hg) |
| 5️⃣ | `chol` | Serum cholesterol (mg/dl) |
| 6️⃣ | `fbs` | Fasting blood sugar >120 mg/dl (1 = true, 0 = false) |
| 7️⃣ | `restecg` | Resting ECG results (0 = normal, 1 = ST-T abnormality, 2 = LVH) |
| 8️⃣ | `thalach` | Max heart rate achieved |
| 9️⃣ | `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| 🔟 | `oldpeak` | ST depression induced by exercise |
| 1️⃣1️⃣ | `slope` | Slope of ST segment (0 = up, 1 = flat, 2 = down) |
| 1️⃣2️⃣ | `ca` | Number of major vessels colored (0–3) |
| 1️⃣3️⃣ | `thal` | Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect) |
"""
    )

# Show dataset and model performance
st.subheader("📋About the Data")
if st.checkbox("Show full dataset"):
    st.dataframe(heart_df)
else:
    st.dataframe(heart_df.head())

st.subheader("📊 Model Performance")
st.write(f"* **Training Accuracy**: {training_data_accuracy*100:.2f}%")
st.write(f"* **Test Accuracy**: {test_data_accuracy*100:.2f}%")
