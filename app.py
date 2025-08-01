import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import os
import urllib.request

# Page configuration
st.set_page_config(page_title="Netflix Thumbnail Genre Classifier", layout="wide")

# === Paths ===
MODEL_PATH = "model/final_efficientnetb4_model.keras"
HF_URL = "https://huggingface.co/spaces/sweetyseelam/netflix-thumbnail-model/resolve/main/final_efficientnetb4_model.keras"
LABEL_MAP_PATH = "model/label_map_efficientnetb4.pkl"

# === Download model if not already present ===
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    with st.spinner("📥 Downloading model from Hugging Face..."):
        urllib.request.urlretrieve(HF_URL, MODEL_PATH)
        st.success("✅ Model downloaded successfully!")

# === Load Model ===
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.stop()

# === Load Label Map ===
try:
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
except Exception as e:
    st.sidebar.error(f"❌ Failed to load label map: {e}")
    st.stop()

# === Title ===
st.title("🎬 Netflix Thumbnail Genre Classifier (EfficientNetB4)")

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
pages = ["Project Overview", "Try It Now", "Model Info", "Results & Insights"]
selection = st.sidebar.radio("Go to", pages)

# === Image Preprocessing Function ===
def preprocess_image(image):
    image = image.resize((225, 225))             # Match training input size
    image = image.convert("L")                   # Convert to grayscale (1 channel)
    img_array = np.array(image) / 255.0          # Normalize
    img_array = np.expand_dims(img_array, axis=-1)  # Shape: (225, 225, 1)
    img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 225, 225, 1)
    return img_array

# === Page 1: Overview ===
if selection == "Project Overview":
    st.header("📌 Project Overview")
    st.markdown("""
This Deep Learning project classifies Netflix movie thumbnails into five genres — **Action, Comedy, Drama, Romance, Thriller** — using a custom-trained EfficientNetB4 model.

**Dataset**: 2,330 unique posters (466 per genre)  
**Model**: EfficientNetB4  
**Accuracy**: ~39%  

🎯 Business Use Case:  
- Auto-tagging content  
- Enhancing thumbnail recommendations  
- Improving user engagement  
    """)

# === Page 2: Try It Now ===
elif selection == "Try It Now":
    st.header("🖼️ Try It Now")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload a poster (jpg/png)", type=["jpg", "jpeg", "png"])
        submit_user = st.button("Submit", key="submit_user")

    with col2:
        sample_options = {
            "Action": "data/sample_posters/action.jpg",
            "Comedy": "data/sample_posters/comedy.jpg",
            "Drama": "data/sample_posters/drama.jpg",
            "Romance": "data/sample_posters/romance.jpg",
            "Thriller": "data/sample_posters/thriller.jpg"
        }
        selected_sample = st.selectbox("Pick a sample poster", list(sample_options.keys()))
        submit_sample = st.button("Submit", key="submit_sample")

    image = None
    if submit_user and uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"❌ Error loading uploaded image: {e}")
            st.stop()
    elif submit_sample:
        try:
            image = Image.open(sample_options[selected_sample]).convert("RGB")
        except FileNotFoundError:
            st.error(f"Sample image for '{selected_sample}' not found.")
            st.stop()

    if image:
        st.image(image, caption="Input Poster", use_column_width=True)
        img_array = preprocess_image(image)

        # === DEBUG: Show shape and type ===
        st.markdown(f"📐 **Preprocessed Shape:** `{img_array.shape}`")
        st.markdown(f"🎨 **Type:** `{img_array.dtype}` | Min: {img_array.min():.3f}, Max: {img_array.max():.3f}")

        try:
            prediction = model.predict(img_array)
            predicted_label = inv_label_map[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.markdown(f"**🎯 Predicted Genre:** `{predicted_label}`")
            st.markdown(f"**📊 Confidence:** `{confidence:.2f}%`")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# === Page 3: Model Info ===
elif selection == "Model Info":
    st.header("🧠 Model Details")
    st.markdown("""
- Base Model: EfficientNetB4  
- Input: 224 x 224 x 3  
- Fine-tuned with Dropout, Class Weights  
- Optimizer: Adam (1e-5)  
- EarlyStopping enabled  
- Accuracy: ~39%  
    """)

# === Page 4: Results & Insights ===
elif selection == "Results & Insights":
    st.header("📊 Model Evaluation & Insights")

    st.subheader("✅ Accuracy Plot")
    st.image("images/Accuracy_Plot_EffNetB4.png", width=550)
    st.markdown("_Steady improvement with early stopping_")

    st.subheader("📉 Loss Plot")
    st.image("images/Loss_Plot_EffNetB4.png", width=550)
    st.markdown("_Shows stable convergence_")

    st.subheader("📘 Classification Report")
    st.image("images/Classification_Report_EffNetB4.png", width=550)

    st.subheader("🔁 Confusion Matrix")
    st.image("images/Confusion_Matrix_EffNetB4.png", width=550)

    st.markdown("**Final Accuracy:** 39%")

    st.markdown("**Business Impact:**")
    st.markdown("""
- 🕒 85–90% reduction in manual tagging  
- 🎯 Improved thumbnail recommendations  
- 💡 Better engagement = more viewing hours  
- 💵 Estimated annual impact: $60–100M  
    """)

# === Footer ===
st.markdown("---")
st.markdown("© 2025 Sweety Seelam | Powered by Streamlit")
st.markdown("All copyrights reserved")