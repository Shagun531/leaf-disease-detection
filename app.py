import streamlit as st
from predict import predict_leaf
from PIL import Image
import os
import tempfile

# Remedies mapping: map model class names to short, practical suggestions.
# If your model uses different class names, update the keys to match
# the values in models/class_names.json.
REMEDY_DICT = {
    "Tomato___Bacterial_spot": (
        "Remove and destroy heavily infected leaves; avoid overhead watering; "
        "apply copper-based bactericide where appropriate; rotate crops and "
        "practice good sanitation."
    ),
    "Tomato___Early_blight": (
        "Remove infected debris and lower leaves; use resistant cultivars if available; "
        "apply approved fungicides (e.g., protectants) as per label; improve air circulation."
    ),
    "Tomato___Late_blight": (
        "Destroy infected plants; avoid handling when foliage is wet; apply recommended fungicides "
        "quickly and follow local extension guidance."
    ),
    "Tomato___Leaf_Mold": (
        "Increase ventilation and reduce humidity; remove affected leaves; use fungicidal sprays if needed."
    ),
    "Tomato___healthy": ("Plant appears healthy — continue routine monitoring and good cultural practices.") ,
    "Tomato_Septoria_leaf_spot": (
        "Remove lower infected leaves; mulching and crop rotation help; consider fungicide sprays if severe."
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Inspect underside of leaves; wash plants with water; introduce natural predators or use insecticidal soap/miticide "
        "according to label. Reduce plant stress and dust."
    ),
    "Tomato__Tomato_mosaic_virus": (
        "No chemical cure for viral infections. Remove infected plants to reduce spread; practice strict sanitation and control aphid/whitefly vectors."
    ),
    "Tomato_Tomato_YellowLeaf_Curl_Virus": (
        "Viral disease — remove infected plants and control insect vectors; plant resistant varieties where available."
    ),
    "Potato___Early_blight": (
        "Remove and destroy infected foliage; apply fungicides as recommended; rotate crops and avoid wetting foliage."
    ),
    "Potato___Late_blight": (
        "A serious disease — remove infected plants and debris; apply appropriate fungicides and follow local extension advice."
    ),
    "Potato___healthy": ("Plant appears healthy — continue good cultural practices and monitor regularly."),
    "Pepper_bell__Bacterial_spot": (
        "Remove infected fruit and foliage; avoid overhead irrigation; use copper sprays if recommended; rotate crops."
    ),
    "Pepper_bell__healthy": ("Plant appears healthy — maintain good watering and nutrient practices."),
}

# Page config
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("🌾 Crop Disease AI")
    st.write("Upload a leaf image to detect crop diseases using a fine-tuned MobileNetV2 model.")
    st.divider()
    st.subheader("Model Info")
    st.markdown("""
    - Architecture: *MobileNetV2 (Transfer Learning)*  
    - Framework: *TensorFlow/Keras*  
    - Dataset: Custom Crop Leaves  
    """)
    st.divider()
    st.caption("Developed by 🌱 Leaf Learners")
    st.caption("PBL 5th Sem | Dept. of CSE")

# Main title
st.markdown("<h1 style='text-align:center; color:#2D6A4F;'>🌿 Crop Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#40916C;'>Upload a leaf image and get disease prediction with probabilities.</p>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Convert RGBA to RGB if needed
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Resize image (max width 300px)
    max_width = 300
    w_percent = max_width / float(image.size[0])
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((max_width, h_size))

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name, format="JPEG")
        tmp_path = tmp_file.name


    # Two column layout

    col1, col2 = st.columns([1, 2])  # Left 1/3 for image, Right 2/3 for predictions

    # Left col: Image
    with col1:
        st.image(image, caption="📸 Uploaded Leaf", use_container_width=True)

    # right col: Prediction + Progress bar
    with col2:
        try:
            # Predict all class probabilities
            img_probs = predict_leaf(tmp_path, return_all=True)
            predicted_class = max(img_probs, key=img_probs.get)
            confidence = img_probs[predicted_class]

            # Display main prediction
            st.success(
                f"### ✅ Predicted Disease: *{predicted_class}*  \n"
                f"*Confidence:* {confidence*100:.2f}%"
            )

            # Show suggested remedy if available
            remedy = REMEDY_DICT.get(predicted_class)
            if remedy:
                st.markdown("### 🩺 Suggested Remedy")
                st.info(remedy)
            else:
                st.warning(
                    "No remedy suggestions available for this class. "
                    "You can add remedies in the app code or consult local agricultural extension services."
                )

            # Display probabilities with horizontal bars
            st.markdown("### 📊 Disease Probabilities")
            for cls, prob in img_probs.items():
                # Create 2 columns: one for class name, one for progress bar
                cls_col, bar_col = st.columns([1, 3])
                with cls_col:
                    st.markdown(f"{cls}")
                with bar_col:
                    st.progress(prob)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    os.remove(tmp_path)

# Training result
st.markdown("---")
with st.expander("📊 Model Training Performance (Click to Expand)"):
    if os.path.exists("training_results.png"):
        st.image("training_results.png", caption="Training vs Validation Accuracy/Loss", use_container_width=True)
    else:
        st.warning("⚠ Training results not found. Run train_model.py first to generate graphs.")



# --- IGNORE ---# 
