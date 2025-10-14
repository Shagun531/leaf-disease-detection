# 🌿 Crop Disease Detection using Transfer Learning (MobileNetV2)

This project uses **Deep Learning** (Transfer Learning with MobileNetV2) to detect **crop diseases** from leaf images.  
It helps farmers or researchers identify plant diseases quickly using image classification.

---

## 📁 Project Structure

Crop-Disease-Detection/
│
├── dataset/ # Folder containing subfolders for each crop disease
│ ├── Tomato___Bacterial_spot/
│ ├── Tomato___Late_blight/
│ └── ...
│
├── models/
│ ├── class_names.json # Saved class names (labels)
│ └── crop_disease_model.h5 # Saved trained model
│
├── save_classes_only.py # Script to extract and save class names
├── train_model.py # Main training script
├── requirements.txt # Python dependencies
├── training_results.png # Accuracy/Loss graph after training
├── app.py # Streamlit web based application
├── predict.py # Prediction engine for leaf images


---

**⚙️ Features**

✅ Uses MobileNetV2 pre-trained on ImageNet
✅ Fine-tunes only the last 60 layers for crop disease classification
✅ Includes Dropout and L2 regularization to reduce overfitting
✅ Saves model and class names for later predictions
✅ Visualizes training accuracy and loss
✅ Web-based interface using Streamlit

---

## 🧠 How It Works

1. **Data Loading**  
   Images are loaded from `dataset/` using `tf.keras.utils.image_dataset_from_directory()`  
   - Automatically splits into training and validation (80/20).  
   - Auto-generates class labels based on folder names.  

2. **Feature Extraction**  
   Pre-trained **MobileNetV2** (ImageNet weights) extracts image features.  
   Early layers are **frozen** to retain general visual knowledge.  

3. **Fine-Tuning**  
   The last 60 layers of MobileNetV2 are made trainable for disease-specific learning.  

4. **Classification Layers**
   - Global Average Pooling → Summarizes features  
   - Dropout(0.5) → Reduces overfitting  
   - Dense(128, ReLU) + L2 Regularization → Learns complex patterns  
   - Dense(N, Softmax) → Outputs probabilities per class  

5. **Training**
   - Optimizer: Adam (lr=0.0001)  
   - Loss: Sparse Categorical Crossentropy  
   - Metric: Accuracy  

6. **Visualization**
   - Accuracy & Loss plotted and saved to `training_results.png`

---

## 🖥️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crop-disease-detection.git
   cd crop-disease-detection
   
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    
3. Prepare your dataset:
   dataset/
   ├──Pepper__bell___Bacterial_spot/
   ├──Pepper__bell___healthy/
   ├──Potato___Early_blight/
   ├──Potato___healthy/
   ├──Potato___Late_blight/
   ├──Tomato__Target_Spot/
   ├──Tomato__Tomato_mosaic_virus/
   ├──Tomato__Tomato_YellowLeaf__Curl_Virus/
   ├── Tomato___Bacterial_spot/
   ├── Tomato___Early_blight/
   ├── Tomato___Late_blight/
   ├── Tomato___Leaf_Mold/
   ├── Tomato___healthy/
   ├──Tomato_Septoria_leaf_spot/
   ├──Tomato_Spider_mites_Two_spotted_spider_mite/

      💡 Optional: If dataset is too large, use Git LFS:
            git lfs install
            git lfs track "dataset/*"
            git add .gitattributes
            git add dataset/
            git commit -m "Add dataset with Git LFS"
            git push

       For someone cloning the repo:
            git clone <repo_url>
            git lfs pull



**🚀 Training the Model**
Step 1 — Save Class Names
```bash
python save_classes_only.py
This creates:
models/class_names.json

Step 2 — Train the Model
```bash
python train_model.py

This script:
Loads the dataset
Builds the model
Trains for 20 epochs
Saves the trained model in models/
Generates and saves the accuracy/loss plots

Example output:
```bash
✅ Model saved to models/crop_disease_model.h5
🎉 Training complete and plots saved!

🖼️ Running the Web App
```bash
streamlit run app.py

Upload a leaf image
View predicted disease and class probabilities
See training performance in the expandable section

📊 Results
Model: MobileNetV2 (Fine-tuned)
Epochs: 20
Optimizer: Adam (lr=0.0001)
Accuracy: ~95–98% (depending on dataset quality)
Output Graph: training_results.png # Accuracy/Loss graph after training

🧰 Requirements
Add these to your requirements.txt:
```shell
matplotlib==3.8.1
numpy==1.26.0
Pillow==10.1.0
streamlit==1.27.0
tensorflow==2.15.0


🪴 Acknowledgments
TensorFlow
MobileNetV2 Paper
PlantVillage Dataset
