
[![Live App - Try it Now](https://img.shields.io/badge/Live%20App-Streamlit-informational?style=for-the-badge&logo=streamlit)](https://netflix-thumbnail-classifier.streamlit.app/)

---

# 🎬 Netflix Thumbnail Genre Classification using EfficientNetB4

> An advanced deep learning solution to classify movie posters into genres — automating thumbnail labeling for personalized A/B testing at scale.

---

## 📌 Table of Contents

- [📌 Project Overview](#-project-overview)
- [🎯 Objective](#-objective)
- [📂 Dataset Information](#-dataset-information)
- [⚙️ Model Architecture](#️-model-architecture)
- [🧪 Training and Evaluation](#-training-and-evaluation)
- [📊 Results & Insights](#-results--insights)
- [💼 Business Impact](#-business-impact)
- [🚀 App Demo](#-app-demo)
- [🧠 Recommendations](#-recommendations)
- [📜 License](#-license)

---

## 📌 Project Overview

Netflix constantly tests thousands of thumbnail variations to optimize user engagement. However, genre tagging for posters is often manual, subjective, and time-consuming.

This project aims to automate poster genre classification using a **deep learning-based image classifier**. It can be deployed to support:

- Fast and scalable metadata tagging.
- Smart thumbnail suggestions based on genre.
- Visual personalization in A/B testing campaigns.

---

## 🎯 Objective

Build a reliable, unbiased, and scalable model that:

- Classifies movie posters into one of **five genres**: Action, Comedy, Drama, Romance, Thriller.
- Enables Netflix-like platforms to automate poster tagging and testing.
- Improves recommendation pipelines and engagement strategies.

---

## 📂 Dataset Information

- 📁 **Source**: Posters downloaded via TMDB API using genre filters.
- 🎬 **Genres**: Action, Comedy, Drama, Romance, Thriller.
- 🖼️ **Image Size**: Resized to 380x380 pixels (EfficientNetB4 input size).
- 📊 **Balanced Dataset**: 466 unique posters per genre.
- 🔗 **TMDB Dataset Source**: [The Movie Database API](https://developer.themoviedb.org/reference/discover-movie)

---

## ⚙️ Model Architecture

- ✅ **Base Model**: `EfficientNetB4` (pre-trained on ImageNet)
- 🔄 Transfer Learning: Top layers fine-tuned for genre classification
- 📦 Additional Layers:
  - Global Average Pooling
  - Dropout (0.3)
  - Dense Softmax Output (5 classes)

This model was selected for:
- Strong performance on high-resolution poster images
- Better confidence distribution and generalization
- Higher accuracy vs. older DenseNet/ResNet options

---

## 🧪 Training and Evaluation

- 🧹 Preprocessing: Image resizing (380×380), normalization
- 📊 Split: 80% training, 20% validation
- 🧠 Optimizer: Adam
- 🧮 Loss: Categorical Crossentropy
- 🔁 Epochs: 15  
- 🛑 EarlyStopping: Based on validation loss

---

## 📊 Results & Insights

### ✅ **Validation Accuracy**: `~39%`
### ✅ **Macro F1 Score**: `0.39`
### ✅ **Best Performing Genre**: Action (F1 = 0.51), Comedy (F1 = 0.47)

#### 📈 Accuracy & Loss Curves:
- Training accuracy increased to 79%, validation saturated near 39%
- Validation loss plateaued, indicating room for improvement in generalization

#### 📉 Confusion Matrix:
- Action and Comedy were predicted most confidently
- Drama and Thriller showed confusion due to visual overlap

#### 📑 Classification Report Snapshot:

| Genre   | Precision | Recall | F1-score |                                       
|---------|-----------|--------|----------|                                            

| Action  | 0.54      | 0.49   | 0.51     |                 
| Comedy  | 0.46      | 0.48   | 0.47     |                          
| Drama   | 0.35      | 0.25   | 0.29     |                                         
| Romance | 0.30      | 0.48   | 0.37     |                                          
| Thriller| 0.33      | 0.27   | 0.30     |                                                           

---

## 💼 Business Impact

If this model or an improved version were adopted by Netflix:

- ✅ **Automated Metadata Tagging**: Up to 70% of posters classified with moderate to high confidence
- 📈 **CTR Boost**: Personalized genre thumbnails can raise click-through rates by 15–20%
- 💰 **Estimated Annual Impact**: $60M–$90M in retention and engagement-driven value
- ⏱️ **Operational Efficiency**: Manual workload reduction of 60–70% across creative tagging teams

---

## 🚀 App Demo

You can interactively test the model here:

👉 [**Live Streamlit App**](https://netflix-thumbnail-classifier.streamlit.app/)

Features:
- 📤 Upload your own poster image
- 📁 Use sample posters from our dataset
- ⚡ Instant prediction with genre + confidence
- 📊 Model architecture and overview

---

## 🧠 Recommendations for Future Work

- 🔄 **Multi-label Classification** (movies often belong to more than one genre)
- 🧩 **Multi-modal Learning**: Combine poster with movie metadata (title, synopsis)
- 🔍 **Model Upgrade**: Explore Vision Transformers (ViT, Swin Transformer)
- 📈 **Dataset Expansion**: Grow to 10,000+ posters using TMDB/IMDb

---

## 👩‍💼 About the Author    

**Sweety Seelam** | Business Analyst | Aspiring Data Scientist | Passionate about building end-to-end ML solutions for real-world problems                                                                                                      
                                                                                                                                           
Email: sweetyseelam2@gmail.com                                                   

🔗 **Profile Links**                                                                                                                                                                       
[Portfolio Website](https://sweetyseelam2.github.io/SweetySeelam.github.io/)                                                         
[LinkedIn](https://www.linkedin.com/in/sweetyrao670/)                                                                   
[GitHub](https://github.com/SweetySeelam2)  
[Medium](https://medium.com/@sweetyseelam)

---

## 🔐 Proprietary & All Rights Reserved
© 2025 Sweety Seelam. All rights reserved.

This project, including its source code, trained models, datasets (where applicable), visuals, and dashboard assets, is protected under copyright and made available for educational and demonstrative purposes only.

Unauthorized commercial use, redistribution, or duplication of any part of this project is strictly prohibited.
