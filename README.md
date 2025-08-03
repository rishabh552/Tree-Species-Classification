Tree Species Classification

Overview
An AI-powered image classification system to identify tree species from leaf images. This tool supports forestry management, biodiversity conservation, and environmental monitoring by automating tree species identification and reducing the need for expert manual labeling.

Features
Automated classification of 30 tree species from leaf photos.
Data preprocessing with duplicate removal and visualization.
Trained using deep learning models with Batch Normalization for stability.
Deployed as an easy-to-use Streamlit web app.
Lightweight models suitable for real-time predictions.

Technologies
TensorFlow & Keras for model building and training.
Google Colab for training on cloud GPUs.
Streamlit for app deployment.
NumPy, Pandas, Pillow, and Matplotlib for data processing and visualization.

Setup & Usage
Clone the repo and install dependencies.
Prepare dataset in expected folder structure (images sorted by species).
Run training scripts in Colab (optional).
Launch the Streamlit app and upload leaf images to get species predictions.

Future Enhancements
Expand dataset size and augmentation.
Integrate ensemble learning for better accuracy.
Optimize for mobile deployment and faster inference.
