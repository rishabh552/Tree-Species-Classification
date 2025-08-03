# Tree-Species-Classification-

Project Overview
This project develops an automated image classification system to identify various tree species from leaf images. Accurate and efficient classification supports forestry management, biodiversity conservation, agriculture, and environmental monitoring by reducing reliance on expert manual identification and enabling scalable ecological assessments.

The system leverages deep learning techniques using multiple convolutional neural network architectures. Models are trained via Google Colab using TensorFlow and Keras, and the final application is deployed as a user-friendly web app with Streamlit.

Features
Automated tree species classification from leaf images.
Data preprocessing including duplicate and corrupted data removal.
Data visualization to understand dataset properties.
Utilizes multiple model architectures to benchmark performance.
Achieved up to 69.7% validation accuracy with MobileNetV2.
Lightweight and efficient models suitable for real-time applications.
Easy deployment and usage via a Streamlit interface.

Technologies Used
TensorFlow & Keras - Deep learning model building and training.
Google Colab - Cloud-based training environment.
Streamlit - Web app deployment interface.
NumPy & Pandas - Data manipulation and processing.
Pillow (PIL) - Image reading and preprocessing.
Matplotlib - Data visualization.
Python Standard Libraries - OS/file management and utility functions.

Dataset
Dataset consists of approximately 1600 labeled leaf images representing 30 tree species.

The data was cleaned, duplicates and corrupted files removed.

Data augmentation techniques were applied during training to improve model generalization.

Installation
Clone the repository:
git clone <your-repo-url>
cd tree-species-classification
Set up Python environment (recommended with virtualenv or conda):
pip install -r requirements.txt
Prepare your dataset following the expected folder structure (organized by species).

Launch Streamlit app:

streamlit run app.py
Usage
Open the Streamlit app in your browser.

Upload leaf images or use test images provided.
The app predicts tree species with confidence scores.
Visualizations help interpret model confidence and input data.

Training
Model training scripts are provided using TensorFlow/Keras.
Training conducted on Google Colab using augmented data generators.
Hyperparameter tuning implemented to maximize validation accuracy.
Models saved as .h5 files for easy loading and deployment.

Future Work
Expand dataset to improve model robustness.
Integrate ensemble methods combining multiple models.
Experiment with attention mechanisms and advanced architectures.
Optimize inference speed and memory footprint for mobile deployment.
Implement multi-modal inputs (leaf shapes, textures, environmental data).
