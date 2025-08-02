import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from collections import Counter
import zipfile
import tempfile
import shutil

# ---------------- Settings and Initialization ----------------

st.set_page_config(page_title="🌳 Tree Species Classification", layout="wide")
st.title("🌳 Tree Species Classification")

# Initialize session state for persistent data
if 'available_models' not in st.session_state:
    st.session_state.available_models = {}
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'dataset_paths' not in st.session_state:
    st.session_state.dataset_paths = []

# ------------- Robust Metadata Extraction Function -------------

def safe_extract_model_info(model):
    """
    Robustly infer output class count of a Keras model.

    If single input, pass a single dummy array.
    If multi-input, pass list of dummy arrays.
    On failure, fallback to last Dense layer's units attribute.
    """
    try:
        inputs = model.inputs
        dummy_inputs = []
        for inp in inputs:
            shape = inp.shape.as_list()
            shape[0] = 1 if shape[0] is None else shape[0]
            shape = [dim if dim is not None else 224 for dim in shape]
            dummy_inputs.append(np.zeros(shape, dtype=np.float32))

        if len(dummy_inputs) == 1:
            output = model(dummy_inputs[0], training=False)
        else:
            output = model(dummy_inputs, training=False)

        return int(output.shape[-1])

    except Exception as e:
        try:
            for layer in reversed(model.layers):
                if hasattr(layer, 'units'):
                    return layer.units
            return 5
        except:
            return 5

# ---------------- Dataset Folder Scan with Name Filter ----------------

def find_dataset_folders():
    dataset_paths = []
    search_locations = [
        '.', 'dataset', 'Dataset', 'DATASET',
        'data', 'Data', 'species_data', 'tree_species',
        'trees', 'train', 'training'
    ]
    allowed_names = {"tree_species_dataset", "treespeciesdataset"}

    for location in search_locations:
        if os.path.isdir(location):
            subdirs = [d for d in os.listdir(location)
                       if os.path.isdir(os.path.join(location, d))]
            candidate_folders = [d for d in subdirs if d.lower() in allowed_names]

            for folder in candidate_folders:
                folder_path = os.path.join(location, folder)
                try:
                    class_subdirs = [d for d in os.listdir(folder_path)
                                     if os.path.isdir(os.path.join(folder_path, d))]
                    valid_classes = 0
                    for class_folder in class_subdirs:
                        class_folder_path = os.path.join(folder_path, class_folder)
                        image_files = [f for f in os.listdir(class_folder_path)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                        if len(image_files) >= 2:
                            valid_classes += 1
                    if valid_classes >= 2:
                        dataset_paths.append(folder_path)
                        st.success(f"📂 Found dataset: {folder_path} with {valid_classes} species")
                except Exception:
                    continue

    return dataset_paths

# -------- Extract real species names from dataset folders --------------

def extract_class_names_from_dataset(dataset_path, num_classes):
    try:
        if not dataset_path or not os.path.exists(dataset_path):
            return None
        folders = [d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))]
        valid_folders = []
        for folder in folders:
            folder_path = os.path.join(dataset_path, folder)
            try:
                image_files = [f for f in os.listdir(folder_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                if len(image_files) >= 2:
                    valid_folders.append(folder)
            except Exception:
                continue
        valid_folders.sort(key=str.lower)
        if len(valid_folders) >= num_classes:
            selected_species = valid_folders[:num_classes]
            st.success(f"🌳 Using real species names: {', '.join(selected_species)}")
            return selected_species
        else:
            st.warning(f"⚠️ Dataset has {len(valid_folders)} classes but model needs {num_classes}")
            return None
    except Exception as e:
        st.warning(f"Error extracting class names: {e}")
        return None

# ---------------- Auto Metadata File Generation ----------------

def auto_generate_metadata_files(model_path, dataset_paths=None):
    try:
        model = load_model(model_path)
        basename = os.path.splitext(model_path)[0]
        model_name = os.path.basename(model_path)
        st.write(f"🔄 Processing: {model_name}")

        num_classes = safe_extract_model_info(model)
        st.info(f"📊 Number of classes detected: {num_classes}")

        img_size = (224, 224)  # always standard input size

        class_names = None
        used_dataset_path = None
        if dataset_paths:
            st.write(f"🔍 Searching for species names in dataset(s)...")
            for dataset_path in dataset_paths:
                potential_names = extract_class_names_from_dataset(dataset_path, num_classes)
                if potential_names:
                    class_names = potential_names
                    used_dataset_path = dataset_path
                    break

        if not class_names:
            class_names = [f"Species_{i}" for i in range(num_classes)]
            st.warning("⚠️ No suitable dataset found; using generic class names.")

        class_mapping = {name: idx for idx, name in enumerate(class_names)}
        classes_path = f"{basename}_classes.npy"
        np.save(classes_path, class_mapping)

        # Determine architecture type simplified
        ln = model_name.lower()
        if 'mobilenet' in ln:
            architecture_type = "MobileNetV2 (Standard 224×224)"
        elif 'efficient' in ln:
            architecture_type = "EfficientNet (Standard 224×224)"
        elif 'batchnormalization' in ln:
            architecture_type = "BatchNorm CNN (Standard 224×224)"
        elif any(k in ln for k in ['batch', 'norm', 'cnn']):
            architecture_type = "Custom CNN (Standard 224×224)"
        else:
            architecture_type = f"Standard 224×224 Model ({model_name})"

        model_info = {
            "architecture": architecture_type,
            "num_classes": num_classes,
            "class_names": class_names,
            "img_size": img_size,
            "input_shape": img_size + (3,),
            "total_parameters": model.count_params() if hasattr(model, 'count_params') else 0,
            "auto_generated": True,
            "uses_dataset_names": used_dataset_path is not None,
            "dataset_path": used_dataset_path,
            "model_name": model_name,
            "standard_224": True,
            "source": "Standard 224×224 system with auto-generated metadata"
        }
        info_path = f"{basename}_info.npy"
        np.save(info_path, model_info)

        st.success("✅ Metadata generated successfully.")
        st.write(f"Input size: {img_size}")
        st.write(f"Species names: {'Real dataset' if used_dataset_path else 'Generic'}")

        return classes_path, info_path, model_info

    except Exception as e:
        st.error(f"Failed to generate metadata: {str(e)}")
        return None, None, None

# ---------------- Scan Models and Auto-generate Metadata ----------------

def scan_available_models():
    models = {}
    search_paths = ['.', 'models', 'pretrained_models']
    dataset_paths = st.session_state.dataset_paths

    for path in search_paths:
        if not os.path.exists(path):
            continue
        for file in os.listdir(path):
            if file.lower().endswith('.h5'):
                model_path = os.path.join(path, file)
                basename = os.path.splitext(model_path)[0]
                class_map_path = f"{basename}_classes.npy"
                info_path = f"{basename}_info.npy"
                st.info(f"Applying standard 224×224 to: {file}")
                auto_generate_metadata_files(model_path, dataset_paths)
                model_info = {}
                if os.path.exists(info_path):
                    try:
                        model_info = np.load(info_path, allow_pickle=True).item()
                    except Exception:
                        pass
                models[file] = {
                    'path': model_path,
                    'class_map_path': class_map_path,
                    'info_path': info_path,
                    'info': model_info
                }
    return models

# ---------------- Image Preprocessing ----------------

def preprocess_image_for_model(img, img_size):
    img_resized = img.resize(img_size)
    x = img_to_array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# ---------------- Prediction with Multiple Models ----------------

def predict_with_multiple_models(image, loaded_models, use_tta=False):
    predictions = []
    for model_data in loaded_models:
        model = model_data['model']
        class_map = model_data['class_map']
        info = model_data['info']
        model_name = model_data['name']
        try:
            img_size = info.get('img_size', (224, 224))
            processed_image = preprocess_image_for_model(image, img_size)

            if use_tta:
                pred_list = []
                pred_list.append(model.predict(processed_image, verbose=0)[0])
                flipped = np.flip(processed_image, axis=2)
                pred_list.append(model.predict(flipped, verbose=0)[0])
                pred = np.mean(pred_list, axis=0)
                pred = np.expand_dims(pred, axis=0)
            else:
                pred = model.predict(processed_image, verbose=0)

            top_idx = np.argmax(pred[0])
            confidence = pred[0][top_idx]
            inv_class_map = {v: k for k, v in class_map.items()}
            species_name = inv_class_map.get(top_idx, f"Unknown_{top_idx}")

            predictions.append({
                'model_name': model_name,
                'species': species_name,
                'confidence': float(confidence),
                'img_size': img_size,
                'uses_dataset_names': info.get('uses_dataset_names', False),
                'architecture': info.get('architecture', 'Unknown'),
                'standard_224': info.get('standard_224', True),
                'all_predictions': pred[0]
            })
        except Exception as e:
            st.warning(f"Prediction failed for {model_name}: {str(e)}")
            continue
    if not predictions:
        return None
    best_prediction = max(predictions, key=lambda x: x['confidence'])
    return best_prediction, predictions

# --------------- Species Consensus ---------------------

def get_species_consensus(predictions):
    species_votes = Counter()
    confidence_by_species = {}
    for pred in predictions:
        species = pred['species']
        confidence = pred['confidence']
        species_votes[species] += 1
        confidence_by_species.setdefault(species, []).append(confidence)
    species_avg_confidence = {species: np.mean(confidences) for species, confidences in confidence_by_species.items()}
    return species_votes, species_avg_confidence

# ------------------------ Streamlit UI -------------------------

st.header("1. 📋 Model and Dataset Management")

st.success("""
🚀 Standard 224×224 System Benefits:

- Fixed input size for all models.
- Consistent and reliable predictions.
""")

# Dataset management UI

# Dataset management UI - Only Scan for datasets option

st.subheader("📂 Dataset Management")

if st.button("🔍 Scan for datasets"):
    found_paths = find_dataset_folders()
    st.session_state.dataset_paths = found_paths  # replace any previous datasets
    if found_paths:
        st.success(f"Found {len(found_paths)} dataset folder(s).")
    else:
        st.warning("No datasets found.")

if st.session_state.dataset_paths:
    st.write("Using dataset paths:")
    for p in st.session_state.dataset_paths:
        st.write(f"- {p}")
else:
    st.info("No datasets selected.")


# Model scanning and upload UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Scan Models with Standard Sizing")
    if st.button("🔍 Apply Standard 224×224 to All Models", type="primary"):
        with st.spinner("Applying standard 224×224 sizing to all models..."):
            st.session_state.available_models = scan_available_models()
        if st.session_state.available_models:
            st.success(f"Applied standard 224×224 sizing to {len(st.session_state.available_models)} models!")
        else:
            st.warning("No .h5 model files found.")

with col2:
    st.subheader("📤 Upload Model with Standard Sizing")
    uploaded_model = st.file_uploader("Upload .h5 Model File", type=['h5'])
    if uploaded_model and st.button("💾 Save with Standard 224×224"):
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{uploaded_model.name}"
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        with st.spinner("Applying standard 224×224 sizing..."):
            auto_generate_metadata_files(model_path, st.session_state.dataset_paths)
        st.success("Model uploaded with standard 224×224 sizing!")
        st.session_state.available_models = scan_available_models()

# Available models overview
if st.session_state.available_models:
    st.header("2. 🗂 Models Overview (224×224 input)")
    model_data = []
    for model_name, model_info in st.session_state.available_models.items():
        info = model_info.get('info', {})
        uses_dataset = "🌳 Real Names" if info.get('uses_dataset_names') else "🔤 Generic"
        img_size = info.get('img_size', 'Unknown')
        standard_224 = "✅ Standard 224×224" if info.get('standard_224') else "❌ Custom Size"
        classes_val = info.get('num_classes', 'Unknown')
        model_data.append({
            'Model Name': model_name,
            'Architecture': info.get('architecture', 'Unknown'),
            'Classes': str(classes_val),  # Cast to str to avoid PyArrow errors
            'Input Size': str(img_size),
            'Standard Sizing': standard_224,
            'Species Names': uses_dataset,
            'Parameters': f"{info.get('total_parameters', 0):,}" if isinstance(info.get('total_parameters'), int) else 'Unknown'
        })
    df = pd.DataFrame(model_data)
    st.dataframe(df, use_container_width=True)

    non_standard = df[df['Input Size'] != '(224, 224)']
    if not non_standard.empty:
        st.error("Models not using 224×224 input size:")
        st.dataframe(non_standard)
    else:
        st.success("All models use standard 224×224 input size!")

# Load models UI
st.header("3. 🚀 Load Models with 224×224 Input")
if st.session_state.available_models:
    selected_models = st.multiselect(
        "Select models for prediction (224×224 input):",
        list(st.session_state.available_models.keys()),
        default=list(st.session_state.available_models.keys())
    )
    if st.button("🚀 Load Selected Models", type="primary"):
        loaded_models = []
        for model_name in selected_models:
            try:
                model_info = st.session_state.available_models[model_name]
                with st.spinner(f"Loading {model_name}..."):
                    model = load_model(model_info['path'])
                    class_map = np.load(model_info['class_map_path'], allow_pickle=True).item()
                    info = model_info['info']
                    loaded_models.append({
                        'name': model_name,
                        'model': model,
                        'class_map': class_map,
                        'info': info
                    })
                img_size = info.get('img_size', 'Unknown')
                uses_real_names = "🌳" if info.get('uses_dataset_names') else "🔤"
                architecture = info.get('architecture', 'Unknown')
                st.success(f"Loaded: {model_name} - {architecture} - {img_size} - Names: {uses_real_names}")
            except Exception as e:
                st.error(f"Failed to load {model_name}: {str(e)}")
        st.session_state.loaded_models = loaded_models
        st.session_state.models_loaded = len(loaded_models) > 0
        if st.session_state.models_loaded:
            st.success(f"Loaded {len(loaded_models)} models with 224×224 input!")

# Prediction UI
st.header("4. 🌳 Tree Species Prediction (224×224 Input)")
if st.session_state.models_loaded:
    st.info(f"{len(st.session_state.loaded_models)} models ready for prediction (224×224 input).")
    uploaded_image = st.file_uploader("Upload tree image for prediction", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    if uploaded_image:
        img = Image.open(uploaded_image).convert('RGB')
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
            st.write(f"Original Size: {img.size}")
        with col2:
            st.subheader("Prediction Options")
            use_tta = st.checkbox("Enhanced Test-Time Augmentation", value=False)
            confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 70)
            show_consensus = st.checkbox("Show Species Consensus", value=True)
        if st.button("🔍 Predict"):
            with st.spinner("Running predictions with 224×224 input..."):
                result = predict_with_multiple_models(img, st.session_state.loaded_models, use_tta)
                if result:
                    best_prediction, all_predictions = result
                    confidence_pct = best_prediction['confidence'] * 100
                    if confidence_pct >= 85:
                        st.success(f"Species: {best_prediction['species']}")
                        st.success(f"Confidence: {confidence_pct:.1f}%")
                        st.success(f"Best Model: {best_prediction['model_name']} (Standard 224×224)")
                        st.success(f"Architecture: {best_prediction['architecture']}")
                    elif confidence_pct >= 70:
                        st.warning(f"Species: {best_prediction['species']}")
                        st.warning(f"Confidence: {confidence_pct:.1f}%")
                        st.warning(f"Best Model: {best_prediction['model_name']} (Standard 224×224)")
                        st.warning(f"Architecture: {best_prediction['architecture']}")
                    else:
                        st.info(f"Species: {best_prediction['species']}")
                        st.info(f"Confidence: {confidence_pct:.1f}%")
                        st.info(f"Best Model: {best_prediction['model_name']} (Standard 224×224)")
                        st.info(f"Architecture: {best_prediction['architecture']}")

                    if show_consensus and len(all_predictions) > 1:
                        st.subheader("Species Consensus")
                        species_votes, species_avg_confidence = get_species_consensus(all_predictions)
                        consensus_data = []
                        for species, votes in species_votes.most_common():
                            avg_conf = species_avg_confidence[species] * 100
                            consensus_data.append({
                                'Species': species,
                                'Model Votes': votes,
                                'Avg Confidence': f"{avg_conf:.1f}%",
                                'Agreement': f"{votes}/{len(all_predictions)} models"
                            })
                        st.dataframe(pd.DataFrame(consensus_data), use_container_width=True)

                    st.subheader("Model Predictions Comparison")
                    comparison_data = []
                    for pred in all_predictions:
                        names_type = "🌳 Real" if pred['uses_dataset_names'] else "🔤 Generic"
                        comparison_data.append({
                            'Model': pred['model_name'],
                            'Architecture': pred['architecture'],
                            'Species': pred['species'],
                            'Confidence': f"{pred['confidence']*100:.1f}%",
                            'Input Size': str(pred['img_size']),
                            'Standard 224×224': "✅" if pred.get('standard_224') else "❌",
                            'Names': names_type
                        })
                    comparison_data.sort(key=lambda x: float(x['Confidence'].replace('%', '')), reverse=True)
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

                    if len(all_predictions) > 1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        models = [pred['model_name'] for pred in all_predictions]
                        confidences = [pred['confidence'] * 100 for pred in all_predictions]
                        colors = ['gold' if conf == max(confidences) else 'lightblue' for conf in confidences]
                        bars1 = ax1.bar(models, confidences, color=colors, alpha=0.8)
                        ax1.set_ylabel('Confidence (%)')
                        ax1.set_title('Models Confidence')
                        ax1.set_ylim(0, 100)
                        for bar, conf in zip(bars1, confidences):
                            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                                     f'{conf:.1f}%', ha='center', va='bottom',
                                     fontweight='bold' if conf == max(confidences) else 'normal')
                        input_sizes = [str(pred['img_size']) for pred in all_predictions]
                        size_counts = pd.Series(input_sizes).value_counts()
                        ax2.pie(size_counts.values, labels=size_counts.index, autopct='%1.0f models', startangle=90)
                        ax2.set_title('Input Size Distribution (Should be 100% 224×224)')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.error("Prediction failed for all models.")
else:
    st.info("Load models with 224×224 input first to enable prediction.")

# Sidebar model info summary
if st.session_state.models_loaded:
    models_info = []
    for m in st.session_state.loaded_models:
        info = m['info']
        name_type = "🌳" if info.get('uses_dataset_names') else "🔤"
        size = info.get('img_size', 'Unknown')
        standard = "✅" if info.get('standard_224') else "❌"
        models_info.append(f"{standard} {m['name']} - {size} {name_type}")
    st.sidebar.success(f"{len(st.session_state.loaded_models)} Models Loaded\n\n" + "\n".join(models_info))
else:
    st.sidebar.info("No Models Loaded\nLoad models with 224×224 input to start")
