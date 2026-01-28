"""
Streamlit tab for Image training, prediction, and interactive labeling.

Features:
- Generate dataset (wraps `image_data.generate_digit_images`)
- Train CNN from folder (calls `train_cnn`)
- Upload single image for prediction (calls `predict_image`)
- Interactive labeling subtab for labeling predict_examples and adding to labels.csv
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
import shutil
import time

import streamlit as st
from PIL import Image
import json
import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from data_toolkit import image_data
from data_toolkit import image_models

PLOTLY_TEMPLATE = "plotly_white"


def render_image_tab():
    st.header("🖼️ Image Recognition")

    st.markdown("""
    Use this tab to generate synthetic digit images, train a CNN from a labeled CSV,
    upload single images to predict, and interactively label prediction examples.
    """)

    # -------------------- Generate dataset --------------------
    with st.expander("🛠️ Generate Synthetic Dataset", expanded=False):
        out_dir = st.text_input("Output folder (relative to repo)", "test_data/digits")
        n_images = st.number_input("Number of images", value=500, min_value=10, step=10)
        size = st.selectbox("Image size", [64, 96, 128, 160, 224], index=2)
        seed = st.number_input("Random seed", value=42)
        if st.button("Generate images"):
            st.info("Generating images — this may take a few seconds...")
            try:
                image_data.generate_digit_images(output_dir=out_dir, n_images=int(n_images), image_size=int(size), seed=int(seed))
                st.success(f"Generated {n_images} images in {out_dir}")
            except Exception as e:
                st.error(f"Failed to generate images: {e}")

    # -------------------- Train model --------------------
    with st.expander("🚀 Train CNN Model", expanded=True):
        st.write("📁 Browse and select your dataset folder")
        
        # Compact display: show selected folder (read-only) and a small 'Browse' control
        if 'selected_data_folder' not in st.session_state:
            st.session_state.selected_data_folder = None
        if 'show_folder_browser' not in st.session_state:
            st.session_state.show_folder_browser = False

        # Two-column header: selected path (read-only), Browse button and Native Browse
        left, mid, right = st.columns([3, 1, 1])
        selected_display = st.session_state.get('selected_data_folder') or ''
        left.text_input('Selected data folder', value=selected_display, key='data_folder_display', disabled=True)
        if mid.button('Browse'):
            st.session_state.show_folder_browser = not st.session_state.show_folder_browser

        # Native OS folder picker (works when Streamlit runs locally)
        if right.button('Native Browse'):
            try:
                import tkinter as tk
                from tkinter import filedialog
                import threading
                
                # Check if we're on the main thread
                if threading.current_thread() is threading.main_thread():
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)  # Bring dialog to front
                    folder = filedialog.askdirectory(parent=root)
                    root.destroy()
                else:
                    # If not on main thread, try to use a workaround
                    # Create tkinter in a way that might work from non-main thread
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    folder = filedialog.askdirectory(parent=root)
                    root.destroy()
                
                if folder:
                    st.session_state.selected_data_folder = folder
                    st.rerun()
            except RuntimeError as e:
                if "main thread" in str(e).lower():
                    st.warning("⚠️ Native dialog unavailable in this context. Use 'Browse' button or paste path directly below:")
                    manual_path = st.text_input("Paste folder path:", key="manual_path_input")
                    if manual_path:
                        st.session_state.selected_data_folder = manual_path
                        st.rerun()
                else:
                    st.error(f"Native folder dialog error: {e}")
            except Exception as e:
                st.error(f"Native folder dialog not available: {e}")

        # Show browser only when user requests it (keeps UI compact)
        if st.session_state.show_folder_browser:
            with st.expander('Browse folders', expanded=True):
                browse_root = st.selectbox('Browse root', ('Workspace', 'Home', 'Test data'))
                if browse_root == 'Workspace':
                    root_path = Path.cwd()
                elif browse_root == 'Home':
                    root_path = Path.home()
                else:
                    root_path = Path('test_data') if Path('test_data').exists() else Path.cwd()

                # find subfolders up to depth 2 for quick selection
                subfolders = []
                try:
                    for p in sorted(root_path.rglob('*')):
                        if p.is_dir() and len(p.relative_to(root_path).parts) <= 2:
                            subfolders.append(str(p))
                except Exception:
                    subfolders = []

                if subfolders:
                    quick = st.selectbox('Quick pick a folder', ['-- none --'] + subfolders, key='quick_folder_pick')
                    if quick and quick != '-- none --':
                        st.session_state.selected_data_folder = quick

        # Use the selected path
        data_folder = st.session_state.selected_data_folder or None
        if data_folder:
            try:
                folder_path = Path(data_folder)
                imgs = []
                for ext in ('*.png', '*.jpg', '*.jpeg'):
                    imgs += list(folder_path.rglob(ext))
                st.write(f"📊 Found {len(imgs)} images in `{data_folder}`")
                if len(imgs) > 0:
                    cols = st.columns(min(6, len(imgs)))
                    for c, img_path in zip(cols, imgs[:6]):
                        try:
                            # small inline preview
                            c.image(str(img_path), width=100)
                        except Exception:
                            pass
            except Exception as e:
                st.error(f"Error reading selected folder: {e}")
        labels_source = st.radio("Labels source", ("labels.csv in folder", "Upload labels.csv"))
        labels_csv = None
        uploaded_labels = None
        if labels_source == "labels.csv in folder":
            labels_csv = st.text_input("Labels CSV filename (relative to data folder)", "labels.csv")
        else:
            uploaded_labels = st.file_uploader("Upload labels.csv", type=["csv"]) 
            if uploaded_labels is not None:
                st.write(f"Uploaded: {uploaded_labels.name}")
        epochs = st.number_input("Epochs", value=12, min_value=1)
        batch = st.number_input("Batch size", value=32, min_value=1)
        image_size = st.selectbox("Training image size", [64, 96, 128, 160, 224], index=2, key="train_image_size")
        depth = st.slider("CNN depth (number of downsampling blocks)", 2, 6, 4)
        base_filters = st.selectbox("Base filters", [16, 32, 48, 64], index=1)
        dense_units = st.number_input("Dense units", value=256)
        model_out = st.text_input("Model output path (use .keras)", "models/image_cnn.keras")
        save_auto = st.checkbox('Save model automatically', value=False, key='save_auto')
        st.caption("If unchecked, the trained model will remain in memory after training; use 'Save trained model' below to export it to disk.")

        if st.button("Start training"):
            st.info("Starting training — this runs synchronously in the Streamlit worker.")
            try:
                # Validate data folder
                if not data_folder:
                    st.error("Please select a valid data folder before starting training.")
                    raise RuntimeError('No data folder selected')
                if not Path(data_folder).exists() or not Path(data_folder).is_dir():
                    st.error(f"Data folder does not exist or is not a directory: {data_folder}")
                    raise RuntimeError('Invalid data folder')

                # if user uploaded labels.csv, save into data folder
                if uploaded_labels is not None:
                    try:
                        Path(data_folder).mkdir(parents=True, exist_ok=True)
                        save_path = Path(data_folder) / (uploaded_labels.name or 'labels.csv')
                        with open(save_path, 'wb') as fh:
                            fh.write(uploaded_labels.getbuffer())
                        labels_csv_to_use = str(save_path.name)
                    except Exception as e:
                        st.error(f"Failed to save uploaded labels.csv: {e}")
                        raise
                else:
                    labels_csv_to_use = labels_csv or 'labels.csv'

                # Use the model_out and save_auto from above
                model_out_input = model_out

                with st.spinner("Training CNN — this may take minutes depending on CPU/GPU..."):
                    res = image_models.train_cnn(
                        data_dir=data_folder,
                        labels_csv=labels_csv_to_use,
                        image_size=int(image_size),
                        batch_size=int(batch),
                        epochs=int(epochs),
                        model_out=(model_out_input if save_auto else None),
                        save_model=bool(save_auto),
                        base_filters=int(base_filters),
                        depth=int(depth),
                        dense_units=int(dense_units),
                        dropout=0.4,
                    )

                if res.get('model_path'):
                    st.success(f"Training complete — model saved to {res['model_path']}")
                    st.session_state['image_model_path'] = res['model_path']
                else:
                    st.success("Training complete — model not saved automatically.")
                    st.session_state['last_trained_model'] = res['model']
                    st.session_state['last_trained_meta'] = res['class_names']

                if res.get('class_names'):
                    st.session_state['image_class_names'] = res['class_names']
            except Exception as e:
                # Only display the error message (avoid overly verbose tracebacks in UI)
                st.error(f"Training failed: {e}")

    # -------------------- Predict single image --------------------
    with st.expander("🔎 Predict Single Image", expanded=True):
        # allow choosing a model file from the workspace `models/` folder
        try:
            models_dir = Path('models')
            model_candidates = []
            if models_dir.exists():
                # prefer modern .keras files; include .h5 only if no .keras exist
                keras_models = sorted([str(p) for p in models_dir.glob('*.keras')])
                h5_models = sorted([str(p) for p in models_dir.glob('*.h5')])
                model_candidates = keras_models if len(keras_models) > 0 else h5_models
        except Exception:
            model_candidates = []

        if len(model_candidates) > 0:
            sel = st.selectbox("Choose existing model (models/)", ["-- none --"] + model_candidates)
            if st.button("Load selected model") and sel != "-- none --":
                st.session_state['image_model_path'] = sel

        model_path = st.text_input("Model path to use for prediction", st.session_state.get('image_model_path', "models/image_cnn.keras"))

        # keep session model path in sync when user edits the text input
        if st.session_state.get('image_model_path') != model_path:
            st.session_state['image_model_path'] = model_path

        # Attempt to auto-load class names associated with the model
        def _try_load_class_names(model_path_text: str):
            try:
                p = Path(model_path_text)
                cand_files = []
                # common patterns: <model>_class_names.json, <model>.json
                cand_files.append(p.parent / f"{p.stem}_class_names.json")
                cand_files.append(p.parent / f"{p.stem}.json")
                # also search for any json in models/ that mentions the stem
                for j in p.parent.glob('*.json'):
                    if p.stem in j.stem and 'class' in j.stem:
                        cand_files.append(j)

                for c in cand_files:
                    if c.exists():
                        try:
                            return json.loads(c.read_text())
                        except Exception:
                            continue
            except Exception:
                return None
            return None

        # load once per model path
        if st.session_state.get('_last_model_for_classnames') != st.session_state.get('image_model_path'):
            cls = _try_load_class_names(st.session_state.get('image_model_path', ''))
            if cls:
                st.session_state['image_class_names'] = cls
            st.session_state['_last_model_for_classnames'] = st.session_state.get('image_model_path')

        if st.session_state.get('image_class_names'):
            st.write('Loaded class names:', st.session_state.get('image_class_names'))

        # allow either upload or pick an existing file from the data folder
        pick_source = st.radio("Prediction source", ("Upload image", "Choose from data folder"))
        uploaded = None
        chosen_file = None
        if pick_source == "Upload image":
            uploaded = st.file_uploader("Upload image (png/jpg)", type=["png", "jpg", "jpeg"], key="predict_upload")
        else:
            # list images under predict_examples and images in the chosen data folder
            folder = Path(st.text_input("Data folder to pick from", "test_data/digits", key='predict_data_folder'))
            candidates = []
            if folder.exists():
                p1 = folder / 'predict_examples'
                p2 = folder / 'images'
                for p in (p1, p2):
                    if p.exists():
                        candidates += [str(x) for x in sorted(p.glob('*.png'))]
            if len(candidates) == 0:
                st.info('No candidate images found in the data folder')
            else:
                chosen_file = st.selectbox("Choose image to predict", candidates)

        if uploaded is not None or chosen_file:
            try:
                if uploaded is not None:
                    img = Image.open(uploaded).convert('RGB')
                    st.image(img, caption="Uploaded image", width=400)
                    # convert to numpy array and predict from memory to avoid
                    # any file-path caching issues in the Streamlit runtime
                    predict_array = np.array(img)
                    predict_path = None
                else:
                    st.image(chosen_file, caption="Chosen image", width=400)
                    predict_path = chosen_file

                if st.button("Predict image"):
                    st.info("Running prediction...")
                    class_names = st.session_state.get('image_class_names', None)
                    try:
                        if uploaded is not None:
                            out = image_models.predict_array(predict_array, model_path, image_size=int(st.session_state.get('train_image_size', 128)), class_names=class_names)
                        else:
                            out = image_models.predict_image(predict_path, model_path, image_size=int(st.session_state.get('train_image_size', 128)), class_names=class_names)
                        # write a light debug record so we can trace mapping issues
                        try:
                            Path('models').mkdir(parents=True, exist_ok=True)
                            with open(Path('models') / 'predict_debug.log', 'a') as dbg:
                                dbg.write(f"MODEL={model_path} CLS={class_names}\n")
                                dbg.write(f"IMG={predict_path or 'uploaded'} PRED={out['predicted_label']} PROBS={out['probabilities']}\n")
                        except Exception:
                            pass
                        st.success(f"Predicted: {out['predicted_label']}")
                        st.write(out['probabilities'])
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        # ensure models dir exists for logs
                        try:
                            Path('models').mkdir(parents=True, exist_ok=True)
                            with open(Path('models') / 'predict_error.log', 'a') as logf:
                                logf.write('\n===== Prediction error =====\n')
                                logf.write(tb)
                        except Exception:
                            pass
                        print(tb)
                        st.error(f"Prediction failed: {str(e)} — details saved to models/predict_error.log")
                    finally:
                        pass
            except Exception as e:
                st.error(f"Failed to open image: {e}")

        # If training produced an unsaved model, allow saving it now
        if st.session_state.get('last_trained_model') is not None:
            st.write('You have a trained model in memory that was not saved.')
            st.caption("Tip: set 'Save model automatically' before training to save directly, or use the form below to save the in-memory model to a .keras file.")
            save_path = st.text_input('Save trained model to', 'models/unsaved_trained.keras', key='save_trained_path')
            if st.button('Save trained model'):
                try:
                    m = st.session_state.pop('last_trained_model')
                    class_names = st.session_state.pop('last_trained_meta', None)
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    if not str(save_path).endswith('.keras'):
                        save_path = str(save_path) + '.keras'
                    m.save(save_path)
                    # write metadata
                    if class_names is not None:
                        meta_path = Path(save_path).with_suffix('.json')
                        meta_path.write_text(json.dumps({'class_names': class_names}, indent=2))
                    st.success(f'Saved model to {save_path}')
                    st.session_state['image_model_path'] = save_path
                except Exception as e:
                    st.error(f'Failed to save model: {e}')

    # -------------------- Interactive labeling --------------------
    with st.expander("✍️ Interactive Labeling (subtab)", expanded=False):
        st.write("Label images from the `predict_examples` folder and add to training CSV.")
        data_folder = st.text_input("Data folder for interactive labeling", "test_data/digits", key="interactive_data_folder")
        try:
            labels_csv_path = Path(data_folder) / 'labels.csv'
            records = []
            if labels_csv_path.exists():
                import csv
                with open(labels_csv_path, newline='') as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        records.append(r)
        except Exception:
            records = []

        predict_dir = Path(data_folder) / 'predict_examples'
        imgs = []
        if predict_dir.exists():
            imgs = sorted(list(predict_dir.glob('*.png')))

        if len(imgs) == 0:
            st.info(f"No prediction examples found in {predict_dir}. Generate dataset with some predict_examples first.")
        else:
            # persistent index in session
            idx = st.session_state.get('interactive_idx', 0)
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Previous"):
                    idx = max(0, idx - 1)
                if st.button("Next"):
                    idx = min(len(imgs) - 1, idx + 1)
                st.session_state['interactive_idx'] = idx
                st.write(f"Image {idx+1} / {len(imgs)}")

            img_path = imgs[idx]
            with col2:
                st.image(str(img_path), width='stretch')
                # show label buttons
                possible_labels = [str(i) for i in range(10)]
                st.write("Assign label:")
                cols = st.columns(5)
                for i, lab in enumerate(possible_labels):
                    if cols[i % 5].button(lab, key=f'label_{idx}_{lab}'):
                        # move file into images folder and append to labels.csv
                        images_dir = Path(data_folder) / 'images'
                        images_dir.mkdir(parents=True, exist_ok=True)
                        dest = images_dir / img_path.name
                        shutil.move(str(img_path), str(dest))
                        # append to labels.csv
                        import csv
                        with open(labels_csv_path, 'a', newline='') as fh:
                            writer = csv.DictWriter(fh, fieldnames=['filename','label','split'])
                            # write header if file was empty
                            if labels_csv_path.stat().st_size == 0:
                                writer.writeheader()
                            writer.writerow({'filename': str(Path('images')/dest.name), 'label': lab, 'split': 'train'})
                        st.success(f"Labeled {dest.name} as {lab} and added to labels.csv")
                        # refresh list
                        imgs = sorted(list(predict_dir.glob('*.png')))
                        st.session_state['interactive_idx'] = min(idx, max(0, len(imgs)-1))

    st.markdown("---")
    st.caption("Image training uses TensorFlow if available. For large models use a machine with GPU support.")
