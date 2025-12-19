"""
Biomass Segmentation Tab for Streamlit UI
Provides interface for U-Net based semantic segmentation
of aquatic biomass (Dreissena mussels, Cladophora algae).
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import io

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Check for TensorFlow availability
try:
    from biomass_segmentation import BiomassSegmentation, create_sample_dataset, TF_AVAILABLE
except ImportError:
    TF_AVAILABLE = False


def render_biomass_tab():
    """Render biomass segmentation tab"""
    st.header("🌿 Biomass Segmentation")
    st.caption("U-Net based semantic segmentation for aquatic biomass analysis (Dreissena mussels & Cladophora algae)")
    
    if not TF_AVAILABLE:
        st.error("⚠️ TensorFlow is not installed. Install with: `pip install tensorflow`")
        st.info("This feature requires TensorFlow for deep learning-based image segmentation.")
        return
    
    # Initialize session state for model
    if 'biomass_model' not in st.session_state:
        st.session_state.biomass_model = None
    if 'biomass_results' not in st.session_state:
        st.session_state.biomass_results = {}
    
    # Sidebar configuration
    st.sidebar.markdown("### 🌿 Segmentation Settings")
    
    # Model configuration
    with st.expander("🔧 Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            image_size = st.selectbox(
                "Image Size",
                [128, 256, 512],
                index=1,
                help="Input image resolution (higher = more detail but slower)"
            )
            
            n_classes = st.selectbox(
                "Number of Classes",
                [3, 4],
                index=0,
                help="3 = Background/Dreissena/Cladophora, 4 = adds Other Vegetation"
            )
        
        with col2:
            encoder = st.selectbox(
                "Encoder Backbone",
                ["standard", "vgg16"],
                index=0,
                help="VGG16 uses transfer learning (requires downloading weights)"
            )
            
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4
            )
        
        if st.button("🔨 Build Model", width='stretch'):
            with st.spinner("Building U-Net model..."):
                try:
                    model = BiomassSegmentation(
                        input_shape=(image_size, image_size, 3),
                        n_classes=n_classes,
                        encoder=encoder
                    )
                    result = model.build_model(learning_rate=learning_rate)
                    
                    if 'error' in result:
                        st.error(f"Failed to build model: {result['error']}")
                    else:
                        st.session_state.biomass_model = model
                        st.success(f"✅ Model built: {result['n_parameters']:,} parameters")
                        st.info(f"Input: {result['input_shape']}, Classes: {result['n_classes']}")
                except Exception as e:
                    st.error(f"Error building model: {str(e)}")
    
    # Training section
    st.markdown("---")
    st.subheader("🎯 Training")
    
    with st.expander("📚 Training Data", expanded=False):
        st.markdown("""
        **Training Data Requirements:**
        - Images: RGB images (JPG/PNG)
        - Masks: Grayscale where pixel values represent class labels:
            - 0 = Background (water, substrate)
            - 1 = Dreissena (zebra/quagga mussels)
            - 2 = Cladophora (filamentous algae)
            - 3 = Other vegetation (optional)
        """)
        
        # Upload training images
        train_images = st.file_uploader(
            "Upload Training Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="train_images"
        )
        
        train_masks = st.file_uploader(
            "Upload Training Masks",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="train_masks"
        )
        
        # Demo with synthetic data
        st.markdown("---")
        st.markdown("**Or use synthetic demo data:**")
        
        n_demo_samples = st.slider("Demo samples", 10, 100, 20)
        
        if st.button("🎲 Generate Demo Data & Train", width='stretch'):
            if st.session_state.biomass_model is None:
                st.warning("⚠️ Please build model first")
            else:
                with st.spinner("Generating demo data and training..."):
                    try:
                        # Generate demo data
                        img_size = st.session_state.biomass_model.input_shape[0]
                        X_train, y_train = create_sample_dataset(n_demo_samples, (img_size, img_size))
                        
                        # Split for validation
                        split_idx = int(0.8 * len(X_train))
                        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
                        X_train, y_train = X_train[:split_idx], y_train[:split_idx]
                        
                        st.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
                        
                        # Train
                        epochs = st.slider("Training Epochs", 5, 50, 10, key="demo_epochs")
                        result = st.session_state.biomass_model.train(
                            X_train, y_train,
                            X_val, y_val,
                            epochs=epochs,
                            batch_size=4,
                            augment=True
                        )
                        
                        if 'error' in result:
                            st.error(f"Training failed: {result['error']}")
                        else:
                            st.success(f"✅ Training complete! Final accuracy: {result['final_accuracy']:.4f}")
                            st.session_state.biomass_results['training'] = result
                            
                            # Plot training history
                            import plotly.graph_objects as go
                            history = result.get('history', {})
                            
                            fig = go.Figure()
                            if 'loss' in history:
                                fig.add_trace(go.Scatter(y=history['loss'], name='Train Loss'))
                            if 'val_loss' in history:
                                fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss'))
                            fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
                            st.plotly_chart(fig, width='stretch')
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Prediction section
    st.markdown("---")
    st.subheader("🔍 Prediction & Analysis")
    
    with st.expander("📷 Analyze Image", expanded=True):
        uploaded_image = st.file_uploader(
            "Upload image for segmentation",
            type=['jpg', 'jpeg', 'png'],
            key="predict_image"
        )
        
        pixel_size = st.number_input(
            "Pixel size (mm)",
            min_value=0.01,
            value=1.0,
            step=0.1,
            help="Physical size of each pixel for area calculation"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", width='stretch')
            
            if st.button("🔬 Analyze Biomass", width='stretch'):
                if st.session_state.biomass_model is None:
                    st.warning("⚠️ Please build and train model first")
                else:
                    with st.spinner("Analyzing biomass..."):
                        try:
                            # Convert to numpy
                            img_array = np.array(image)
                            
                            # Resize to model input size
                            img_size = st.session_state.biomass_model.input_shape[0]
                            image_resized = np.array(image.resize((img_size, img_size)))
                            
                            if image_resized.max() > 1:
                                image_resized = image_resized / 255.0
                            
                            # Analyze
                            results = st.session_state.biomass_model.analyze_biomass(
                                image_resized, 
                                pixel_size_mm=pixel_size
                            )
                            
                            if 'error' in results:
                                st.error(f"Analysis failed: {results['error']}")
                            else:
                                st.session_state.biomass_results['analysis'] = results
                                
                                # Display results
                                st.success("✅ Analysis complete!")
                                
                                # Summary metrics
                                summary = results.get('summary', {})
                                mcol1, mcol2, mcol3 = st.columns(3)
                                mcol1.metric("Total Biomass", f"{summary.get('total_biomass_coverage', 0):.2f}%")
                                mcol2.metric("Dreissena", f"{summary.get('dreissena_coverage', 0):.2f}%")
                                mcol3.metric("Cladophora", f"{summary.get('cladophora_coverage', 0):.2f}%")
                                
                                # Class breakdown
                                st.markdown("#### Class Breakdown")
                                class_data = results.get('classes', {})
                                class_df = pd.DataFrame([
                                    {
                                        'Class': name,
                                        'Coverage (%)': f"{info.get('coverage_percent', 0):.2f}",
                                        'Area (cm²)': f"{info.get('area_cm2', 0):.4f}",
                                        'Pixels': info.get('pixels', 0)
                                    }
                                    for name, info in class_data.items()
                                ])
                                st.dataframe(class_df, width='stretch')
                                
                                # Visualization
                                import matplotlib.pyplot as plt
                                fig = st.session_state.biomass_model.visualize_prediction(
                                    image_resized,
                                    show_overlay=True,
                                    alpha=0.5
                                )
                                if fig:
                                    st.pyplot(fig, width='stretch')
                                    plt.close(fig)
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    # Model management
    st.markdown("---")
    st.subheader("💾 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model filename", value="biomass_model.h5")
        if st.button("💾 Save Model", width='stretch'):
            if st.session_state.biomass_model is not None:
                result = st.session_state.biomass_model.save_model(model_name)
                if 'error' in result:
                    st.error(f"Save failed: {result['error']}")
                else:
                    st.success(f"✅ Model saved to {model_name}")
            else:
                st.warning("No model to save")
    
    with col2:
        model_file = st.file_uploader("Load saved model", type=['h5'], key="load_model")
        if model_file and st.button("📂 Load Model", width='stretch'):
            # Save uploaded file temporarily
            temp_path = f"temp_{model_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(model_file.read())
            
            model = BiomassSegmentation()
            result = model.load_model(temp_path)
            
            if 'error' in result:
                st.error(f"Load failed: {result['error']}")
            else:
                st.session_state.biomass_model = model
                st.success("✅ Model loaded successfully!")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Help section
    with st.expander("ℹ️ About Biomass Segmentation"):
        st.markdown("""
        ### Biomass Segmentation for Aquatic Ecosystems
        
        This tool uses U-Net, a convolutional neural network architecture designed for 
        biomedical image segmentation, adapted for ecological analysis.
        
        **Supported Organisms:**
        - **Dreissena** (Zebra/Quagga mussels): Invasive freshwater mussels
        - **Cladophora**: Filamentous green algae
        - **Other vegetation**: General aquatic plants
        
        **Workflow:**
        1. **Build Model**: Configure and compile the U-Net architecture
        2. **Train**: Provide labeled images and masks for training
        3. **Predict**: Analyze new images for biomass coverage
        
        **Tips for Best Results:**
        - Use consistent lighting in training images
        - Include diverse examples (different substrates, densities)
        - Aim for 50+ training images for reliable results
        - Use higher resolution (512x512) for detailed analysis
        """)
