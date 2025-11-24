import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 4

DATA_DIR = os.path.join(BASE_DIR, "data", "sr_pairs")
WEIGHTS_PATH = os.path.join(BASE_DIR, "srcnn_best.pth")


# ============================================================================
# METRICS
# ============================================================================
def psnr(gt, pred, data_range=255.0):
    """Calculate Peak Signal-to-Noise Ratio."""
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim(gt, pred, data_range=255.0):
    """Calculate Structural Similarity Index."""
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x, mu_y = gt.mean(), pred.mean()
    var_x, var_y = gt.var(), pred.var()
    cov_xy = ((gt - mu_x) * (pred - mu_y)).mean()

    num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
    return float(num / den)


# ============================================================================
# MODEL
# ============================================================================
class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)
        )
    
    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_model():
    """Load trained model."""
    model = SRCNN().to(DEVICE)
    
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"Model weights not found at: {WEIGHTS_PATH}")
        st.info("Please train the model first using the Jupyter notebook.")
        return None
    
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# ============================================================================
# IMAGE PROCESSING
# ============================================================================
def tensor_to_img(t):
    """Convert tensor to image."""
    t = t.detach().cpu().numpy().squeeze()
    t = np.clip(t, 0, 1)
    return (t * 255.0).astype(np.uint8)


def downsample_image(img, scale=4):
    """Downsample image to create LR version."""
    h, w = img.shape
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def run_srcnn(lr_img, model):
    """Run super-resolution."""
    h, w = lr_img.shape
    
    # Bicubic upsample
    lr_up = cv2.resize(lr_img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    
    # Prepare tensor
    inp = torch.from_numpy(lr_up).float().unsqueeze(0).unsqueeze(0) / 255.0
    inp = inp.to(DEVICE)
    
    # Run SRCNN
    with torch.no_grad():
        sr = model(inp)
    
    sr_img = tensor_to_img(sr[0])
    return lr_up, sr_img


# ============================================================================
# STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="MRI Super-Resolution",
        layout="wide"
    )
    
    st.title("MRI Super-Resolution with SRCNN")
    st.markdown("""
    This application performs 4x super-resolution on MRI brain scans using a trained SRCNN model.
    Upload your own images or explore the dataset.
    """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    st.sidebar.markdown(f"**Device**: `{DEVICE.upper()}`")
    st.sidebar.markdown(f"**Scale Factor**: `{SCALE}x`")
    st.sidebar.markdown(f"**Model**: SRCNN")
    
    with st.sidebar.expander("Path Information"):
        st.code(f"BASE_DIR: {BASE_DIR}")
        st.code(f"DATA_DIR: {DATA_DIR}")
        st.code(f"WEIGHTS: {WEIGHTS_PATH}")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    st.sidebar.success("Model loaded successfully")
    
    # Mode selection
    st.sidebar.header("Mode Selection")
    mode = st.sidebar.radio(
        "Choose input method:",
        ["Use Dataset", "Upload Image"]
    )
    
    # ========================================================================
    # MODE 1: Use prepared dataset
    # ========================================================================
    if mode == "Use Dataset":
        st.header("Dataset Mode")
        
        lr_dir = os.path.join(DATA_DIR, "LR")
        hr_dir = os.path.join(DATA_DIR, "HR")
        
        if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
            st.error(f"Dataset not found at: {DATA_DIR}")
            st.info("Please run the Jupyter notebook first to create the dataset.")
            return
        
        lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        
        if not lr_files:
            st.error(f"No images found in {lr_dir}")
            return
        
        # Image selection
        filenames = [os.path.basename(p) for p in lr_files]
        choice = st.sidebar.selectbox("Select an image:", filenames, index=0)
        
        lr_path = os.path.join(lr_dir, choice)
        hr_path = os.path.join(hr_dir, choice)
        
        # Load images
        lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        
        if lr_img is None or hr_img is None:
            st.error("Failed to load images")
            return
        
        # Run super-resolution
        with st.spinner("Running super-resolution..."):
            lr_up, sr_img = run_srcnn(lr_img, model)
        
        # Ensure same size
        if hr_img.shape != sr_img.shape:
            hr_img = cv2.resize(hr_img, (sr_img.shape[1], sr_img.shape[0]),
                               interpolation=cv2.INTER_CUBIC)
        
        # Display results
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(lr_up, caption="Bicubic Upsampled", use_column_width=True, clamp=True)
        
        with col2:
            st.image(sr_img, caption="SRCNN Output", use_column_width=True, clamp=True)
        
        with col3:
            st.image(hr_img, caption="Ground Truth", use_column_width=True, clamp=True)
        
        # Metrics
        psnr_bic = psnr(hr_img, lr_up)
        ssim_bic = ssim(hr_img, lr_up)
        psnr_sr = psnr(hr_img, sr_img)
        ssim_sr = ssim(hr_img, sr_img)
        
        st.subheader("Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bicubic PSNR", f"{psnr_bic:.2f} dB")
            st.metric("SRCNN PSNR", f"{psnr_sr:.2f} dB", 
                     delta=f"+{psnr_sr - psnr_bic:.2f} dB")
        
        with col2:
            st.metric("Bicubic SSIM", f"{ssim_bic:.4f}")
            st.metric("SRCNN SSIM", f"{ssim_sr:.4f}",
                     delta=f"+{ssim_sr - ssim_bic:.4f}")
        
        with st.expander("About the metrics"):
            st.markdown("""
            **PSNR (Peak Signal-to-Noise Ratio)**:
            - Measures reconstruction quality in decibels (dB)
            - Higher is better (typically 20-50 dB)
            - Values above 30 dB indicate good quality
            
            **SSIM (Structural Similarity Index)**:
            - Measures perceptual similarity
            - Range: -1 to 1 (1 is identical)
            - Values above 0.9 indicate excellent quality
            """)
    
    # ========================================================================
    # MODE 2: Upload image
    # ========================================================================
    else:
        st.header("Upload Mode")
        
        st.info("Upload a high-resolution MRI slice. It will be downsampled 4x and then super-resolved.")
        
        # Processing mode
        process_mode = st.radio(
            "Processing mode:",
            ["Downsample then super-resolve (shows improvement)", 
             "Use as low-resolution directly"]
        )
        
        # File uploader
        uploaded = st.file_uploader(
            "Upload MRI slice (PNG/JPG)",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded is not None:
            # Read image
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                st.error("Could not decode image")
                return
            
            st.subheader("Uploaded Image")
            st.image(img, caption="Original Image", use_column_width=True, clamp=True)
            
            if process_mode == "Downsample then super-resolve (shows improvement)":
                # Downsample
                lr_img = downsample_image(img, SCALE)
                
                # Run SR
                with st.spinner("Running super-resolution..."):
                    lr_up, sr_img = run_srcnn(lr_img, model)
                
                # Display results
                st.subheader("Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(lr_up, caption="Bicubic Upsampled", 
                            use_column_width=True, clamp=True)
                
                with col2:
                    st.image(sr_img, caption="SRCNN Output", 
                            use_column_width=True, clamp=True)
                
                with col3:
                    st.image(img, caption="Original HR", 
                            use_column_width=True, clamp=True)
                
                # Metrics
                if img.shape != sr_img.shape:
                    img_resized = cv2.resize(img, (sr_img.shape[1], sr_img.shape[0]),
                                            interpolation=cv2.INTER_CUBIC)
                else:
                    img_resized = img
                
                psnr_bic = psnr(img_resized, lr_up)
                ssim_bic = ssim(img_resized, lr_up)
                psnr_sr = psnr(img_resized, sr_img)
                ssim_sr = ssim(img_resized, sr_img)
                
                st.subheader("Quality Metrics (vs Original)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Bicubic PSNR", f"{psnr_bic:.2f} dB")
                    st.metric("SRCNN PSNR", f"{psnr_sr:.2f} dB",
                             delta=f"+{psnr_sr - psnr_bic:.2f} dB")
                
                with col2:
                    st.metric("Bicubic SSIM", f"{ssim_bic:.4f}")
                    st.metric("SRCNN SSIM", f"{ssim_sr:.4f}",
                             delta=f"+{ssim_sr - ssim_bic:.4f}")
            
            else:
                # Use as LR directly
                with st.spinner("Running super-resolution..."):
                    lr_up, sr_img = run_srcnn(img, model)
                
                # Display results
                st.subheader("Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(lr_up, caption="Bicubic Upsampled", 
                            use_column_width=True, clamp=True)
                
                with col2:
                    st.image(sr_img, caption="SRCNN Output", 
                            use_column_width=True, clamp=True)
                
                st.info("No ground truth available - metrics cannot be computed")
        
        else:
            st.info("Please upload an image to begin")


if __name__ == "__main__":
    main()