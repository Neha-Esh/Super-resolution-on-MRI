# MRI Super-Resolution using SRCNN

Deep learning-based 4x super-resolution for MRI brain scans using Super-Resolution Convolutional Neural Network (SRCNN).

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Overview

This project implements SRCNN (Super-Resolution Convolutional Neural Network) for enhancing the resolution of MRI brain scans. The model performs 4x upscaling, transforming 64×64 low-resolution images into 256×256 high-resolution images with improved quality metrics.

### Key Features

- 🧠 **4x Super-Resolution**: Upscales MRI images from 64×64 to 256×256
- 📊 **Quality Metrics**: Automatic PSNR and SSIM evaluation
- 🎯 **Interactive Web App**: Streamlit-based demo for easy testing
- 🔬 **Medical Imaging**: Optimized for brain MRI T1-weighted scans
- 📈 **Training Pipeline**: Complete notebook for model training

## 🏗️ Architecture

**SRCNN Model:**
```
Input (256×256×1) → Conv2D(64, 9×9) → ReLU 
                  → Conv2D(32, 5×5) → ReLU 
                  → Conv2D(1, 5×5) → Output (256×256×1)
```

**Parameters:** 57,281

## 📊 Results

- **Training Dataset**: IXI Brain MRI Dataset (T1-weighted)
- **Images Processed**: 30 volumes → ~1,400 image pairs
- **Training Configuration**:
  - Scale Factor: 4x
  - Batch Size: 32
  - Epochs: 10
  - Learning Rate: 1e-3
  - Loss Function: L1 Loss

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (optional, but recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Important**: If you encounter NumPy/OpenCV compatibility issues:
```bash
pip uninstall numpy opencv-python opencv-contrib-python -y
pip install "numpy<2.0.0" opencv-python==4.8.1.78
```

3. **Verify installation**
```bash
python -c "import numpy; import cv2; print(f'NumPy: {numpy.__version__}'); print(f'OpenCV: {cv2.__version__}')"
```

Expected output:
```
NumPy: 1.26.x
OpenCV: 4.8.1
```

## 📁 Project Structure
```
├── Data/
│   └── sr_pairs/              # Processed dataset
│       ├── HR/                # High-resolution images (256×256)
│       └── LR/                # Low-resolution images (64×64)
├── IXI-T1/                    # Raw NIfTI dataset (not included)
├── Final_Project.ipynb        # Training notebook
├── app.py                     # Streamlit web application
├── srcnn_best.pth            # Trained model weights
├── training_history.png      # Training curves visualization
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## 🎓 Training

### 1. Prepare Dataset

Download the IXI Brain MRI Dataset and place NIfTI files in `IXI-T1/` folder.

### 2. Run Training Notebook

Open `Final_Project.ipynb` in Jupyter:
```bash
jupyter notebook Final_Project.ipynb
```

**Training Pipeline:**
1. **Data Preprocessing**: Converts NIfTI volumes to PNG image pairs
2. **Dataset Creation**: Creates train/validation split (90/10)
3. **Model Training**: Trains SRCNN with L1 loss
4. **Evaluation**: Computes PSNR and SSIM metrics
5. **Visualization**: Generates comparison plots

**Training Time**: ~1 hour on CPU (30 volumes, 10 epochs)

### 3. Model Outputs

After training, you'll have:
- `srcnn_best.pth` - Best model checkpoint
- `training_history.png` - Training/validation curves
- `results_comparison.png` - Visual results
- `data/sr_pairs/` - Processed dataset

## 🖥️ Web Application

### Launch the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Features

**Dataset Mode:**
- Browse processed MRI images
- Compare Bicubic vs SRCNN upscaling
- View PSNR and SSIM metrics

**Upload Mode:**
- Upload your own MRI images
- Two processing options:
  - Downsample then super-resolve (demonstrates improvement)
  - Use uploaded image as low-resolution directly
- Real-time quality metrics

## 📈 Performance Metrics

### Evaluation Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
- Measures reconstruction quality
- Unit: decibels (dB)
- Higher is better (typically 20-50 dB)
- Good quality: >30 dB

**SSIM (Structural Similarity Index)**
- Measures perceptual similarity
- Range: -1 to 1 (1 = identical)
- Considers: luminance, contrast, structure
- Excellent quality: >0.9

### Typical Results

| Method | PSNR | SSIM |
|--------|------|------|
| Bicubic | ~25 dB | ~0.85 |
| SRCNN | ~28-30 dB | ~0.90+ |
| **Improvement** | **+3-5 dB** | **+0.05** |

## 🔧 Configuration

### Training Parameters

Edit in `Final_Project.ipynb`:
```python
SCALE = 4              # Upscaling factor
HR_SIZE = 256          # High-res image size
BATCH_SIZE = 32        # Training batch size
EPOCHS = 10            # Training epochs
LEARNING_RATE = 1e-3   # Initial learning rate
```

### Model Parameters

Edit `SRCNN` class in notebook or `app.py`:
```python
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),  # Patch extraction
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),  # Non-linear mapping
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)    # Reconstruction
        )
```

## 🐛 Troubleshooting

### NumPy/OpenCV Compatibility Error

**Error:** `ImportError: numpy.core.multiarray failed to import`

**Solution:**
```bash
pip uninstall numpy opencv-python opencv-contrib-python -y
pip install "numpy<2.0.0" opencv-python==4.8.1.78
```

Then **restart Jupyter** completely (close browser and terminal).

### Model Not Found

**Error:** `Model weights not found`

**Solution:** Train the model first using `Final_Project.ipynb`

### Dataset Not Found

**Error:** `Dataset not found at: data/sr_pairs`

**Solution:** Run preprocessing cells in `Final_Project.ipynb` to create dataset

### CUDA Out of Memory

**Solution:** Reduce `BATCH_SIZE` in configuration or use CPU:
```python
DEVICE = "cpu"
```

## 📚 Dataset

**IXI Brain MRI Dataset**
- Source: [IXI Dataset](https://brain-development.org/ixi-dataset/)
- Modality: T1-weighted MRI
- Format: NIfTI (.nii.gz)
- Preprocessing: Normalized, cropped, resized

## 🎯 Use Cases

- Medical image enhancement
- MRI quality improvement
- Research in super-resolution
- Deep learning for medical imaging
- Educational purposes

## 📖 References

**Original SRCNN Paper:**
```
Dong, C., Loy, C. C., He, K., & Tang, X. (2015).
Image super-resolution using deep convolutional networks.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(2), 295-307.
```

**Additional Resources:**
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [IXI Dataset](https://brain-development.org/ixi-dataset/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@Neha-Esh]([https://github.com/yourusername](https://github.com/Neha-Esh/Super-resolution-on-MRI.git))
- Email: your.email@example.com

## 🙏 Acknowledgments

- IXI Dataset contributors
- SRCNN paper authors
- PyTorch and Streamlit communities

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Contact the author

---

**Note**: This project is for educational and research purposes. Always consult medical professionals for clinical decisions.
