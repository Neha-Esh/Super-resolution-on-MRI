# MRI Super-Resolution using SRCNN

Deep learning-based 4x super-resolution for MRI brain scans using Super-Resolution Convolutional Neural Network (SRCNN).

## Features
- 4x upscaling of MRI images
- PSNR and SSIM evaluation metrics
- Interactive Streamlit web application
- Trained on IXI dataset

## Requirements
```
numpy<2.0.0
opencv-python==4.8.1.78
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
nibabel>=5.0.0
matplotlib>=3.5.0
tqdm>=4.65.0
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
Open and run `Final_Project.ipynb` in Jupyter Notebook to train the model.

### Web Application
```bash
streamlit run app.py
```

## Project Structure
```
├── Data/sr_pairs/           # Processed dataset
│   ├── HR/                  # High-resolution images
│   └── LR/                  # Low-resolution images
├── Final_Project.ipynb      # Training notebook
├── app.py                   # Streamlit web app
├── srcnn_best.pth          # Trained model weights
├── training_history.png    # Training curves
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Model Architecture
- Layer 1: Conv2D (1→64 filters, 9×9 kernel)
- Layer 2: Conv2D (64→32 filters, 5×5 kernel)
- Layer 3: Conv2D (32→1 filter, 5×5 kernel)

## Results
- Average PSNR: ~XX.XX dB
- Average SSIM: ~0.XXXX

## Dataset
IXI Brain Dataset - T1-weighted MRI scans

## License
MIT License

## Author
Your Name
```

### 2. **requirements.txt**
```
numpy<2.0.0
opencv-python==4.8.1.78
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
nibabel>=5.0.0
matplotlib>=3.5.0
tqdm>=4.65.0
```

### 3. **.gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Large files
*.nii
*.nii.gz

# Model weights (if too large for GitHub)
# srcnn_best.pth

# Data directories (optional - can be large)
# Data/
# IXI-T1/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
```

### 4. **LICENSE**
```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 5. **results_comparison.png** (Optional)
Add the visualization output from your notebook showing before/after comparisons.

### 6. **docs/** folder (Optional but professional)
Create a `docs` folder with:
- `SETUP.md` - Detailed setup instructions
- `TRAINING.md` - Training procedure and tips
- `TROUBLESHOOTING.md` - Common issues and solutions

## Final Repository Structure:
```
your-repo/
├── Data/
│   └── sr_pairs/
│       ├── HR/
│       └── LR/
├── docs/                    # ← ADD THIS (optional)
│   ├── SETUP.md
│   ├── TRAINING.md
│   └── TROUBLESHOOTING.md
├── Final_Project.ipynb
├── app.py
├── srcnn_best.pth
├── training_history.png
├── results_comparison.png   # ← ADD THIS (optional)
├── README.md                # ← ADD THIS (essential)
├── requirements.txt         # ← ADD THIS (essential)
├── .gitignore              # ← ADD THIS (essential)
└── LICENSE                 # ← ADD THIS (recommended)

