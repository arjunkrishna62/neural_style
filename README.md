# Neural Style Transfer with Canvas Preservation 🎨🔒

This project implements a **Neural Style Transfer (NST)** pipeline with a focus on preserving canvas structure, image integrity, and addressing copyright-related transformations.

---

## 🔍 Overview

- Built using **PyTorch**, this NST pipeline applies visual styles to content images while:
  - **Preserving canvas borders and framing**, avoiding overflow or cropping.
  - **Securing styled outputs** against automated copying, e.g., through watermarking or visual diffs.

- Implemented core NST components:
  - Gatys-style feature extraction via pretrained VGG network.
  - **Canvas masks** to restrict stylization to interior regions.
  - Optional **image-protection filters** to embed subtle visual signatures.

What sets this work apart is its **dual focus**:
1. **Artistic quality** using standard content/style losses.
2. **Content protection**, important for digital artists, copyright compliance, and content licensing.

---

## 🛠️ Project Structure

neural_style/
├── data/
│ ├── content/ # Content images
│ └── style/ # Style images
├── models/
│ └── vgg_features.py # VGG feature extractor
├── canvas_protect.py # Canvas mask & image-protection modules
├── train.py # Full NST training flow
├── stylize.py # Generate stylized image(s)
├── utils.py # Helpers (e.g., image loading, saving)
└── README.md


---

## 🚀 Quick Start

1. **Clone repo & install dependencies**
   ```bash
   git clone https://github.com/arjunkrishna62/neural_style.git
   cd neural_style
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
2. **Stylize an image with canvas preservation**
python stylize.py \
  --content data/content/example.png \
  --style data/style/paint.png \
  --canvas-mask 50 50 450 450 \
  --output output/styled.png

3.**Train or fine-tune style weights**
python train.py \
  --content-dir data/content \
  --style-dir data/style \
  --epochs 500 \
  --canvas-mask 50 50 450 450
  
Feel free to update parameter names or hyperparameters based on your code.
