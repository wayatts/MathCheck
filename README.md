# MathCheck: Diagnostic Step Verification for Handwritten Equations

MathCheck is a hybrid neuro-symbolic artificial intelligence system designed for automated handwritten mathematical formula recognition and step-by-step diagnostic tutoring.

## System Overview
- **Perception (Neural OCR):** CRNN (CNN + Bidirectional GRU) trained with Connectionist Temporal Classification (CTC) loss.
- **Reasoning (Symbolic Engine):** Abstract Syntax Tree (AST) validation and diagnostic misconception analysis powered by SymPy.
- **Interface:** Interactive Streamlit dashboard for real-time verification and automated step correction.

## Project Structure
- `app.py`: Interactive Streamlit web interface.
- `model.py`: CRNN architecture (CNN feature extractor + Bi-GRU + CTC decoder).
- `symbolic_engine.py`: Algebraic verification, misconception analysis, and LaTeX generation.
- `evaluate.py`: Model evaluation script (ExpRate, CER, and Confusion Matrix).
- `checkpoints/`: Pretrained model weights (`best_model.pt`).
- `figures/`: Training loss convergence plots and confusion matrices.

## Quickstart

### Prerequisites
- **Python:** `3.12.x` (Tested on Python 3.12.3; PyTorch and Torchvision may fail on Python 3.13+)

### 1. Installation
Clone the repository, create a virtual environment, and install dependencies:
```bash
git clone [https://github.com/wayatts/MathCheck.git](https://github.com/wayatts/MathCheck.git)
cd MathCheck

# Create and activate a Python 3.12 virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```
### 2. Run Interactive Demo
```
streamlit run app.py
```