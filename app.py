import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import MathFormulaCRNN
from symbolic_engine import verify_step

st.set_page_config(page_title="MathCheck", layout="centered")

st.title("MathCheck: Diagnostic Step Verification for Handwritten Equations")
st.markdown("MCS 306 Final Project: Neuro-Symbolic Algebraic Assistant")

VOCAB = [
    '<blank>', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', '*', '/', '=', '^', '(', ')', '[', ']',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
idx_to_char = {i: c for i, c in enumerate(VOCAB)}

@st.cache_resource
def load_model():
    model = MathFormulaCRNN(num_classes=len(VOCAB))
    try:
        model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.warning("checkpoints/best_model.pt not found. Running with initialized weights.")
    model.eval()
    return model

model = load_model()

transform_pipeline = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(pil_img):
    """
    Direct preprocessing matching the dataset distribution:
    - Grayscale conversion
    - Resize to (128, 32)
    - Normalized to [-1, 1] tensor
    """
    clean_img = pil_img.convert("L").resize((128, 32), Image.Resampling.BILINEAR)
    tensor = transform_pipeline(clean_img)
    return clean_img, tensor.unsqueeze(0)

def decode_prediction(output_tensor):
    """Greedy CTC decoder to map model log-probabilities to formula tokens."""
    preds = output_tensor.argmax(dim=2).squeeze(1).tolist()
    decoded = []
    prev = None
    for p in preds:
        if p != 0 and p != prev:
            decoded.append(idx_to_char.get(p, ''))
        prev = p
    return "".join(decoded)

st.header("Step-by-Step Diagnostic Verification")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1 (Problem Setup)")
    img_step1 = st.file_uploader("Upload Step 1", type=["png", "jpg", "jpeg"], key="step1")

with col2:
    st.subheader("Step 2 (Student Derivation)")
    img_step2 = st.file_uploader("Upload Step 2", type=["png", "jpg", "jpeg"], key="step2")

if "text1" not in st.session_state:
    st.session_state.text1 = ""
if "text2" not in st.session_state:
    st.session_state.text2 = ""

if img_step1 is not None:
    raw_img1 = Image.open(img_step1)
    clean_img1, tensor1 = preprocess_image(raw_img1)
    with torch.no_grad():
        out1 = model(tensor1).permute(1, 0, 2)
        st.session_state.text1 = decode_prediction(out1)
    with col1:
        st.image(clean_img1, caption="Processed Step 1 Input (32x128)", use_container_width=True)

if img_step2 is not None:
    raw_img2 = Image.open(img_step2)
    clean_img2, tensor2 = preprocess_image(raw_img2)
    with torch.no_grad():
        out2 = model(tensor2).permute(1, 0, 2)
        st.session_state.text2 = decode_prediction(out2)
    with col2:
        st.image(clean_img2, caption="Processed Step 2 Input (32x128)", use_container_width=True)

st.markdown("---")
st.subheader("Verification & Diagnostic Review")

final_text1 = st.text_input("Formula Step 1 (Detected / Confirmed)", value=st.session_state.text1)
final_text2 = st.text_input("Formula Step 2 (Detected / Confirmed)", value=st.session_state.text2)

if st.button("Run Diagnostic Tutor"):
    if not final_text1.strip() or not final_text2.strip():
        st.warning("Please provide or confirm both formulas to run verification.")
    else:
        valid, message, latex_prev, latex_curr, latex_correction = verify_step(final_text1, final_text2)
        
        st.write("### Evaluated Derivation")
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            st.caption("Step 1 (Given):")
            if latex_prev:
                st.latex(latex_prev)
            else:
                st.code(final_text1)
        with col_eval2:
            st.caption("Step 2 (Student):")
            if latex_curr:
                st.latex(latex_curr)
            else:
                st.code(final_text2)

        st.markdown("---")
        if valid:
            st.success(f"**Verification Result:** {message}")
        else:
            st.error(f"**Verification Result:** {message}")
            if latex_correction:
                st.info("### Suggested Correction")
                st.markdown("The mathematically valid transition from Step 1 should be:")
                st.latex(latex_correction)