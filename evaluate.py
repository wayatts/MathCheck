import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import MathFormulaCRNN

VOCAB = [
    '<blank>', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', '*', '/', '=', '^', '(', ')', '[', ']',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}

class MathTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.samples = []
        self.transform = transform
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                label_str = str(row["label"]).strip()
                if os.path.exists(img_path) and len(label_str) > 0:
                    self.samples.append((img_path, label_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label_str

def decode_prediction(output_tensor):
    """Greedy CTC decoder."""
    preds = output_tensor.argmax(dim=2).squeeze(1).tolist()
    decoded = []
    prev = None
    for p in preds:
        if p != 0 and p != prev:
            decoded.append(idx_to_char.get(p, ''))
        prev = p
    return "".join(decoded)

def levenshtein_distance(s1, s2):
    """Computes minimum character edits between reference and prediction."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def evaluate():
    os.makedirs("figures", exist_ok=True)
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MathTestDataset("data/processed/metadata.csv", transform=transform)
    
    test_size = int(0.20 * len(dataset))
    test_samples = dataset.samples[-test_size:]
    print(f"Running rigorous evaluation across {len(test_samples)} test samples...")

    model = MathFormulaCRNN(num_classes=len(VOCAB))
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()

    exact_matches = 0
    total_edit_distance = 0
    total_ref_chars = 0

    cm_tokens = ['0', '1', '2', '3', '4', 'x', 'y', '+', '-', '=', '(', ')']
    cm_map = {c: i for i, c in enumerate(cm_tokens)}
    confusion_matrix = np.zeros((len(cm_tokens), len(cm_tokens)), dtype=int)

    with torch.no_grad():
        for img_path, ref_label in test_samples:
            img = Image.open(img_path).convert("L")
            tensor = transform(img).unsqueeze(0)
            
            out = model(tensor).permute(1, 0, 2)
            pred_label = decode_prediction(out)

            if pred_label == ref_label:
                exact_matches += 1

            edits = levenshtein_distance(ref_label, pred_label)
            total_edit_distance += edits
            total_ref_chars += len(ref_label)

            min_len = min(len(ref_label), len(pred_label))
            for r_char, p_char in zip(ref_label[:min_len], pred_label[:min_len]):
                if r_char in cm_map and p_char in cm_map:
                    confusion_matrix[cm_map[r_char], cm_map[p_char]] += 1

    expr_acc = (exact_matches / len(test_samples)) * 100.0
    cer = (total_edit_distance / max(1, total_ref_chars)) * 100.0
    char_acc = max(0.0, 100.0 - cer)

    print("\n--- Model Evaluation Results ---")
    print(f"Total Test Set Samples:          {len(test_samples)}")
    print(f"Exact Expression Rate (ExpRate): {expr_acc:.2f}%")
    print(f"Character Error Rate (CER):      {cer:.2f}%")
    print(f"Estimated Character Accuracy:    {char_acc:.2f}%")
    print("--------------------------------\n")

    plt.figure(figsize=(8, 6.5))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Character-Level Confusion Matrix (Key Mathematical Tokens)")
    plt.colorbar()
    tick_marks = np.arange(len(cm_tokens))
    plt.xticks(tick_marks, cm_tokens, fontsize=10)
    plt.yticks(tick_marks, cm_tokens, fontsize=10)
    plt.xlabel("Predicted Character", fontsize=11)
    plt.ylabel("Ground Truth Character", fontsize=11)

    thresh = confusion_matrix.max() / 2.0
    for i in range(len(cm_tokens)):
        for j in range(len(cm_tokens)):
            val = confusion_matrix[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         ha="center", va="center",
                         color="white" if val > thresh else "black", fontsize=9)

    plt.tight_layout()
    cm_path = "figures/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot successfully exported to {cm_path}")

if __name__ == "__main__":
    evaluate()