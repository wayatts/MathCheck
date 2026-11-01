import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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

class MathDataset(Dataset):
    def __init__(self, csv_file, transform=None, max_target_len=28):
        self.samples = []
        self.transform = transform
        
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                label_str = str(row["label"]).strip()
                
                encoded = [char_to_idx[c] for c in label_str if c in char_to_idx]
                
                if 0 < len(encoded) <= max_target_len and os.path.exists(img_path):
                    self.samples.append((img_path, encoded))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, encoded = self.samples[idx]
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            
        target = torch.tensor(encoded, dtype=torch.long)
        return image, target

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets, dim=0)
    
    return images, targets_concat, target_lengths

def train_model():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MathDataset(csv_file="data/processed/metadata.csv", transform=transform)
    print(f"Total valid samples loaded: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No valid samples found. Please check data/processed/metadata.csv.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    model = MathFormulaCRNN(num_classes=len(VOCAB)).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 15
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print("Beginning training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        for images, targets_concat, target_lengths in train_loader:
            images = images.to(device)
            targets_concat = targets_concat.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images).permute(1, 0, 2)
            time_steps, current_batch_size, _ = outputs.size()
            
            input_lengths = torch.full(size=(current_batch_size,), fill_value=time_steps, dtype=torch.long, device=device)
            
            loss = criterion(outputs, targets_concat, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1

        epoch_train_loss = running_train_loss / max(1, train_batches)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets_concat, target_lengths in val_loader:
                images = images.to(device)
                targets_concat = targets_concat.to(device)
                target_lengths = target_lengths.to(device)
                
                outputs = model(images).permute(1, 0, 2)
                time_steps, current_batch_size, _ = outputs.size()
                input_lengths = torch.full(size=(current_batch_size,), fill_value=time_steps, dtype=torch.long, device=device)
                
                v_loss = criterion(outputs, targets_concat, input_lengths, target_lengths)
                if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                    running_val_loss += v_loss.item()
                    val_batches += 1

        epoch_val_loss = running_val_loss / max(1, val_batches)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss and epoch_val_loss > 0:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print("  --> Saved new best checkpoint.")

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, epochs + 1), train_losses, label="Training CTC Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation CTC Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence Behavior of CRNN Model")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/loss_curve.png", dpi=300)
    plt.close()
    print("Training finished. Convergence plot saved to figures/loss_curve.png")

if __name__ == "__main__":
    train_model()