import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import config
from dataset import MicroMacroDataset
from model import MicroToMacroGuidedTransformer

def train_pipeline():
    excel_path = os.path.join(config.BASE_DIR, "MMEW_Micro_Exp.xlsx")
    image_dir = os.path.join(config.BASE_DIR, "Micro_Expression")
    
    dataset = MicroMacroDataset(excel_path=excel_path, image_dir=image_dir)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    device = torch.device(config.DEVICE)
    model = MicroToMacroGuidedTransformer().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)
    
    best_macro_acc = 0.0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for frames, flows, labels in train_loader:
            frames, flows, labels = frames.to(device), flows.to(device), labels.to(device)
            
            optimizer.zero_grad()
            micro_logits, macro_logits = model(frames, flows)
            
            # Multi-Task Joint Loss: Combining Micro and Macro Losses
            loss_micro = criterion(micro_logits, labels)
            loss_macro = criterion(macro_logits, labels)
            total_loss = loss_macro + 0.5 * loss_micro  # Micro features guide macro recognition
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            
        # Validation Evaluation
        model.eval()
        macro_correct, total_samples = 0, 0
        with torch.no_grad():
            for frames, flows, labels in val_loader:
                frames, flows, labels = frames.to(device), flows.to(device), labels.to(device)
                _, macro_logits = model(frames, flows)
                preds = macro_logits.argmax(dim=-1)
                macro_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
        val_acc = macro_correct / total_samples if total_samples > 0 else 0
        print(f"Epoch [{epoch:02d}/{config.NUM_EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | Guided Macro Val Acc: {val_acc:.4f}")
        
        if val_acc > best_macro_acc:
            best_macro_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_guided_model.pt"))

if __name__ == "__main__":
    train_pipeline()