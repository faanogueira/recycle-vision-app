import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Configurações globais
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 5          # pode ajustar para mais se tiver tempo
LEARNING_RATE = 1e-3     # mais estável para fine-tuning
NUM_CLASSES = 4          # papel, plastico, vidro, metal

cm = confusion_matrix(best_y_true, best_y_pred)
print("Matriz de confusão:")
print(cm)

def get_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tfms, valid_tfms


def get_dataloaders():
    train_tfms, valid_tfms = get_transforms()

    train_path = DATA_DIR / "train"
    valid_path = DATA_DIR / "valid"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            "Pastas data/train e data/valid não encontradas. "
            "Organize o dataset (ex: usando split_dataset.py) antes de treinar."
        )

    train_ds = datasets.ImageFolder(train_path, transform=train_tfms)
    valid_ds = datasets.ImageFolder(valid_path, transform=valid_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, train_ds.classes


def build_model(num_classes: int):
    # carrega ResNet18 pré treinada
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # congela pesos do backbone
    for param in model.parameters():
        param.requires_grad = False

    # substitui a camada final
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(xb)
            loss = criterion(outputs, yb)

            running_loss += loss.item() * xb.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, macro_f1, all_labels, all_preds


def main():
    print(f"Usando dispositivo: {DEVICE}")

    train_loader, valid_loader, class_names = get_dataloaders()
    print(f"Classes encontradas: {class_names}")

    model = build_model(len(class_names))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0
    best_state = None
    best_y_true = None
    best_y_pred = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, y_true, y_pred = evaluate(model, valid_loader, criterion)

        print(f"Época {epoch}/{NUM_EPOCHS}")
        print(f"  Treino - Loss: {train_loss:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1 macro: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }
            best_y_true = y_true
            best_y_pred = y_pred

    if best_state is not None:
        model_path = MODEL_DIR / "model.pth"
        torch.save(best_state, model_path)
        print(f"\nMelhor modelo salvo em: {model_path}")
        print(f"Melhor F1 macro: {best_f1:.4f}")

        print("\nRelatório de classificação (melhor época):")
        print(classification_report(best_y_true, best_y_pred, target_names=class_names))
    else:
        print("Nenhum modelo foi salvo. Verifique se o treinamento ocorreu corretamente.")


if __name__ == "__main__":
    main()
