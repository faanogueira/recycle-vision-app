import os
import shutil
import random
from pathlib import Path

# Caminho onde você colocou todas as pastas de classes antes da divisão
SOURCE_DIR = Path("dataset_raw")  # ex: dataset_raw/papel, dataset_raw/plastico...
TARGET_DIR = Path("data")
TRAIN_RATIO = 0.8  # 80 por cento treino, 20 por cento validação

def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def split_class(class_name: str):
    src_path = SOURCE_DIR / class_name

    images = [f for f in src_path.iterdir() if f.is_file()]
    random.shuffle(images)

    n_train = int(len(images) * TRAIN_RATIO)

    train_images = images[:n_train]
    valid_images = images[n_train:]

    # pastas de destino
    train_dst = TARGET_DIR / "train" / class_name
    valid_dst = TARGET_DIR / "valid" / class_name

    create_dir(train_dst)
    create_dir(valid_dst)

    # mover arquivos
    for img in train_images:
        shutil.copy(img, train_dst / img.name)

    for img in valid_images:
        shutil.copy(img, valid_dst / img.name)

    print(f"Classe {class_name}: {len(train_images)} treino, {len(valid_images)} validação")

def main():
    if not SOURCE_DIR.exists():
        raise ValueError(f"Pasta de origem {SOURCE_DIR} não encontrada.")

    class_dirs = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]

    print("Classes detectadas:", class_dirs)

    for class_name in class_dirs:
        split_class(class_name)

    print("Divisão concluída com sucesso.")

if __name__ == "__main__":
    main()
