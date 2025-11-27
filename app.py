import io
from pathlib import Path

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/model.pth")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.warning("Modelo não encontrado. Treine o modelo executando 'python train.py' antes de rodar o app.")
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint.get("Classes", ["papel", "plastico", "vidro", "metal"])

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def predict_image(model, class_names, image: Image.Image):
    tfm = get_transform()
    tensor = tfm(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    return probs

def main():
    st.title("Classificador de Resíduos Recicláveis com Visão Computacional")
    st.write("Envie uma imagem de um resíduo para identificar o tipo de material.")

    model, class_names = load_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem enviada", use_column_width=True)

        if st.button("Classificar"):
            probs = predict_image(model, class_names, image)
            st.subheader("Resultado")
            best_idx = probs.argmax()
            st.write(f"Classe prevista: **{class_names[best_idx]}**")

            st.write("Probabilidades:")
            for cls, p in zip(class_names, probs):
                st.write(f"- {cls}: {p:.2f}")

            st.info("Os resultados dependem da qualidade do dataset utilizado no treinamento.")

if __name__ == "__main__":
    main()