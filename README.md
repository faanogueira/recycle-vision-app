# ♻️ Classificador de Resíduos Recicláveis com Visão Computacional

Este projeto desenvolve um modelo de visão computacional capaz de identificar o tipo de resíduo (papel, plástico, vidro e metal) a partir de uma imagem. A solução foi criada para auxiliar cooperativas de reciclagem a aumentar produtividade e reduzir erros na triagem manual.

---

## 🎯 Objetivo
Criar um protótipo funcional capaz de:
1. Receber uma imagem enviada pelo usuário.
2. Classificar o tipo de material usando um modelo CNN com transfer learning.
3. Exibir probabilidades por classe.
4. Rodar em CPU com baixo custo computacional.

---

## 💡 Motivação
A triagem manual é lenta e sujeita a erro. Um classificador automático pode ajudar cooperativas a separar materiais de forma mais eficiente, reduzindo desperdício e aumentando valor de venda.

---

## 📁 Dataset
Utilizamos a combinação de três datasets:
1. TrashNet
2. Garbage Classification Dataset
3. Waste Vision Dataset

As classes finais:
- `papel`
- `plastico`
- `vidro`
- `metal`

### 📦 A estrutura dos dados:

| Classe      | Treino | Validação | Total |
|-------------|--------|-----------|-------|
| **Papel**   | 210    | 50        | 260   |
| **Plástico**| 133    | 30        | 163   |
| **Vidro**   | 403    | 100       | 503   |
| **Metal**   | 154    | 40        | 194   |

---

## 🧠 Arquitetura da Solução
1. Organização do dataset em subpastas por classe.
2. Aumentação com rotação leve, variação de brilho e recortes aleatórios.
3. Transfer learning com ResNet18 pré-treinada no ImageNet.
4. Treinamos apenas o classificador final para acelerar o processo.
5. Métrica principal: **F1 macro.**

---

## 🧪 Resultados

### 🔍 Melhor modelo (época 1 de 5)

- **Loss de validação:** 0.4128  
- **F1 macro:** 0.8419  
- **Acurácia:** 86%  
- **Tempo de inferência:** ~0.3s em CPU  
- **Modelo salvo em:** `models/model.pth`

### 📋 Relatório de Classificação

| Classe     | Precisão | Recall | F1-score | Suporte |
|------------|----------|--------|----------|---------|
| **Metal**  | 0.83     | 0.72   | 0.77     | 154     |
| **Papel**  | 0.92     | 0.93   | 0.92     | 210     |
| **Plástico**| 0.94     | 0.58   | 0.72     | 133     |
| **Vidro**  | 0.82     | 0.96   | 0.88     | 403     |

---

## ⚙️ Tecnologias Utilizadas
- Python 3.10+
- PyTorch
- Torchvision
- Scikit-learn
- Streamlit
- Pillow
- Numpy

---

## 🚀 Como Rodar

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

Certifique-se de que o dataset está organizado em `data/train` e `data/valid` conforme descrito acima.

Execute o treinamento:

```bash
python train.py
```

Isso irá salvar o modelo treinado em `models/model.pth`.

Em seguida, rode o app:

```bash
streamlit run app.py
```

---

## 📌 Próximos Passos
1. Integrar GradCAM para interpretabilidade.
2. Expandir classes para incluir orgânico e papelão.
3. Testar modelo em câmera de celular para triagem em tempo real.

---

<br>
<!-- Início da seção "Contato" -->
<h2>🌐 Contate-me: </h2>
<div>
  <p>Developed by <b>Fábio Nogueira</b></p>
</div>
<p>
<a href="https://www.linkedin.com/in/faanogueira/" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=13930&format=png&color=000000" target="_blank" width="80"></a>
<a href="https://github.com/faanogueira" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=AZOZNnY73haj&format=png&color=000000" target="_blank" width="80"></a>
<a href="https://api.whatsapp.com/send?phone=5571983937557" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=16713&format=png&color=000000" target="_blank" width="80"></a>
<a href="mailto:faanogueira@gmail.com"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=P7UIlhbpWzZm&format=png&color=000000" target="_blank" width="80"></a> 
</p>
<!-- Fim da seção "Contato" -->
<br>

