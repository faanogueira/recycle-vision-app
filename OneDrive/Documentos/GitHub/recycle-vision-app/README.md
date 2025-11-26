# Classificador de Resíduos Recicláveis com Visão Computacional

Este projeto desenvolve um modelo de visão computacional capaz de identificar o tipo de resíduo (papel, plástico, vidro e metal) a partir de uma imagem. A solução foi criada para auxiliar cooperativas de reciclagem a aumentar produtividade e reduzir erros na triagem manual.

## Objetivo
Criar um protótipo funcional capaz de:
1. Receber uma imagem enviada pelo usuário.
2. Classificar o tipo de material usando um modelo CNN com transfer learning.
3. Exibir probabilidades por classe.
4. Rodar em CPU com baixo custo computacional.

## Motivação
A triagem manual é lenta e sujeita a erro. Um classificador automático pode ajudar cooperativas a separar materiais de forma mais eficiente, reduzindo desperdício e aumentando valor de venda.

## Dataset
Utilizamos a combinação de três datasets:
1. TrashNet
2. Garbage Classification Dataset
3. Waste Vision Dataset

As classes finais:
- papel
- plastico
- vidro
- metal

A estrutura de pastas esperada é:

data/
  train/
    papel/
    plastico/
    vidro/
    metal/
  valid/
    papel/
    plastico/
    vidro/
    metal/

## Arquitetura da Solução
1. Organização do dataset em subpastas por classe.
2. Aumentação com rotação leve, variação de brilho e recortes aleatórios.
3. Transfer learning com ResNet18 pré-treinada no ImageNet.
4. Treinamos apenas o classificador final para acelerar o processo.
5. Métrica principal: F1 macro.

## Tecnologias
- Python 3.10
- PyTorch
- Torchvision
- Scikit-learn
- Streamlit
- Pillow
- Numpy

## Como Rodar

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

## Resultados (exemplo esperado)
- Acurácia: ~85%
- F1 macro: ~0.84
- Tempo de inferência: ~0.3s em CPU

(Os valores reais vão depender da quantidade e qualidade das imagens utilizadas.)

## Próximos Passos
1. Integrar GradCAM para interpretabilidade.
2. Expandir classes para incluir orgânico e papelão.
3. Testar modelo em câmera de celular para triagem em tempo real.

## Autor
Projeto de referência criado para hackathon de Data Science por Fábio Nogueira.