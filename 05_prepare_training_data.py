import os
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

# 1. Configuração do Modelo
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()

# 2. Transformações
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Caminhos (USE A PASTA QUE VOCÊ ORGANIZOU)
pasta_organizada = "dataset_organizado/"
extensoes_validas = (".jpg", ".jpeg", ".png")

X = []  # Lista para os Embeddings
y = []  # Lista para os IDs das Vacas
nomes_arquivos = []

# 4. Loop de Processamento
# Listamos os arquivos da pasta organizada
arquivos = [f for f in os.listdir(pasta_organizada) if f.lower().endswith(extensoes_validas)]

print(f"Processando {len(arquivos)} imagens da pasta organizada...")

with torch.no_grad():
    for nome_arquivo in tqdm(arquivos):
        caminho_completo = os.path.join(pasta_organizada, nome_arquivo)

        # --- EXTRAÇÃO DO Y (LABEL) ---
        # Se o nome for "1_02.jpg", o ID da vaca é "1"
        id_vaca = nome_arquivo.split('_')[0]

        # Carregar imagem
        img_cv2 = cv2.imread(caminho_completo)
        if img_cv2 is None: continue

        # Preparar para o modelo
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = preprocess(img_pil).unsqueeze(0)

        # Gerar embedding (X)
        embedding = model(img_tensor).squeeze().cpu().numpy()

        # Salvar nas listas
        X.append(embedding)
        y.append(id_vaca)
        nomes_arquivos.append(nome_arquivo)

# 5. Converter para arrays do NumPy (formato padrão para ML)
X = np.array(X)
y = np.array(y)

# 6. Salvar tudo em um único arquivo para o Treinamento
np.savez("dados_treinamento_vaca.npz", X=X, y=y, nomes=nomes_arquivos)

print(f"\nSucesso!")
print(f"Matriz X: {X.shape} (Fotos x Características)")
print(f"Vetor y: {len(y)} labels gerados.")
print("Arquivo 'dados_treinamento_vaca.npz' salvo.")