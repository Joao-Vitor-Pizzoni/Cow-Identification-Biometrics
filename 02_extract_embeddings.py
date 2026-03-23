import os
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import tqdm  # Uma barra de progresso visual (instale com: pip install tqdm)

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

# 3. Caminhos
pasta_fotos = "dataset_final/train/"
extensoes_validas = (".jpg", ".jpeg", ".png")

# Dicionário para guardar: { "nome_da_foto": embedding }
banco_de_focinhos = {}

# 4. O Loop de Processamento
fotos = [f for f in os.listdir(pasta_fotos) if f.lower().endswith(extensoes_validas)]

print(f"Iniciando o processamento de {len(fotos)} fotos...")

with torch.no_grad():
    for nome_arquivo in tqdm(fotos):
        caminho_completo = os.path.join(pasta_fotos, nome_arquivo)

        # Carregar imagem
        img_cv2 = cv2.imread(caminho_completo)
        if img_cv2 is None: continue

        # Preparar para o modelo
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = preprocess(img_pil).unsqueeze(0)

        # Gerar embedding
        embedding = model(img_tensor)

        # Salvar no dicionário (removendo a dimensão extra e convertendo para cpu/numpy se quiser)
        banco_de_focinhos[nome_arquivo] = embedding.squeeze().cpu().numpy()

# 5. Salvar o resultado final para não ter que processar tudo de novo amanhã
import numpy as np

np.save("embeddings_vacas.npy", banco_de_focinhos)

print("\nFeito! Embeddings salvos em 'embeddings_vacas.npy'")