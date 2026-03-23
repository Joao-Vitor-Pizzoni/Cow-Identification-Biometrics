import torch
import numpy as np
import cv2
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# 1. Carregar o Modelo e os Dados Salvos
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()

# Carrega o dicionário que você criou com as 1018 fotos
# Certifique-se de que o arquivo .npy está na mesma pasta
banco_de_dados = np.load("embeddings_vacas.npy", allow_pickle=True).item()

# 2. Transformações (Mesmas do treino!)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def buscar_vaca(caminho_da_foto_nova):
    # Carregar e processar a foto nova
    img_cv2 = cv2.imread(caminho_da_foto_nova)
    if img_cv2 is None:
        return "Erro: Imagem não encontrada."

    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0)

    # Gerar o embedding da foto nova
    with torch.no_grad():
        embedding_novo = model(img_tensor).squeeze().cpu().numpy().reshape(1, -1)

    # Comparar com todo o banco de dados
    melhor_match = None
    maior_similaridade = -1

    for nome_arquivo, embedding_salvo in banco_de_dados.items():
        # Calcula a similaridade de cosseno
        emb_salvo_ready = embedding_salvo.reshape(1, -1)
        sim = cosine_similarity(embedding_novo, emb_salvo_ready)[0][0]

        if sim > maior_similaridade:
            maior_similaridade = sim
            melhor_match = nome_arquivo

    return melhor_match, maior_similaridade


# --- TESTE DO BUSCADOR ---
foto_teste = "dataset_final/valid/10.jpg"  # Troque pelo caminho de uma foto real
resultado, confianca = buscar_vaca(foto_teste)

print("-" * 30)
print(f"Foto buscada: {foto_teste}")
print(f"Vaca identificada: {resultado}")
print(f"Nível de similaridade: {confianca:.4f}")
print("-" * 30)