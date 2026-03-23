import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# 1. Configurações
pasta_origem = "dataset_final/train/"
pasta_destino = "dataset_organizado/"
arquivo_embeddings = "embeddings_vacas.npy"

# Se você acha que tem, por exemplo, 100 vacas diferentes no total:
n_clusters = 200

# 2. Carregar dados
banco = np.load(arquivo_embeddings, allow_pickle=True).item()
nomes_arquivos = list(banco.keys())
X = np.array(list(banco.values()))

print(f"Agrupando {len(nomes_arquivos)} fotos em {n_clusters} possíveis vacas...")

# 3. Agrupamento (K-Means)
# O algoritmo vai dar um número (ID) para cada foto
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# 4. Criar pasta de destino
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# 5. Organizar e Renomear
# Dicionário para contar quantas fotos já salvamos de cada vaca
contador_vaca = {}

print("Copiando e renomeando arquivos...")
for i, nome_origem in enumerate(tqdm(nomes_arquivos)):
    id_vaca = labels[i] + 1  # Somamos 1 para começar da Vaca 1 em vez da 0

    if id_vaca not in contador_vaca:
        contador_vaca[id_vaca] = 1
    else:
        contador_vaca[id_vaca] += 1

    # Novo nome: IDdaVaca_IDdaFoto.jpg (ex: 1_01.jpg, 1_02.jpg)
    novo_nome = f"{id_vaca}_{contador_vaca[id_vaca]:02d}.jpg"

    caminho_origem = os.path.join(pasta_origem, nome_origem)
    caminho_destino = os.path.join(pasta_destino, novo_nome)

    # Copia o arquivo para a nova pasta com o nome novo
    shutil.copy(caminho_origem, caminho_destino)

print(f"\nConcluído! Verifique a pasta '{pasta_destino}'")