import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados extraídos
dados = np.load("dados_treinamento_vaca.npz")
X = dados['X']
y = dados['y']

# 2. Divisão de Treino e Teste (80% treina, 20% testa)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Iniciando treinamento com {len(X_train)} amostras...")

# 3. Treinar Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# 4. Treinar SVM
svm_model = svm.SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# 5. Visualização dos Resultados
modelos = ['Random Forest', 'SVM (Linear)']
acuracias = [acc_rf * 100, acc_svm * 100]

plt.figure(figsize=(8, 5))
sns.barplot(x=modelos, y=acuracias, hue=modelos, palette='viridis', legend=False)
plt.ylabel('Acurácia (%)')
plt.title('Comparação de Modelos: Identificação Bovina')
plt.ylim(0, 100)

for i, v in enumerate(acuracias):
    plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')

plt.savefig('resultado_acuracia.png')
print("Gráfico salvo como 'resultado_acuracia.png'")

print(f"Resultado Final:")
print(f"Acurácia RF: {acc_rf:.2%}")
print(f"Acurácia SVM: {acc_svm:.2%}")