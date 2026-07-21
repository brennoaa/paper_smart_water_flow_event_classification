import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

X = [] # Valores extraidos em pre-processing 
Y = [] # Valores extraidos em pre-processing

#Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converter rótulos para valores numéricos
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Selecionar duas features para visualização
X_plot = X_scaled[:, :2]

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.3, random_state=13)

# 1. Grid Search para encontrar o melhor K
# ============================================
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 30)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Melhor K encontrado:", grid.best_params_)
print("Melhor score (CV):", grid.best_score_)

# Substituir n_neighbors pelo melhor encontrado
best_k = grid.best_params_['n_neighbors']
# ============================================

# Treinar o modelo
#n_neighbors = 4
clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Calcular acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)
print("\nRelatório de classificação:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Validação cruzada
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
print("\nScores em cada fold:", scores)
print("Média dos scores:", scores.mean())
print("Desvio padrão:", scores.std())