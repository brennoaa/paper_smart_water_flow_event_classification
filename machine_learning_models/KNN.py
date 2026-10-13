from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

X = [] # Values ​​extracted in pre-processing 
Y = [] # Values ​​extracted in pre-processing

# Seleciona colunas de X
X = df[colunas_X].to_numpy()

# Seleciona a coluna Y
Y = df['class'].to_numpy()

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_plot = X_scaled[:, :2]

# mexi aqui
X_train, X_test, y_train, y_test = train_test_split(
    X_plot,
    Y,
    test_size=0.3,
    random_state=13,
    stratify=Y      # mantém a proporção das classes
)

# ===========================================================
# Grid Search
# ===========================================================

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1,10)}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

print("Melhor K encontrado:", grid.best_params_)
print("Melhor score (CV):", grid.best_score_)

best_k = grid.best_params_['n_neighbors']

#===========================================================
clf = KNeighborsClassifier(n_neighbors=best_k)
#===========================================================

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nRelatório de Classificação:\n")
print("Acurácia:", accuracy)

print(classification_report(y_test, y_pred))
#===========================================================

scores = cross_val_score(
    clf,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("\nScores em cada fold:")
print(scores)

print("\nMédia dos scores:", scores.mean())
print("Desvio padrão:", scores.std())
