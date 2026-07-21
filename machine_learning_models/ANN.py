from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = [] # Valores extraidos em pre-processing 
Y = [] # Valores extraidos em pre-processing

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)

# Normalizar os dados para melhor desempenho da rede
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em conjunto de treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.3, random_state=42, stratify=Y_encoded)

# Criar o modelo
model = tf.keras.Sequential()

# Camada de entrada (6 atributos numéricos)
model.add(tf.keras.layers.Input(shape=(4,)))

# Camadas ocultas
model.add(tf.keras.layers.Dense(64, activation='relu'))

# 🔥 Nova camada oculta de 32 neurônios
model.add(tf.keras.layers.Dense(32, activation='relu'))


# Camada de saída para 3 classes (Softmax para classificação)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Resumo do modelo
model.summary()

# Configuração do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))

# Avaliar no conjunto de teste
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {accuracy:.2f}")
