import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Funciones de Activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Clase de Red Neuronal MLP
class SimpleMLP:
    def __init__(self, layers, activation_function='tanh', learning_rate=0.01):
        self.layers = layers
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward(self, X):
        self.A = [X]
        self.Z = []

        for i in range(len(self.layers) - 1):
            self.Z.append(np.dot(self.A[-1], self.weights[i]) + self.biases[i])

            if i == len(self.layers) - 2:
                self.A.append(softmax(self.Z[-1]))
            else:
                if self.activation_function == 'sigmoid':
                    self.A.append(sigmoid(self.Z[-1]))
                elif self.activation_function == 'tanh':
                    self.A.append(tanh(self.Z[-1]))
                elif self.activation_function == 'relu':
                    self.A.append(relu(self.Z[-1]))

        return self.A[-1]

    def compute_loss(self, y):
        n = y.shape[0]
        correct_logprobs = -np.log(self.A[-1][range(n), y])
        loss = np.sum(correct_logprobs) / n
        return loss

    def backward(self, X, y):
        n = y.shape[0]
        grads = {}
        self.dZ = self.A[-1]
        self.dZ[range(n), y] -= 1
        self.dZ /= n

        for i in reversed(range(len(self.layers) - 1)):
            grads['dW' + str(i + 1)] = np.dot(self.A[i].T, self.dZ)
            grads['db' + str(i + 1)] = np.sum(self.dZ, axis=0, keepdims=True)
            self.dZ = np.dot(self.dZ, self.weights[i].T)

            if self.activation_function == 'sigmoid':
                self.dZ *= self.A[i] * (1 - self.A[i])
            elif self.activation_function == 'tanh':
                self.dZ *= 1 - np.power(self.A[i], 2)
            elif self.activation_function == 'relu':
                self.dZ[self.A[i] <= 0] = 0

        for i in range(len(self.layers) - 1):
            self.weights[i] -= self.learning_rate * grads['dW' + str(i + 1)]
            self.biases[i] -= self.learning_rate * grads['db' + str(i + 1)]

    def train(self, X, y, epochs=100):
        loss_history = []
        for epoch in range(epochs):
            self.forward(X)
            loss = self.compute_loss(y)
            self.backward(X, y)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss}")
            loss_history.append(loss)

        return loss_history

# Cargar el dataset
# Asegúrate de tener el archivo CSV en la misma carpeta que este script o proporciona la ruta completa
df = pd.read_csv("data.csv")
df.dropna(inplace=True)

# Separar características y etiquetas
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
y = np.where(y == 'M', 1, 0)

# Normalizar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, recall_score
import os
import numpy as np
import matplotlib.pyplot as plt

def predict(model, X):
    probabilities = model.forward(X)
    return np.argmax(probabilities, axis=1)

# Definición de las configuraciones
learning_rates = [0.2, 0.1, 0.08, 0.5]
configurations = []
for lr in learning_rates:
    configurations.extend([
        {'layers': [30, 10, 2], 'activation_function': 'sigmoid', 'learning_rate': lr},
        {'layers': [30, 10, 2], 'activation_function': 'relu', 'learning_rate': lr},
        {'layers': [30, 10, 2], 'activation_function': 'tanh', 'learning_rate': lr},
        {'layers': [30, 20, 10, 2], 'activation_function': 'sigmoid', 'learning_rate': lr},
        {'layers': [30, 20, 10, 2], 'activation_function': 'relu', 'learning_rate': lr},
        {'layers': [30, 20, 10, 2], 'activation_function': 'tanh', 'learning_rate': lr},
        {'layers': [30, 20, 10, 5, 2], 'activation_function': 'sigmoid', 'learning_rate': lr},
        {'layers': [30, 20, 10, 5, 2], 'activation_function': 'relu', 'learning_rate': lr},
        {'layers': [30, 20, 10, 5, 2], 'activation_function': 'tanh', 'learning_rate': lr},
    ])

models = []

# Entrenar y almacenar cada modelo (Asegúrate de definir SimpleMLP y las funciones de entrenamiento/predicción)
for config in configurations:
    nn = SimpleMLP(config['layers'], activation_function=config['activation_function'], learning_rate=config['learning_rate'])
    loss_history = nn.train(X_train, y_train, epochs=10000)
    models.append((config, nn, loss_history))

# Realizar predicciones, graficar resultados y guardar gráficas y métricas para cada modelo
for config, nn, loss_history in models:
    y_pred = predict(nn, X_test)
    
    # Cálculo de métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Crear la carpeta si no existe
    folder_name = f"results/{config['activation_function']}_{config['layers']}_LR{config['learning_rate']}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Guardar las métricas en un archivo de texto
    with open(f"{folder_name}/metrics.txt", 'w') as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")

    # Graficar y guardar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {config["activation_function"]} - {config["layers"]} - LR: {config["learning_rate"]}\nAccuracy: {accuracy:.2f}, F1: {f1:.2f}, Recall: {recall:.2f}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{folder_name}/confusion_matrix.png")
    plt.close()

    # Graficar y guardar la gráfica de pérdida
    plt.plot(loss_history, label=f"{config['activation_function']} - {config['layers']}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(f"{folder_name}/loss_history.png")
    plt.close()
