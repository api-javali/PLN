import math
import random

class MLPClassifier:
    """MLP com backpropagation para classificação"""
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicializa pesos com Xavier initialization
        self.weights1 = self._xavier_init(input_size, hidden_size)
        self.weights2 = self._xavier_init(hidden_size, output_size)
        self.bias1 = [0.0] * hidden_size
        self.bias2 = [0.0] * output_size
    
    def _xavier_init(self, rows, cols):
        """Xavier initialization para melhor convergência"""
        limit = math.sqrt(6.0 / (rows + cols))
        return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
    
    def sigmoid(self, x):
        """Função sigmoid"""
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    
    def sigmoid_derivative(self, x):
        """Derivada da sigmoid"""
        return x * (1.0 - x)
    
    def softmax(self, x):
        """Função softmax"""
        exp_x = [math.exp(i - max(x)) for i in x]
        sum_exp = sum(exp_x)
        return [i / sum_exp for i in exp_x]
    
    def forward(self, x):
        """Propagação para frente"""
        # Camada oculta
        hidden = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            for i in range(self.input_size):
                hidden[j] += x[i] * self.weights1[i][j]
            hidden[j] = self.sigmoid(hidden[j] + self.bias1[j])
        
        # Camada de saída
        output = [0.0] * self.output_size
        for j in range(self.output_size):
            for i in range(self.hidden_size):
                output[j] += hidden[i] * self.weights2[i][j]
            output[j] += self.bias2[j]
        
        return self.softmax(output), hidden
    
    def backward(self, x, y_true, output, hidden):
        """Backpropagation"""
        # Calcula erro da saída
        output_error = [output[i] - y_true[i] for i in range(self.output_size)]
        
        # Calcula erro da camada oculta
        hidden_error = [0.0] * self.hidden_size
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                hidden_error[i] += output_error[j] * self.weights2[i][j]
            hidden_error[i] *= self.sigmoid_derivative(hidden[i])
        
        # Atualiza pesos da camada de saída
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights2[i][j] -= self.learning_rate * output_error[j] * hidden[i]
        
        # Atualiza bias da camada de saída
        for j in range(self.output_size):
            self.bias2[j] -= self.learning_rate * output_error[j]
        
        # Atualiza pesos da camada oculta
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights1[i][j] -= self.learning_rate * hidden_error[j] * x[i]
        
        # Atualiza bias da camada oculta
        for j in range(self.hidden_size):
            self.bias1[j] -= self.learning_rate * hidden_error[j]
    
    def train(self, X, y, epochs=100):
        """Treina o modelo"""
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(X)):
                # Forward pass
                output, hidden = self.forward(X[i])
                
                # Calcula loss (cross-entropy)
                loss = -sum(y[i][j] * math.log(output[j] + 1e-10) for j in range(self.output_size))
                total_loss += loss
                
                # Backward pass
                self.backward(X[i], y[i], output, hidden)
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(X)
                print(f"Época {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def predict(self, x):
        """Faz predição"""
        probabilities, _ = self.forward(x)
        return probabilities
    
    def predict_class(self, x):
        """Retorna a classe predita"""
        probabilities = self.predict(x)
        return probabilities.index(max(probabilities)), max(probabilities)