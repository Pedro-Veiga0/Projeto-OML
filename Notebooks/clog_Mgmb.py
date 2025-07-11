import numpy as np
from utils import plot_decision_boundary
import matplotlib.pyplot as plt


class CLog_MGmB:
    def __init__(self, learning_rate=0.5, n_iter=2000, batch_size=None, tolerance=1e-6, iterative=False, verbose=False):
        """
        Logistic Classifier using Mini-Batch Gradient Descent.
        Parameters:
            learning_rate : float
                Learning rate for gradient descent.
            n_iter : int
                Maximum number of iterations for gradient descent.
            batch_size : int
                Size of the mini-batch for gradient descent.
                If None, uses the entire dataset for each iteration.
            tolerance : float
                Tolerance for convergence. If the norm of the gradient is less than this value, the algorithm stops.
            iterative : bool
                If True, uses an iterative approach for gradient descent. Sets batch_size to 1.
            verbose : bool
                If True, prints the progress of the training.
        """
        self.learning_rate = learning_rate
        self.max_iter = n_iter
        self.batch_size = batch_size if not iterative else 1
        self.verbose = verbose
        self.weights = None
        self.X_train = None  # armazenar X com bias
        self.errors = []
        self.kernel = 1  # grau do polinômio
        self.tolerance = tolerance
        self.iterative = iterative

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        N = X.shape[0]
        self.X_train = np.hstack((np.ones((N, 1)), X))  # adiciona termo de bias

        features = self.X_train.shape[1]
        self.weights = np.random.normal(0, 1, (features))  # inicializa pesos com média 0 e desvio padrão 0.1
        self.errors = []

        for t in range(self.max_iter):
            if self.iterative:
                # Iterative approach
                erros = [] # armazena os erros de cada iteração
                grads = []
                indices = np.arange(N)
                np.random.shuffle(indices)  # Embaralha os índices
                for i in indices:
                    x_i = self.X_train[i]
                    y_i = y[i]

                    y_pred = self.sigmoid(x_i @ self.weights)  # (1,)
                    error = y_pred - y_i
                    erros.append(np.abs(error))

                    grad = error * x_i
                    grads.append(grad)
                    self.weights -= self.learning_rate * grad

                self.errors.append(np.mean(erros)) # erro médio da iteração t
                grad = np.mean(grads, axis=0)  # gradiente médio da iteração t
            else:
                batch_indices = np.random.choice(N, self.batch_size, replace=False)
                x_batch = self.X_train[batch_indices]
                y_batch = y[batch_indices]

                y_pred = self.sigmoid(x_batch @ self.weights)  # (batch_size,)
                error = y_pred - y_batch
                self.errors.append(np.mean(np.abs(error)))

                # basicaly each error is multiplied by the corresponding x_tilde and summ along axis = 0, but we can do it in a vectorized way
                grad = (error @ x_batch) / self.batch_size
                self.weights -= self.learning_rate * grad

            if self.verbose:
                print(f"Iteration {t}, Norm of grad: {np.linalg.norm(grad)}")
            
            # Previne convergência prematura do algoritmo
            if self.batch_size != 1 and np.linalg.norm(grad) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {t}")
                break

        return self

    def predict_proba(self, X):
        X_test = np.hstack((np.ones((X.shape[0], 1)), X))  # adiciona termo de bias
        probs = self.sigmoid(X_test @ self.weights.T)
        return probs

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_errors(self):
        return self.errors

# Teste do classificador
if __name__ == "__main__":
    # Dados XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    model = CLog_MGmB( learning_rate=0.5, n_iter=2000, batch_size=4, verbose=True, iterative=True)
    
    # Treinamento
    model.fit(X, y)
    # Plot da fronteira de decisão
    plot_decision_boundary(model, X, y, resolution=500)
    
    # Plot da curva de aprendizado
    plt.figure()
    plt.plot(model.get_errors())
    plt.title("Curva de Aprendizado")
    plt.xlabel("Iteração")
    plt.ylabel("Erro médio no batch")
    plt.grid(True)
    plt.show()