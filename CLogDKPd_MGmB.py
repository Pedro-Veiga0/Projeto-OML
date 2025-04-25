import numpy as np
from utils import plot_decision_boundary
import matplotlib.pyplot as plt


class CLogDKPd_MGmB:
    def __init__(self, kernel=1, learning_rate=0.5, n_iter=2000, batch_size=None, tolerance=1e-6,verbose=False):
        """
        Logistic Classifier with Kernelized Polynomial Decision Boundary using Mini-Batch Gradient Descent.
        Parameters:
        kernel : int
            Degree of the polynomial kernel.
        learning_rate : float
            Learning rate for gradient descent.
        n_iter : int
            Maximum number of iterations for gradient descent.
        batch_size : int
            Size of the mini-batch for gradient descent.
            If None, uses the entire dataset for each iteration.
        verbose : bool
            If True, prints the progress of the training.
        """
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.max_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.alpha = None
        self.X_train = None  # armazenar X com bias
        self.errors = []
        self.tolerance = tolerance

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        N = X.shape[0]
        self.X_train = np.hstack((np.ones((N, 1)), X))  # adiciona termo de bias
        self.alpha = np.zeros(N)
        self.errors = []

        # Matriz de Gram com kernel polinomial
        K = (self.X_train @ self.X_train.T) ** self.kernel
        for t in range(self.max_iter):
            batch_indices = np.random.choice(N, self.batch_size, replace=False)
            K_batch = K[batch_indices]
            y_batch = y[batch_indices]

            y_pred = self.sigmoid(K_batch @ self.alpha)  # (batch_size,)
            error = y_pred - y_batch # (batch_size,)
            self.errors.append(np.mean(np.abs(error)))

            # basicaly each error is multiplied by the corresponding row in K_batch and summ along axis = 0, but we can do it in a vectorized way
            # numpy broadcasting will take care of the multiplication
            grad = (error @ K_batch) / self.batch_size # (batch_size, N)

            if self.verbose:
                print(f"Iteration {t}, Norm of grad: {np.linalg.norm(grad)}")
            if np.linalg.norm(grad) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {t}")
                break
            self.alpha -= self.learning_rate * grad

        return self

    def predict_proba(self, X):
        X_test = np.hstack((np.ones((X.shape[0], 1)), X))  # adiciona termo de bias
        K_test = (X_test @ self.X_train.T) ** self.kernel  # (M, N)
        probs = self.sigmoid(K_test @ self.alpha)
        return probs

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_errors(self):
        return self.errors

# Teste do classificador
if __name__ == "__main__":
    # Dados XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # Parâmetros do classificador
    kernel = 2
    model = CLogDKPd_MGmB(kernel=kernel, learning_rate=0.5, n_iter=2000, batch_size=4, verbose=True)
    
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