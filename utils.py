import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, resolution=100, figsize=(8, 6)):
    if X.shape[1] < 2:
        raise ValueError("A plotagem só é suportada para dados com pelo menos 2 características")
    
    # Define limites do gráfico
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Gera grid
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, resolution),
                           np.linspace(x2_min, x2_max, resolution))
    
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    
    # Faz predições usando o modelo
    Z = model.predict(grid_points)
    Z = Z.reshape(xx1.shape)
    
    # Plota
    plt.figure(figsize=figsize)
    classes = np.unique(y)
    plt.contourf(xx1, xx2, Z, alpha=0.4)
    for cls in classes:
        plt.scatter(
            X[y == cls, 0], 
            X[y == cls, 1], 
            label=f"Classe {cls}", 
            edgecolors='k'
        )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    if hasattr(model, 'kernel'):
        plt.title(f"Decision Bound (Kernel Polinomial degree: {model.kernel})")
    else:
        plt.title("Decision Boundary")
    # plt.title(f"Decision Bound (Kernel Polinomial degree {model.kernel})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()