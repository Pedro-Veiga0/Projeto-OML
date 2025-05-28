import numpy as np

def gerar_classes_pizza(n_classes=5, n_pontos_por_classe=100, raio_min=1, raio_max=5, ruido_angular=None, delta_angulos = None):
    X = []
    y = []

    if ruido_angular is None:
       ruido_angular = np.full(n_classes, 0.2)
    for classe in range(n_classes):
        # Ângulo central do setor para a classe
        angulo_central = 2 * np.pi * classe / n_classes
        if delta_angulos is not None:
            angulo_central += delta_angulos[classe]
        
        # Geração de ângulos ao redor do centro, com ruído
        angulos = np.random.normal(loc=angulo_central, scale=ruido_angular[classe], size=n_pontos_por_classe)
        
        # Raio aleatório entre min e max
        raios = np.random.uniform(low=raio_min, high=raio_max, size=n_pontos_por_classe)
        
        # Converter para coordenadas cartesianas
        x = raios * np.cos(angulos)
        y_coords = raios * np.sin(angulos)
        z_coords = np.random.uniform(low=-1, high=+1, size=n_pontos_por_classe) * raios

        X.append(np.stack((x, y_coords, z_coords), axis=1))
        y += [classe] * n_pontos_por_classe

    X = np.vstack(X)
    y = np.array(y)
    return X, y
    