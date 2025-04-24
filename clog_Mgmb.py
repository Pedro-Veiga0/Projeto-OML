import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

t = 0 
p_chapeu = 0

w = np.array([0, 0, 0])
x = np.array([[1, -1, 1], [1, -1, -1], [1, 0, 0], [1, 1, 1], [1, -1, 0], [1, 1, -1]])
y = np.array([0, 1, 1, 1, 0, 0])
w_new = np.zeros_like(w) 

N = len(x)
lr = 0.1
tol = 10**-6
batch_size = 3


while t < 50 or np.linalg.norm(w_new-w) > tol:
    batch_indices = np.random.choice(N, batch_size, replace=False)
    print(batch_indices)
    p_chapeu = sigmoid(np.dot(x[batch_indices],w.T))
    print(p_chapeu)

    d = (1/batch_size)*np.dot((p_chapeu-y[batch_indices]), x[batch_indices])

    w_new = w - (lr*d)
    w = w_new.copy()
    t += 1
    print(f"Iteration {t}: d = {d}, ||w_new - w|| = {np.linalg.norm(w_new-w)}")