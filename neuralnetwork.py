import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
m = 400  # number of samples
N = int(m / 2)  # number of points per class
D = 2  # dimensionality
X = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8')
a = 4  # maximum ray of the flower

for j in range(2):
    ix = range(N * j, N * (j + 1))
    t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
    r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title("Data Distribution")
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

input_dim = 2
hidden_dim = 4
output_dim = 1

W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def compute_loss(A2, y):
    m = y.shape[0]
    logprobs = -np.log(A2) * y - np.log(1 - A2) * (1 - y)
    loss = 1./m * np.sum(logprobs)
    return loss

def backward_propagation(X, y, Z1, A1, Z2, A2):
    m = X.shape[0]

    dZ2 = A2 - y
    dW2 = 1./m * np.dot(A1.T, dZ2)
    db2 = 1./m * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = 1./m * np.dot(X.T, dZ1)
    db1 = 1./m * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.01):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

num_iterations = 10000
learning_rate = 0.01
losses = []

for i in range(num_iterations):
    Z1, A1, Z2, A2 = forward_propagation(X)
    loss = compute_loss(A2, y)
    losses.append(loss)
    
    dW1, db1, dW2, db2 = backward_propagation(X, y, Z1, A1, Z2, A2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    if i % 1000 == 0:
        print(f"Loss after iteration {i}: {loss}")

# Step 9: Visualize the Results
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Decision boundary plot
def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title("Decision Boundary")
    plt.show()

def predict(X):
    _, _, _, A2 = forward_propagation(X)
    return A2 > 0.5

plot_decision_boundary(lambda x: predict(x), X, y)
