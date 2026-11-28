# Cell 1
import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray,
                 initial_weights: np.ndarray, initial_bias: float,
                 learning_rate: float, epochs: int) -> tuple[np.ndarray, float, list[float]]:
    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []
    n = len(labels)
    for epoch in range(epochs):
        # 1. Forward propagation
        z = np.dot(features, weights) + bias
        predictions = 1 / (1 + np.exp(-z))
        # 2. Calculate MSE
        error = predictions - labels
        mse = np.mean(error ** 2)
        mse_values.append(mse)
        # 3. Backpropagation
        dz = 2 * error * predictions * (1 - predictions)
        dw = np.dot(features.T, dz) / n
        db = np.mean(dz)
        # 4. Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db
    # 5. Round results
    updated_weights = np.round(weights, 4)
    updated_bias = round(float(bias), 4)
    mse_values = [round(float(x), 4) for x in mse_values]
    return updated_weights, updated_bias, mse_values