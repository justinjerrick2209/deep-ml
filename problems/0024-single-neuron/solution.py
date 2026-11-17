# Cell 1
import math
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    for x in features: 
        z = 0 
        for i in range(len(weights)): 
            z += x[i] * weights[i]
        z += bias
        p = 1 / (1 + math.exp(-z))
        probabilities.append(p)
    mse = 0
    for i in range(len(labels)): 
        mse += (probabilities[i] - labels[i])**2
    mse = mse/len(labels) 
    return probabilities, mse