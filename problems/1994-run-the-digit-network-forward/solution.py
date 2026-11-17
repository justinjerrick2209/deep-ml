# Cell 1
import numpy as np
def digit_network_forward(image, weights, biases):
    # Flatten the image
    a = np.array(image).flatten().tolist()
    # Process each layer
    for l in range(len(weights)):
        new_a = []
        for i in range(len(weights[l])):
            z = 0
            for j in range(len(a)):
                z += weights[l][i][j] * a[j]
            z += biases[l][i]
            p = 1 / (1 + np.exp(-z))
            new_a.append(p)
        a = new_a

    # Find predicted digit
    digit = a.index(max(a))
    # Round activations
    activations = [round(x, 4) for x in a]
    return activations, digit