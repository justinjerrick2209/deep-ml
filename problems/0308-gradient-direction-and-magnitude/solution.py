# Cell 1
import numpy as np
def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    Args:
        gradient: A list representing the gradient vector
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    
    arr_grad = np.array(gradient)
    magnitude = np.linalg.norm(arr_grad, ord = 2)
    if magnitude == 0:
        direction = np.zeros_like(arr_grad, dtype=float)
        descent_direction = np.zeros_like(arr_grad, dtype=float)
    else:
        direction = arr_grad / magnitude
        descent_direction = -direction
    dict_grad = {
        "magnitude": magnitude,
        "direction": direction,
        "descent_direction": descent_direction
    }
    return dict_grad