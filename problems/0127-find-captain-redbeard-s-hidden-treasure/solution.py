# Cell 1
def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.
  Returns:
        float: The x-coordinate of the minimum point.
    """
    f = lambda x: x**4 - 3*x**3 + 2
    critical_pts = [0.0, 2.25]
    return min(critical_pts, key = f)