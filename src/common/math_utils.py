def threshold(x, max_ratio: int):
    return min(max_ratio, max(x, 1/max_ratio))
