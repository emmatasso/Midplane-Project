import numpy as np

def col_to_float_array(table, name):
    col = table[name]
    if hasattr(col, "filled"):
        return np.asarray(col.filled(np.nan), float)
    return np.asarray(col, float)

def masked_fallback_array(table, primary, fallback):
    col = table[primary].copy()
    if hasattr(col, "mask"):
        col[col.mask] = table[fallback][col.mask]
    return np.asarray(col, float)

