import math

class FloatWrapper:
    """
    Un envoltorio alrededor del float estándar de Python para imitar la API de Posit64/Posit32.
    Esto permite que la misma lógica del solver se ejecute en floats para verificación algorítmica.
    """
    def __init__(self, value=0.0):
        if isinstance(value, FloatWrapper):
            self.value = value.value
        else:
            self.value = float(value)

    def __add__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return FloatWrapper(self.value + other_val)

    def __sub__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return FloatWrapper(self.value - other_val)

    def __mul__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return FloatWrapper(self.value * other_val)

    def __truediv__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return FloatWrapper(self.value / other_val)

    def __pow__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return FloatWrapper(self.value ** other_val)

    def __lt__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return self.value < other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return self.value > other_val

    def __le__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return self.value <= other_val

    def __ge__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return self.value >= other_val

    def __eq__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return self.value == other_val

    def __ne__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else float(other)
        return self.value != other_val
    
    def __neg__(self):
        return FloatWrapper(-self.value)

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"{self.value}"

    def sqrt(self):
        return FloatWrapper(math.sqrt(self.value))
    
    def to_float(self):
        return self.value

import numpy as np

class Float32Wrapper:
    """Envoltorio para simular precisión simple (float32)."""
    def __init__(self, value=0.0):
        if isinstance(value, Float32Wrapper):
            self.value = value.value
        else:
            self.value = np.float32(value)

    def __add__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return Float32Wrapper(self.value + other_val)
    
    def __sub__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return Float32Wrapper(self.value - other_val)
    
    def __mul__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return Float32Wrapper(self.value * other_val)
    
    def __truediv__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return Float32Wrapper(self.value / other_val)
    
    def __lt__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return self.value < other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return self.value > other_val

    def __le__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return self.value <= other_val

    def __ge__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return self.value >= other_val

    def __eq__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return self.value == other_val
    
    def __ne__(self, other):
        other_val = other.value if isinstance(other, Float32Wrapper) else other
        return self.value != other_val

    def __neg__(self):
        return Float32Wrapper(-self.value)

    def __float__(self):
        return float(self.value)
    
    def __repr__(self):
        return f"{self.value}"
    
    def sqrt(self):
        return Float32Wrapper(np.sqrt(self.value))


class Float16Wrapper:
    """Envoltorio para simular media precisión (float16)."""
    def __init__(self, value=0.0):
        if isinstance(value, Float16Wrapper):
            self.value = value.value
        else:
            self.value = np.float16(value)

    def __add__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return Float16Wrapper(self.value + other_val)
    
    def __sub__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return Float16Wrapper(self.value - other_val)
    
    def __mul__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return Float16Wrapper(self.value * other_val)
    
    def __truediv__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return Float16Wrapper(self.value / other_val)
    
    def __lt__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return self.value < other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return self.value > other_val

    def __le__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return self.value <= other_val

    def __ge__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return self.value >= other_val

    def __eq__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return self.value == other_val
    
    def __ne__(self, other):
        other_val = other.value if isinstance(other, Float16Wrapper) else other
        return self.value != other_val

    def __neg__(self):
        return Float16Wrapper(-self.value)

    def __float__(self):
        return float(self.value)
    
    def __repr__(self):
        return f"{self.value}"
    
    def sqrt(self):
        return Float16Wrapper(np.sqrt(self.value))
