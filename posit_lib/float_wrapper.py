import math
import numpy as np
from functools import partial

class FloatWrapper:
    """
    Un envoltorio genérico alrededor del float estándar o tipos numpy (float16, float32)
    para imitar la API de Posit64/Posit32 y simular comportamiento de precisión reducida.
    """
    def __init__(self, value=0.0, dtype=np.float64):
        self.dtype = dtype
        if isinstance(value, FloatWrapper):
            # Si copiamos de otro wrapper, tomamos su valor y lo casteamos a NUESTRO dtype
            self.value = self.dtype(value.value)
        else:
            self.value = self.dtype(value)

    def _wrap(self, value):
        # Helper para envolver resultados manteniendo el dtype actual
        # Aseguramos que el valor se trunque al dtype correcto
        return FloatWrapper(value, dtype=self.dtype)

    def __add__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        # Operación en el dominio del tipo, forzando cast
        res = self.dtype(self.value + self.dtype(other_val))
        return self._wrap(res)

    def __sub__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        res = self.dtype(self.value - self.dtype(other_val))
        return self._wrap(res)

    def __mul__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        res = self.dtype(self.value * self.dtype(other_val))
        return self._wrap(res)

    def __truediv__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        res = self.dtype(self.value / self.dtype(other_val))
        return self._wrap(res)

    def __pow__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        # pow puede promocionar tipos, forzamos cast al final
        res = self.dtype(self.value ** self.dtype(other_val))
        return self._wrap(res)

    def __lt__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        return self.value < other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        return self.value > other_val

    def __le__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        return self.value <= other_val

    def __ge__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        return self.value >= other_val

    def __eq__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        return self.value == other_val

    def __ne__(self, other):
        other_val = other.value if isinstance(other, FloatWrapper) else other
        return self.value != other_val
    
    def __neg__(self):
        return self._wrap(-self.value)

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"{self.value}"

    def sqrt(self):
        return self._wrap(np.sqrt(self.value, dtype=self.dtype))
    
    def to_float(self):
        return self.value
    
    def exp(self):
         return self._wrap(np.exp(self.value, dtype=self.dtype))

    def log(self):
         return self._wrap(np.log(self.value, dtype=self.dtype))
         
    def abs(self):
        return self._wrap(np.abs(self.value, dtype=self.dtype))

# Definiciones parciales para compatibilidad y uso fácil
# functools.partial crea un callable que funciona como constructor
Float16Wrapper = partial(FloatWrapper, dtype=np.float16)
Float32Wrapper = partial(FloatWrapper, dtype=np.float32)
Float64Wrapper = partial(FloatWrapper, dtype=np.float64)

# Asignar nombres para que __name__ sea útil (útil para skfolio_adapter)
Float16Wrapper.__name__ = "Float16Wrapper"
Float32Wrapper.__name__ = "Float32Wrapper"
Float64Wrapper.__name__ = "Float64Wrapper"
