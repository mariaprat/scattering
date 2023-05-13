import numpy as np
import torch
from . import filters
from . import padding

class Scattering:
    def __init__(self, shape, J, L, m, padding_mode='none', precision='double'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if len(shape) != 2:
            raise ValueError(f'\'shape\' must have length 2, not {len(shape)}')
        if not isinstance(shape[0], int) or not isinstance(shape[1], int):
            raise ValueError('\'shape\' must have integer elements')
        self.shape = np.array(shape, dtype=int)

        if J != int(J):
            raise ValueError('\'J\' must be an integer')
        self.J = int(J)
        if self.J < 1:
            raise ValueError('\'J\' must be greater or equal to 1')
        if 2**self.J > min(shape[0], shape[1]):
            raise ValueError('The smallest dimension in \'shape\' must be larger or equal than 2^J')
        
        if not isinstance(L, int):
            raise ValueError('\'L\' must be an integer')
        self.L = int(L)
        if self.L < 1:
            raise ValueError('\'L\' must be greater or equal to 1')

        self.padding = padding_mode
        self.padded_shape = padding.padded_shape(self.padding, self.shape, self.J)

        if m not in [0, 1, 2]:
            raise ValueError(f'\'m\' must be 0 <= m <= 2, not {m}')
        self.m = m

        if precision not in {'single', 'double'}:
            raise ValueError(f'\'precision\' must be either \'single\' or \'double\', not {precision}')
        if precision == 'single':
            self.dtype = torch.complex64
        elif precision == 'double':
            self.dtype = torch.complex128

        self.filters = filters.filters(self.padded_shape, self.J, self.L, self.dtype, self.device)
        

    def filters(self):
        return self.filters
        
    from ._coefficients import coefficients
    from ._synthesis import synthesis