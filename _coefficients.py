import numpy as np 
import torch
from . import fourier
from . import padding


def coefficients(self, x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        
    x = x.to(self.device)
        
    if len(x) == 2:
        x = x.unsqueeze(0)
    batch_size = x.shape[:-2]
    
    n_s0 = 1
    n_s1 = self.J*self.L
    n_s2 = self.L**2 * self.J*(self.J - 1) // 2
    
    n_coefficients = n_s0
    s0 = torch.empty(batch_size + (n_s0, *self.shape // 2**self.J), device=self.device)
    if self.m >= 1:
        s1 = torch.empty(batch_size + (n_s1, *self.shape // 2**self.J), device=self.device)
        n_coefficients += n_s1
    if self.m >= 2:
        s2 = torch.empty(batch_size + (self.L**2 * self.J*(self.J - 1) // 2, *self.shape // 2**self.J), device=self.device)
        n_coefficients += n_s2

    x = padding.pad(self.padding, self.padded_shape, x)
    x_hat = fourier.rfft(x)

    c0 = fourier.convolute(x_hat, self.filters['phi_hat'][0], 2**self.J).real
    s0[..., 0, :, :] = padding.unpad(self.padding, c0)

    if self.m >= 1:
        index_s1 = 0
        index_s2 = 0
        for j1 in range(self.J):
            for l1 in range(self.L):
                filter_j1 = self.filters['psi_hat'][j1][l1][0]
                u_j1 = fourier.convolute(x_hat, filter_j1, 2**j1).abs()
                u_j1_hat = fourier.rfft(u_j1)
                lowpass_j1 = self.filters['phi_hat'][j1]
                factor_j1 = 2**(self.J - j1)
                c1 = fourier.convolute(u_j1_hat, lowpass_j1, factor_j1).real
                s1[..., index_s1, :, :] = padding.unpad(self.padding, c1)
                index_s1 += 1
                
                if self.m == 2:
                    for j2 in range(j1 + 1, self.J):
                        for l2 in range(self.L):
                            filter_j1j2 = self.filters['psi_hat'][j2][l2][j1]
                            factor_j1j2 = 2**(j2 - j1)
                            u_j1j2 = fourier.convolute(u_j1_hat, filter_j1j2, factor_j1j2).abs()
                            u_j1j2_hat = fourier.rfft(u_j1j2)

                            lowpass_j2 = self.filters['phi_hat'][j2]
                            factor_j2 = 2**(self.J - j2)
                            c2 = fourier.convolute(u_j1j2_hat, lowpass_j2, factor_j2).real
                            s2[..., index_s2, :, :] = padding.unpad(self.padding, c2)
                            index_s2 += 1

    coefficients = torch.empty(batch_size + (n_coefficients, *self.shape // 2**self.J), device=self.device)
    coefficients[..., :n_s0, :, :] = s0
    if self.m >= 1:
        coefficients[..., n_s0:n_s0 + n_s1, :, :] = s1
    if self.m >= 2:
        if n_s2 > 0: coefficients[..., -n_s2:, :, :] = s2

    return coefficients.squeeze()
