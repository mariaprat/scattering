import math
import torch
from . import fourier
import numpy as np


def filters(shape, J, L, dtype, device='cpu'):
    filters = {'phi_hat' : {}, 'psi_hat' : {}}

    phi_hat = fourier.fft(gabor(shape, sigma=0.8 * 2**(J - 1), theta=0.0, chi=0.0, slant=1.0, dtype=dtype)).real
    for j in range(J):
        filters['phi_hat'][j] = fourier.periodize(fourier.complexify(phi_hat), 2**j).to(device)
        filters['psi_hat'][j] = {}
        for l in range(L):
            filters['psi_hat'][j][l] = {}
            psi = morlet(shape, sigma=0.8 * 2**j, theta=math.pi*(1/2 - (l + 1)/L), 
                         chi=(3/4)*math.pi/2**j, slant=4/L, dtype=dtype)
            psi_hat = fourier.fft(psi).real

            for resolution in range(j + 1):
                if resolution != J: 
                    filters['psi_hat'][j][l][resolution] = fourier.periodize(fourier.complexify(psi_hat), 2**resolution).to(device)
                
    return filters


def gabor(shape, sigma, theta, chi, slant, dtype):
    rx = shape[0]
    ry = shape[1]
    x, y = torch.meshgrid(torch.arange(-rx, rx), torch.arange(-ry, ry), indexing='ij')
     
    x_rotated = x * math.cos(theta) + y * math.sin(theta)

    rotation = torch.tensor([[math.cos(theta), -math.sin(theta)], 
                             [math.sin(theta), math.cos(theta)]])
    rotation_inv = torch.tensor([[math.cos(theta), math.sin(theta)], 
                                 [-math.sin(theta), math.cos(theta)]])
    
    curvature = torch.tensor([[1, 0], [0, slant**2]]) / (2 * sigma**2)
    curvature = torch.mm(rotation, torch.mm(curvature, rotation_inv)) 

    gabor_aux = torch.exp(1j*chi*x_rotated)
    gabor_aux *= torch.exp(-(curvature[0, 0]*x**2 + (curvature[1, 0] + curvature[0, 1])*x*y + curvature[1, 1]*y**2))
    gabor_aux *= slant / (2 * math.pi * sigma**2)
    
    gabor = torch.zeros((rx, ry), dtype=dtype)

    for i in [0, 1]:
        for j in [0, 1]:
            gabor += gabor_aux[i*rx:(i + 1)*rx, j*ry:(j + 1)*ry]
    
    return gabor
    

def morlet(shape, sigma, theta, chi, slant, dtype):
    gabor_aux = gabor(shape, sigma, theta, chi, slant, dtype)
    return gabor_aux - torch.sum(gabor_aux) / torch.sum(gabor_aux.abs()) * gabor_aux.abs()
    