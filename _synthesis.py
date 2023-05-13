import math
import numpy as np
import time
import torch
from . import fourier


def synthesis(self, mode, image=None, estimator=None, seed=None, steps=100,
              learning_rate=1.0, algorithm='adam', clipping=False,
              remove_checkered_effect=True, remove_checkered_effect_mode='soft',
              verbose=True):
    if image is None and estimator is None:
        raise ValueError('Either \'image\' or \'estimator\' must be provided')
    if image is not None and estimator is not None:
        raise ValueError('Only one of \'image\' or \'estimator\' can be provided')
    
    if mode == 'coefficients':
        estimate = lambda x : self.coefficients(x)
    else:
        raise ValueError(f'\'mode\' must be \'coefficients\'')

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if image is not None:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        estimator = estimate(image)
        mean = image.mean()
        std = image.std()
    else:
        mean = 0.0
        std = 1.0

    synthesized_image = torch.normal(mean=mean, std=std, size=(self.shape[0], self.shape[1]))
    synthesized_image.requires_grad = True

    if algorithm == 'lbfgs':
        optimizer = torch.optim.LBFGS([synthesized_image], lr=learning_rate,
                                      max_iter=steps, max_eval=None,
                                      tolerance_grad=1e-20, tolerance_change=1e-20,
                                      history_size=min(steps // 2, 150))
    elif algorithm == 'adam':
        optimizer = torch.optim.Adam([synthesized_image], lr=learning_rate)
    else:
        raise ValueError('\'algorithm\' must be \'lbfgs\' or \'adam\'')

    if verbose: print('estimator size: ', int(len(estimator.flatten())))

    def closure():
        optimizer.zero_grad()
        synthesized_estimator = estimate(synthesized_image)
        loss = _l2_loss(synthesized_estimator, estimator)
        if algorithm != 'lbfgs' and step % 10 == 0:
            loss_history.append(loss.detach().numpy())
        loss.backward()
        if clipping:
            synthesized_image.data = torch.clip(synthesized_image, min=mean - 3*std, max=mean + 3*std)
        if remove_checkered_effect:
            if algorithm == 'lbfgs' or step % 10 == 0:
                synthesized_image_hat = fourier.fftshift(fourier.rfft(synthesized_image))
                synthesized_image.data = fourier.ifft(fourier.fftshift(circle_mask*synthesized_image_hat)).real
        return loss
    
    loss_history = []    
    circle_mask = _circle_mask(self.shape, remove_checkered_effect_mode)
    start = time.time()
    if algorithm == 'lbfgs':
        optimizer.step(closure)
    else:
        for step in range(steps):
            if verbose: print(f'step {step}', end='\r')
            optimizer.step(closure)
    end = time.time()
    
    if remove_checkered_effect:
        synthesized_image_hat = fourier.fftshift(fourier.rfft(synthesized_image))
        synthesized_image = fourier.ifft(fourier.fftshift(circle_mask*synthesized_image_hat)).real

    loss = _l2_loss(estimate(synthesized_image), estimator)
    if verbose:
        print(f'final loss: {loss:.13e}')
        print(f'time: {end - start:.2f}s')
    return synthesized_image.detach().numpy(), loss_history


def _l2_loss(x, y):
    return ((x - y)**2).mean()


def _circle_mask(shape, mode='soft'):
    n = shape[0]
    m = shape[1]
    if n == 64:
        r = 26
    elif n == 256:
        r = 100
        rm = 90
        rs = 26
    else:
        r = int(max(n / 2, m / 2))
    mask = np.sqrt(((torch.arange(n) - (n - 1) / 2)**2).unsqueeze(-1) + (torch.arange(m) - (m - 1) / 2)**2)
    circle_mask = (mask <= r).type(torch.uint8)
    if mode == 'soft' and n == 256:
        rs_mask = (mask <= rs).type(torch.uint8)
        rm_mask = (np.logical_and(mask > rm, mask <= r)).type(torch.uint8)
        exponential_mask = np.exp(-1.8*(mask - rs)/rm) * (1 - rm_mask)
        linear_mask = (np.exp(-1.8*((rm-rs)/rm))-3.2*(mask - rm)/rm) * rm_mask
        return np.minimum(np.maximum(np.maximum(exponential_mask, linear_mask), rs_mask), circle_mask)
    return circle_mask