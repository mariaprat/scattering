import torch


complexify = lambda z : torch.complex(z, torch.zeros(z.shape, dtype=z.dtype, device=z.device))

fftshift = lambda x : torch.fft.fftshift(x, dim=(-2,-1))
ifftshift = lambda x : torch.fft.ifftshift(x, dim=(-2,-1))
fft = lambda x : torch.fft.fft2(x, dim=(-2,-1))
rfft = lambda x : fft(complexify(x))
ifft = lambda x : torch.fft.ifft2(x, dim=(-2,-1))


def subsample(f, factor):
    batch_shape, signal_shape = f.shape[:-2], f.shape[-2:]
    f = f.view(-1, factor, signal_shape[-2] // factor, factor, signal_shape[-1] // factor)
    f = f.mean(3, keepdim=False).mean(1, keepdim=False)
    return f.view(batch_shape + (signal_shape[-2] // factor, signal_shape[1] // factor))


def periodize(f_hat, factor):
    n = f_hat.shape[0]
    m = f_hat.shape[1]
    f_hat = f_hat.view(factor, n // factor, factor, m // factor)
    return f_hat.sum(2, keepdim=False).sum(0, keepdim=False)


def convolute(x_hat, f_hat, subsampling_factor=1):
    convolution_hat = torch.mul(x_hat, f_hat)
    perodized_convolution_hat = subsample(convolution_hat, subsampling_factor)
    return ifft(perodized_convolution_hat)
    