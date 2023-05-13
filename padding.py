import torch


def padded_shape(mode, shape, J):
    if mode == 'none': return shape
    return (shape // 2**J + 2) * 2**J


def pad(mode, padded_shape, x):
    if mode == 'none':
        return x
    if mode == 'reflect':
        return reflect(x, padded_shape)
    raise ValueError('\'padding\' must be \'none\' or \'reflect\'')


def unpad(mode, x):
    if mode == 'none':
        return x
    return x[..., 1:-1, 1:-1]


def reflect(x, padded_shape):
    n = x.shape[-2]
    m = x.shape[-1]
        
    pad_top = (padded_shape[0] - n) // 2
    pad_bottom = (padded_shape[0] - n + 1) // 2
    pad_left = (padded_shape[1] - m) // 2
    pad_right = (padded_shape[1] - m + 1) // 2
        
    batch_shape = x.shape[:-2]
    signal_shape = x.shape[-2:]
    
    if pad_top == n: 
        pad_top -= 1
        pad_bottom -= 1
    if pad_left == m: 
        pad_left -= 1
        pad_right -= 1
    
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    x = torch.nn.ReflectionPad2d(padding)(x.view((-1,) + signal_shape))

    if (padded_shape[0] - n) // 2 == n:
        top_line = x[..., 1, :].unsqueeze(-2)
        bottom_line = x[..., x.shape[-2] - 2, :].unsqueeze(-2)
        x = torch.cat([top_line, x, bottom_line], dim=-2)
    if (padded_shape[1] - m) // 2 == m:
        left_line = x[..., 1].unsqueeze(-1)
        right_line = x[..., x.shape[-1] - 2].unsqueeze(-1)
        x = torch.cat([left_line, x, right_line], dim=-1)
        
    x = x.view(batch_shape + x.shape[-2:])
    return x
