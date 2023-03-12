import numpy as np
from collections.abc import Iterable

def _padding_tuple(padding, ndim, ignore_dims=2):
    if not isinstance(padding, Iterable):
        padding = [*(0 for i in range(ignore_dims)), *(padding for i in range(ndim - ignore_dims))]
    assert len(padding) == ndim, f'padding {padding} doesn\'t match {ndim}'
    new_padding = []
    for p in padding:
        if isinstance(p, Iterable):
            p1, p2, *_ = p
        else:
            p1, p2 = p, p
        new_padding.append((p1, p2))
    return tuple(new_padding)

def _np_padding(input : np.ndarray, padding=0, ignore_dims=2):
    new_padding = _padding_tuple(padding, input.ndim, ignore_dims=ignore_dims)
    return np.pad(input, new_padding), new_padding

def _np_conv(input : np.ndarray, kernel : np.ndarray, stride=1, padding=0, dilation=1, keep_in_channel=False, batch_kernel=False, detail=False):
    kernel_ndim = kernel.ndim - 2
    if batch_kernel:
        kernel_ndim -= 1
    batch_dim = (input.ndim - 1) - kernel_ndim
    assert batch_dim >= 0, f'shape mismatch for input {input.shape} and kernel {kernel.shape}'
    batch_shape = input.shape[:batch_dim]

    # now input is (N, in_channels, *size)
    input = input.reshape(-1, *input.shape[batch_dim:])
    if batch_kernel:
        assert input.shape[0] == kernel.shape[0], f'batch mismatch for input {input.shape} and kernel {kernel.shape}'
    else:
        kernel = np.expand_dims(kernel, 0)
    input, *_ = _np_padding(input, padding=padding, ignore_dims=2)

    in_channels = input.shape[1]
    _, out_channels, kernel_in_channels = kernel.shape[:3]
    groups = in_channels // kernel_in_channels
    assert in_channels % groups == 0 and out_channels % groups == 0, f'group shape mismatch for input {input.shape} and kernel {kernel.shape}'

    if not isinstance(stride, Iterable):
        stride = [stride for i in range(kernel_ndim)]
    stride = tuple(stride)
    assert len(stride) == kernel_ndim, f'stride {stride} doesn\'t match kernel shape {kernel.shape}'
    if not isinstance(dilation, Iterable):
        dilation = [dilation for i in range(kernel_ndim)]
    dilation = tuple(dilation)
    assert len(dilation) == kernel_ndim, f'dilation {dilation} doesn\'t match kernel shape {kernel.shape}'

    # now input is (N, groups, in_channels / groups, *size)
    input = np.reshape(input, (input.shape[0], groups, kernel_in_channels, *input.shape[2:]))
    # use np.repeat (but not np.tile) to increase size
    input = np.repeat(input, out_channels // groups, axis=1)

    data_shape = input.shape[3:]
    _batch_shape = input.shape[:3]
    kernel_shape = kernel.shape[3:]

    batch_stride, data_stride = input.strides[:3], input.strides[3:]
    strided_shape = tuple((1 + (i - d * (k - 1) - 1) // s) for i, k, s, d in zip(data_shape, kernel_shape, stride, dilation))
    new_shape = (*_batch_shape, *strided_shape, *kernel_shape)
    new_stride = ((s * ds) for s, ds in zip(stride, data_stride))
    new_dilated_stride = ((d * ds) for d, ds in zip(dilation, data_stride))
    new_stride = (*batch_stride, *new_stride, *new_dilated_stride)

    # strided is (N, out_channels, in_channels / groups, *out_shape, *kernel_size)
    strided = np.lib.stride_tricks.as_strided(input, shape=new_shape, strides=new_stride)
    kernel = np.reshape(kernel, (kernel.shape[0], out_channels, kernel_in_channels, *(1 for i in strided_shape), *kernel_shape))
    res = np.reshape(strided * kernel, (input.shape[0], out_channels, kernel_in_channels, *strided_shape, -1))
    # res is (N, out_channels, *out_shape)
    res = np.sum(res, axis=-1)

    if not keep_in_channel:
        res = np.sum(res, axis=2)

    res = np.reshape(res, (*batch_shape, *res.shape[1:]))

    if detail:
        return res, padding, stride, dilation

    return res

def _make_weight(in_channels, kernel_size):
    kernel_size_numel = int(np.prod(kernel_size))
    repeat = (in_channels, 1, *(1 for _ in kernel_size))
    res = np.eye(kernel_size_numel)
    res = np.reshape(res, (kernel_size_numel, 1, *kernel_size))
    res = np.tile(res, repeat)
    return res

def _unfold(input, kernel_size, padding=0, stride=1, dilation=1):
    ddim = input.ndim - len(kernel_size)
    assert ddim in (-1, 0), f'size mismatch of input ({input.shape}) and kernel ({kernel_size})'
    if ddim == -1:
        in_channels = input.shape[0]
    else:
        in_channels = input.shape[1]
    kernel_size_numel = int(np.prod(kernel_size))
    weight = _make_weight(in_channels, kernel_size)
    res = _np_conv(input, weight, stride=stride, padding=padding, dilation=dilation)
    new_shape = (in_channels * kernel_size_numel, -1)
    if ddim != -1:
        new_shape = (input.shape[0], *new_shape)
    return np.reshape(res, new_shape)
