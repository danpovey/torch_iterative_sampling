import os

import random # for testing and diagnostics..
import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional, Union
from torch.utils.cpp_extension import load

VERBOSE = True

def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_scheduled_sampling_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_sampling_cpu')
    torch_sampling_cpu = load(
        name='sample_combined_forward_cpu',
        sources=[
            _resolve('sampling_cpu.cpp'),
        ],
        verbose=VERBOSE,
    )


try:
    import torch_scheduled_sampling_cuda
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_sampling_cuda')
    torch_sampling_cuda = None
    if torch.cuda.is_available():
        torch_sampling_cuda = load(
            name='sample_combined_forward_cuda',
            sources=[
                _resolve('sampling_cuda.cpp'),
                _resolve('sampling_cuda_kernel.cu'),
            ],
            verbose=VERBOSE,
        )



def _sample_combined_forward_dispatcher(
        probs: torch.Tensor,
        rand: torch.Tensor,
        K: int, input_is_log: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dispatcher for sample_combined_forward
    """
    if probs.is_cuda:
        if torch_sampling_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_scheduled_sampling_cuda.sample_combined_forward_cpu(
            probs, rand, K, input_is_log)
    else:
        return torch_scheduled_sampling_cpu.sample_combined_forward_cpu(
            probs, rand, K, input_is_log)

_max_bits = 54  # used in sample_combined_forward and sample_combined_backward,
                # see comment in sample_combined_forward.

def sample_combined_forward(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
    """
    Sample from a distribution that is the product of softmaxes.  We will sample
    K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
    for a computed beta.
    Args:
         p: A Tensor of shape (*, N, M): either normalized log-probs (if input_is_log==False),
             or normalized probabilities; normalized along the M axis.  M must be
             a power of 2, and N must be in [1,2,3,4].
         K: An integer, the number of samples required, with 0 < K < N
   input_is_log:  True if p represents normalized log-probs, False if it represents
             probabilities.

    Returns: (indexes, combined_indexes, weights)
       indexes: of shape (*, K, N), for each of K samples from a distribution it contains
            an N-tuple of indexes saying which combination of indexes from the
            component distributions were sampled.
      combined_indexes: of shape(*, K),  contains the same information as `indexes` but
            in a different format, specifically:
               `combined_indexes[...,k] = sum_n indexes[...,k,n] * M**n`
       weights: of shape (*, K), gives the weight associated with each sample,
            which will equal max(p, beta) for a beta specific to the batch element,
            i.e. to the product of the distributions (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
    """
    p = p.detach()  # call sample_combined() if you need derivatives.
    N = p.shape[-2]
    M = p.shape[-1]
    assert K & (K-1) == 0
    assert K > 0 and K < M

    pshape = p.shape
    p = p.reshape(-1, N, M)
    B = p.shape[0]
    rand = torch.randint(2**63 - 1, (B,), device=p.device, dtype=torch.int64)
    (indexes, indexes_combined, weights) = _sample_combined_forward_dispatcher(p, rand, K, input_is_log)
    star = p.shape[:-2]
    indexes = indexes.reshape(*star, K, N)
    indexes_combined = indexes_combined.reshape(*star, K)
    weights = weights.reshape(*star, K)
    return (indexes, indexes_combined, weights)



def _test_sample_combined_forward():
    B = 2
    N = 2
    M = 16
    K = 4
    l = 3.0 * torch.randn(B, N, M)
    l = l.log_softmax(dim=-1)
    print("p = ", l.exp())
    (indexes, indexes_combined, weights) = sample_combined_forward(l, K, True)
    print("indexes = ", indexes)
    print("indexes_combined = ", indexes_combined)
    print("weights = ", weights)
    assert torch.all((weights.sum(dim=-1) - 1.0).abs() < 0.1)

def _test_sample_combined_forward_average():
    B = 2
    N = 2
    M = 32
    K = 8
    l = 3.0 * torch.randn(B, N, M)
    l = l.log_softmax(dim=-1)
    print("p = ", l.exp())


    avg_p = torch.zeros_like(l)
    num_samples = 900
    for _ in range(num_samples):
        # weights: (B, K)
        # indexes: (B, K, N)
        indexes, indexes_combined, weights = sample_combined_forward(l, K, True)

        sampled_p = torch.zeros_like(l)
        weights_expanded = weights.unsqueeze(-2).expand(*weights.shape[:-1], N, K)
        sampled_p.scatter_add_(dim=-1, index=indexes.transpose(-2, -1),
                               src=weights_expanded)
        avg_p += sampled_p * (1.0/num_samples)
    print("sample_combined_mean(): N = ", N, ", p = ", l.exp())
    print("avg_p = ", avg_p)
    print("max-diff = ", (l.exp() - avg_p).abs().max())



if __name__ == '__main__':
    _test_sample_combined_forward()
    _test_sample_combined_forward_average()
