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
    import torch_sampling_cpu
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
    import torch_sampling_cuda
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
        K: int, input_is_log: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    Dispatcher for sample_combined_forward
    """
    if cumsum.is_cuda:
        if torch_sampling_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_sampling_cuda.sample_combined_cuda_forward(
            cumsum, rand, seq_len)
    else:
        return torch_sampling_cpu.sample_combind_cpu_forward(
            cumsum, rand, seq_len)

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
    assert K & (K-1) == 0
    assert K > 0 and K < M

    pshape = p.shape
    N = p.shape[-2]
    M = p.shape[-1]
    p = p.reshape(-1, N, M)
    B = p.shape[0]
    rand = torch.randn(B, 2)
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
    l = 8.0 * torch.randn(B, N, M)
    l = l.log_softmax()
    print("p = ", p.exp())
    (indexes, indexes_combined, weights) = sample_combined_forward(p, K, True)
    print("indexes = ", indexes)
    print("indexes_combined = ", indexes_combined)
    print("weights = ", indexes_combined)


if __name__ == '__main__':
    _test_sample_combined_forward()
