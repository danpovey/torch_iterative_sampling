import os

import random # for testing and diagnostics..
import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional, Union
from torch.utils.cpp_extension import load
from torch.cuda.amp import custom_fwd, custom_bwd

VERBOSE = True

def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_scheduled_sampling_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_scheduled_sampling_cpu')
    torch_scheduled_sampling_cpu = load(
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
        print('Falling back to JIT compiling torch_scheduled_sampling_cuda')
    torch_scheduled_sampling_cuda = None
    if torch.cuda.is_available():
        torch_scheduled_sampling_cuda = load(
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
        if torch_scheduled_sampling_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_scheduled_sampling_cuda.sample_combined_forward_cuda(
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
    star = pshape[:-2]
    indexes = indexes.reshape(*star, K, N)
    indexes_combined = indexes_combined.reshape(*star, K)
    weights = weights.reshape(*star, K)
    return (indexes, indexes_combined, weights)

def sample_combined_backward(p: Tensor, input_is_log: bool, indexes: Tensor,
                             weights: Tensor, weights_grad: Tensor) -> Tensor:
    """
    Backward for sample_combined(); see sample_combined_forward() for detailed docs on
    the forward pass.  Notice that we don't use Torch's inbuilt autograd for this;
    that would not give us the answer we want.

    View the output of the forward pass as a sparse vector q.  You can view the
    forward pass as implementing: q = z p, where z is a sparse vector whose
    *expected* value is [1,1,..].  Because the expected value of z does not change
    with p, we treat z as being independent of p, even though actually
    the detailed distribution of z does depend on p.  So the backprop in non-log
    space would just be:
          p_grad = z * output_grad
    where z is the sparse vector we multiplied by in the forward pass.  Since
    we can express z as just q / p, this becomes:
          p_grad = q / p * output_grad
    where q is the sparse output of the forward pass.  In log-space, this is just
    equivalent to log_p_grad = log_output_grad.
    In non-log space, division by p could lead to infinite output if p is zero;
    in the forward pass we smoothed p by adding 2**-(num_bits_per_sample), and
    if you work it out, the backprop rule correcting for this would just become
          p_grad = q / (p + 2**-(num_bits_per_sample) * output_grad

    Args:
         p: the probabilities as used in the forward pass, of shape (*, N, M)
  input_is_log: if False, p should be probabilities; if True, p should
         be normalized log-probs, e.g. the output of log_softmax.
      weights: the `weights` output of simple_combined_forward, of shape (*, K)
      indexes:  the `indexes` output of simple_combined_forward, of shape (*, K, N)
   weights_grad: the loss-function gradient w.r.t the output weights, of shape
               (*, K)
    """
    K = weights.shape[-1]
    N = indexes.shape[-1]

    log_p_grad = torch.zeros_like(p)  # (*, N, M)
    # log_weights_grad is derivative w.r.t. log(weights).
    log_weights_grad = weights_grad * weights
    # expanded_log_weights_grad: (*, N, K),
    # duplicate along the N dimension
    expanded_log_weights_grad = log_weights_grad.unsqueeze(-2).expand(*weights.shape[:-1],
                                                                      N, K)
    log_p_grad.scatter_add_(dim=-1, index=indexes.transpose(-2, -1), src=expanded_log_weights_grad)

    if not input_is_log:
        if p.dtype == torch.float16:
            raise ValueError("For float16 input you have to use log-space for input probabilities, "
                             "require input_is_log=True")
        num_bits_per_sample = _max_bits // N
        # 2**-num_bits_per_sample is very small, so don't worry about renormalizing p.
        # This is just to stop division by zero.
        p_smoothed = p + (2.0**-num_bits_per_sample)
        log_p_grad.divide_(p_smoothed)
        return log_p_grad
    return log_p_grad

class SampleCombinedFunction(torch.autograd.Function):
    # please see sample_combined() or sample_combined_forward() or
    # sample_combined_backward() for documentation
    @staticmethod
    @custom_fwd
    def forward(ctx, p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            indexes, combined_indexes, weights = sample_combined_forward(p, K, input_is_log)
        ctx.save_for_backward(p, indexes, weights)
        ctx.input_is_log = input_is_log
        return indexes, combined_indexes, weights

    @staticmethod
    @custom_bwd
    def backward(ctx, indexes_grad: Optional[Tensor], combined_indexes_grad: Optional[Tensor], weights_grad: Optional[Tensor]) -> Tuple[Tensor, None, None]:
        p, indexes, weights = ctx.saved_tensors
        p_grad = sample_combined_backward(p, ctx.input_is_log, indexes,
                                          weights, weights_grad)
        return p_grad, None, None


def sample_combined(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
    """
    Sample from a distribution that is the product of softmaxes.  We will sample
    K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
    for a computed beta.
    Args:
         p: A Tensor of shape (*, N, M): either normalized log-probs (if input_is_log==False),
             or normalized probabilities; normalized along the M axis.  M must be
             a power of 2, and N must be in [1,2,3,4].
             [We can later relax the requirement that M be a power of 2.  The assumption is now
              only used for convenience of sampling a random number coprime to M, i.e. with
              s = 1 + 2*(torch.randint(M//2, shape)), but in general we can replace the 2 in
              this formula by the product of distinct prime factors in M, e.g. for 768
              this would be 6 because the prime factors are 2 and 3.  This is not totally optimal
              as we end up only choosing a subset of the numbers coprime to M,
              but it will be fine for typical cases because no sane person chooses parameter sizes
              with large prime factors and anyway we only need to randomize the order a little
              bit to reduce the correlation between indexes to a reasonable level.]
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
    return SampleCombinedFunction.apply(p, K, input_is_log)


def _test_sample_combined_forward():
    for device in [torch.device('cpu'), torch.device('cuda')]:
        B = 2
        N = 2
        M = 16
        K = 4
        l = 3.0 * torch.randn(B, N, M, device=device)
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

def _test_sample_combined_mean():
    for N in [2, 3, 1]:
        print("N = ", N)
        K = 4
        M = 8

        p = (5.0 * torch.randn(1, 2, N, M)).log_softmax(dim=-1)

        avg_p = torch.zeros_like(p)
        num_samples = 1000
        for _ in range(num_samples):
            # weights: (1, 2, K)
            # indexes: (1, 2, K, N)
            indexes, combined_indexes, weights = sample_combined_forward(p, K, True)

            sampled_p = torch.zeros_like(p)
            weights_expanded = weights.unsqueeze(-2).expand(*weights.shape[:-1], N, K)
            sampled_p.scatter_add_(dim=-1, index=indexes.transpose(-2, -1),
                                   src=weights_expanded)
            avg_p += sampled_p * (1.0/num_samples)
        print("sample_combined_mean(): N = ", N, ", p = ", p.exp())
        print("avg_p = ", avg_p)

        print("max err = ", (p.exp()-avg_p).abs().max())



if __name__ == '__main__':
    _test_sample_combined_forward_average()
    _test_sample_combined_mean()
    _test_sample_combined_forward()
