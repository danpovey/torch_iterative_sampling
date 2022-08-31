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
    import torch_balanced_sampling_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_balanced_sampling_cpu')
    if False:
        torch_balanced_sampling_cpu = load(
        name='sample_combined_forward_cpu',
        sources=[
            _resolve('sampling_cpu.cpp'),
        ],
        verbose=VERBOSE,
    )


try:
    import torch_balanced_sampling_cuda
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_balanced_sampling_cuda')
    torch_balanced_sampling_cuda = None
    if False and torch.cuda.is_available(): # TEMP
        torch_balanced_sampling_cuda = load(
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
        if torch_balanced_sampling_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_balanced_sampling_cuda.sample_combined_forward_cuda(
            probs, rand, K, input_is_log)
    else:
        return torch_balanced_sampling_cpu.sample_combined_forward_cpu(
            probs, rand, K, input_is_log)

_max_bits = 54  # used in sample_combined_forward and sample_combined_backward,
                # see comment in sample_combined_forward.

def sample_combined_forward(p: Tensor, K: int, input_is_log: bool,
                            rand: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample from a distribution that is the product of softmaxes.  We will sample
    K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
    for a computed beta.
    Args:
         p: A Tensor of shape (*, N, M): either normalized log-probs (if input_is_log==True),
             or normalized probabilities; normalized along the M axis.  M must be
             a power of 2, and N must be in [1,2,3,4].
         K: An integer, the number of samples required, with 0 < K < N
   input_is_log:  True if p represents normalized log-probs, False if it represents
             probabilities.
       rand: of shape (*,), containing random numbers in 0..2**63-1, this is provided
           for testing purposes, you will not normally need to pass this in.

    Returns: (indexes, combined_indexes, weights, epsilon)
       indexes: of shape (*, K, N), for each of K samples from a distribution it contains
            an N-tuple of indexes saying which combination of indexes from the
            component distributions were sampled.
      combined_indexes: of shape (*, K),  contains the same information as `indexes` but
            in a different format, specifically:
               `combined_indexes[...,k] = sum_n indexes[...,k,n] * M**n`
       weights: of shape (*, K), gives the weight associated with each sample,
            which will equal max(p, beta) for a beta specific to the batch element,
            i.e. to the product of the distributions (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
       epsilon: of shape (,), i.e. a scalar, contains a small value that
           can be used to prevent division by zero in the backward pass.
    """
    p = p.detach()  # call sample_combined() if you need derivatives.
    N = p.shape[-2]
    M = p.shape[-1]
    assert K & (K-1) == 0
    assert K > 0 and K < M

    pshape = p.shape
    p = p.reshape(-1, N, M)
    B = p.shape[0]
    if rand is None:
        rand = torch.randint(low=0, high=2**63 - 1, size=(B,), device=p.device, dtype=torch.int64)
    else:
        rand = rand.flatten()
    (indexes, indexes_combined, weights, epsilon) = _sample_combined_forward_dispatcher(p, rand, K, input_is_log)
    star = pshape[:-2]
    indexes = indexes.reshape(*star, K, N)
    indexes_combined = indexes_combined.reshape(*star, K)
    weights = weights.reshape(*star, K)
    return (indexes, indexes_combined, weights, epsilon)

def sample_combined_backward(p: Tensor, input_is_log: bool, indexes: Tensor,
                             weights: Tensor, epsilon: Tensor, weights_grad: Tensor) -> Tensor:
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
      epsilon: of shape (,), i.e. a scalar, contains a small value that can
           be used to prevent division by zero in backprop if input_is_log is False.
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
        p_smoothed = p + epsilon
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
            indexes, combined_indexes, weights, epsilon = sample_combined_forward(p, K, input_is_log)
        ctx.save_for_backward(p, indexes, weights, epsilon)
        ctx.input_is_log = input_is_log
        return indexes, combined_indexes, weights

    @staticmethod
    @custom_bwd
    def backward(ctx, indexes_grad: Optional[Tensor], combined_indexes_grad: Optional[Tensor], weights_grad: Optional[Tensor]) -> Tuple[Tensor, None, None]:
        p, indexes, weights, epsilon = ctx.saved_tensors
        p_grad = sample_combined_backward(p, ctx.input_is_log, indexes,
                                          weights, epsilon, weights_grad)
        return p_grad, None, None




def compute_target_marginals(p: Tensor, K: int) -> Tensor:
    """
    This function, which is non-differentiable,
    returns the optimal marginal probabilities (in the sense of
    reducing variance of the sampled output) of the classes, if we are
    to sample exactly K distinct classes from a distribution over M > K
    items.  If the probabilities for one of the distributions are p_i,
    with 0 <= i < M, this will be min(1, beta p), where beta is chosen so that
    the sum of the resulting marginals is exactly K.

    Args:
       p: the input probabilities, of shape (*, M);
          these do not have to be properly normalized to sum to 1.
       K: the number of items to sample, 0 < K < M.
    Returns:
       Returns a Tensor `ans` of the same shape (*, M) and the same dtype and device
       as p.  If input_is_log == False, it will be an expression of the form:
            ans == (p * beta).clamp_(max=1.0)
       where beta is a tensor of shape (*, 1) with values >= K, that is not returned; and will satisfy:
             ans.sum(dim=1) == K
    """
    M = p.shape[-1]
    p_sorted, _ = torch.sort(p, dim=-1)

    p_sorted_cumsum = p_sorted.cumsum(dim=-1)

    # If want to compute, for each item in the batch, the number of items 0 <= N < K
    # such that N items of (p * beta) are greater than 1, with beta computed
    # such that the marginals sum to exactly K, i.e. that
    #     (p * beta).clamp_(max=1.0) == K,
    # OK, if (p * beta).clamp_(max=1.0) == K, and exactly 0 <= N < K items of (p * beta)
    # are greater than 1, then...
    #    N + beta_N * (p_sorted_cumsum[M - N - 1]) == K,
    # i.e. we add N for the items exactly equal to 1, plus the sum of the smallest (M - N)
    # items,
    # where beta_N is "the computed beta, assuming exactly N items of (p * beta) > 1."
    # So
    #    beta_N = (K - N) / (p_sorted_cumsum[M - N - 1]).        (eqn:1)
    # Now, beta_N is valid [in the sense that exactly N items of (p * beta_N)
    # are >1.0], if:
    #
    #   N == 0 or p_sorted[M - N] * beta_N > 1.0   # this condition will fail if N is too large.
    #       p_sorted[M - 1 - N] * beta_N <= 1.0    # this condition will fail if N is too small.
    #
    # ... because I am concerned about ties, I think we can compute just one condition.
    #  if we compute just the 2nd condition, i.e.:
    #       p_sorted[M - 1 - N] * beta_N <= 1.0            (eqn:2)
    # ... then the correct value of N equals the number of items in N=0,1,..K-1 for which this
    # fails (is false).
    # So if we compute:
    #   N = sum(p_sorted[M - 1 - N] * beta_N > 1.0),  [sum of boolean array]
    # the number of nonzero items should equal the correct value of N (with 0 <= N < K).
    # .. because, mathematically, this should always be false
    #
    # However, because in practice we are computing things in reverse order, the index
    # that we really want is not N but K-1-N, which we can get by summing the number
    # of items in N=0,1,2,...K-2 for which (eqn:2) succeeds, i.e.:
    #
    #  K1_minus_N = sum(p_sorted[M - 1 - N] * beta_N <= 1.0) [sum of boolean array]            (eqn:3)
    #
    # .. we exclude the value for N==K-1 because, mathematically, thsi should always be
    # true.

    # ... only including the elements for N < K,this should be the correct value of K-1-N.
    # The element of the sum for N == K should always be true.

    M = p.shape[-1]
    p_sorted_cumsum_part = p_sorted_cumsum[..., M-K:]  # shape: (..., K+1)
    p_sorted_part = p_sorted[..., M-K:]

    # the last dim p_sorted_cumsum_part corresponds to N==[K, K-1, K-2,..., 1, 0],
    # which is the number of classes excluded from the cumsum, so the value of
    # K-N along this array would equal [1, 2, ..., K].
    # torch.arange(K+1)
    K_minus_N = torch.arange(1, K+1, device=p.device, dtype=p.dtype)

    beta_N = K_minus_N / p_sorted_cumsum_part   # (eqn:1) above

    condition = (p_sorted_part * beta_N <= 1.0)
    K_minus_N_chosen = torch.sum(condition[...,1:].to(torch.int64), dim=-1, keepdim=True)  # (eqn:3)

    #print("K_minus_N_chosen = ", K_minus_N_chosen)
    #print("beta = ", beta_N)
    beta = torch.gather(beta_N, dim=-1, index=K_minus_N_chosen)

    return (p * beta).clamp_(max=1.0)


def sample_from_target_marginals(target_marginals: Tensor,
                                 K: int) -> Tensor:
    """
    Does systematic sampling to produce a sample of exactly K items from each
    distribution.
    Args:
        target_marginals: a Tensor of shape (num_frames, num_classes)
             such that 0 <= target_marginals <= 1 and
             target_marginals.sum(dim=1) == K.
            K: the number of samples to draw.
    Returns:
         a Tensor of integer (zero or one) counts, of type torch.int64
         and of shape (num_frames, num_classes), that
         will have exactly K values per frame set to 1 and the remaining
         values set to 0.
    """
    # Make sure that K is a power of 2.  This avoids certain potential errors
    # due to rounding effects, e.g. the possibility that (target_marginals * int_scale)
    # would get rounded up due to limited float precision, prior to conversion to
    # integer, leading to elements of int_marginals greater than int_scale, so potentially
    # duplicates of the same class.
    assert K & (K-1) == 0, K
    # note: for testing and validation of the code, I used a small power here, 7 instead of
    # 30.
    int_scale = (2**30) // K  # the scale on 1.0 in the marginals, when we integerize.
    int_marginals = (target_marginals * int_scale).to(torch.int32)
    int_marginals_cumsum = int_marginals.cumsum(dim=-1)
    # excsum means exclusive cumsum
    int_marginals_excsum = int_marginals_cumsum - int_marginals

    # int_marginals_tot will be very close to int_scale * K == 2**30.
    # OK, mathematically, we are going to choose a random integer 0 <= r < int_scale,
    # and then the classes we choose are the classes m such that, for
    # some integer k,
    #     int_marginals_excsum[f,m] <= r+k*int_scale < int_marginals_cumsum[f,m].
    # If the total (over classes) of int_marginals were exactly equal to
    # K*int_scale, this procedure would give us exactly K chosen classes, with
    # no class repeated twice.
    #
    # But because the total of int_marginals may be slightly different from K*int_scale,
    # we need to be a bit careful when picking r, to make sure that
    # exactly K classes are picked.  The criterion comes down to:
    #  r >= 0,
    #  r + K*int_scale >= tot
    #  r + (K-1)*int_scale < tot
    # where `tot` is the total of the integerized marginals, i.e. int_marginals_cumsum[f,-1].
    # This means:
    #    max(0, tot - K*int_scale) <= r < tot - (K-1)*int_scale.
    # so we can use:
    # r = r_start + rand() % (1 + r_end - r_start), where
    # r_start = max(0, tot - K*int_scale)
    #   r_end = tot - (K-1)*int_scale.

    tot = int_marginals_cumsum[:,-1].unsqueeze(-1)
    r_start = (tot - (K * int_scale)).clamp_(min=0)
    r_end = tot - ((K-1) * int_scale)

    num_frames = target_marginals.shape[0]
    r = r_start + (torch.randint(low=0, high=2**63-1, size=(num_frames, 1),
                                 device=target_marginals.device, dtype=torch.int64) %
                   (r_end - r_start))

    # the "+ K * int_scale" is to avoid the discontinuity in how integer division behaves when the
    # numerator becomes negative.
    # the reason for the "-1" is as follows...
    # We want a discontinuity between, say,
    #     (r+n*K - cumsum)==-1, and (r+n*K - cumsum == 0)
    # but because we are subtracting r from cumsum, the sign is flipped, so we want
    # a discontinuity between (cumsum-r) == n*K and (cumsum-r) == n*K+1, so we have
    # to subtract one because it's between n*K-1 and n*K that operator "//" gives a
    # discontinuity.
    #
    # cur_remainder == r -> (int_scale - 1) -> divide by int_scale -> 0
    # cur_remainder == r+1 -> int_scale -> divide by int_scale -> 1.
    cum_remainder = int_marginals_cumsum + (int_scale - 1) - r
    exc_remainder = int_marginals_excsum + (int_scale - 1) - r

    is_this_class = ((exc_remainder // int_scale) < (cum_remainder // int_scale)).to(torch.int64)
    #sum = is_this_class.sum(dim=1)
    #print(f"K={K}, sum={sum}, r={r.flatten()}")
    #assert torch.all(is_this_class.sum(dim=1) == K)
    return is_this_class



def balance_target_marginals(log_p: Tensor,
                             padding_scale: Tensor,
                             K: int,
                             num_iters: int,
                             momentum: float = 0.0) -> Tensor:
    """
    Approximately balances the target marginals by means of
    a multiplicative factor on the probabilities p.  This is one iteration of an iterative update;
    you can call this in a loop. This update method is safe -- will not diverge --
    but not particularly exact or aggressive.  It is a bit like generalized
    iterative scaling (GIS).

    Args:
          log_p: the log-probabilities from which we'll be computing the target
            marginals to be balanced, of shape (num_frames, num_classes).
            It is not required to be
            normalized (i.e. sum to 1 over the M dimension)
            We want the computed target_marginals to be equal over the classes,
            within each block.
  padding_scale: a Tensor of shape (num_frames, 1), that is 1 for padding frames
            and 0 for non-padding frames.  padding_scale.sum() cannot be 0.
          K: the num-samples we'll be using to randomly approximate the distribution;
             an argument to compute_target_marginals.
      Returns:
           a modified log_p, which is log_p plus a correction factor for
           each class.  Will not be properly normalized to sum to 1 after exp.
    """

    num_non_padding_frames = padding_scale.sum()

    for i in range(num_iters):
        target_marginals = padding_scale * compute_target_marginals(log_p.softmax(dim=-1), K)
        M = log_p.shape[-1]
        # `target` is the average marginal for each class, if they have equal
        # probabilities within each block.
        target = num_non_padding_frames * K / M

        actual = target_marginals.sum(dim=0, keepdim=True).clamp_(min=1.0e-05)
        if i == 0 or momentum == 0.0:
            factor = (target / actual)
        else:
            new_factor = (target / actual)
            different_update_sign = ((factor > 1.0) != (new_factor > 1.0))
            factor.masked_fill_(different_update_sign, 1.0)
            factor = (factor ** momentum) * (target / actual)
        log_p = log_p + factor.log()
    return log_p



def balanced_sample(logprobs: Tensor,
                    K: int,
                    padding: Optional[Tensor] = None,
                    min_pad_items: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
    """
    logprobs: A tensor of shape (B, M) where B will be treated as a batch dimension,
            and M is the number of classes, containing the logprobs of the classes to
            in a weighted sum that we'll be approximating via sampling.
          K: Number of samples that we'll be approximating each weighted sum with.
             Must be >= 1 (but probably, in practice, >= 2).
             K must exactly divide M.
     padding: a Tensor of bool, of shape (B,) containing True in positions that correspond
             to padding frames.  The user asserts that the indexes chosen for these
             frame do not matter (i.e. that they are not going to affect the final
             result of the computation.
 min_pad_items: an integer saying the minimum number of padding items that must be
             added to the B items in `logprobs` (items with elements set to True
             in `padding` count towards this total).  You can just leave this at 0;
             larger values will tend to make the approximation more exact, at the
             expense of extra computation.


    Returns (F, indexes_in, indexes_out, weights), where:

           F: (number of frames after padding): an integer >= the input batch size B,
              which may not exactly equal B
              because of padding to a multiple of M//K, plus possibly extra padding
              depending on the value of min_pad_items.

        indexes_in: a Tensor of torch.int64, of shape (M, samples_per_class),
where samples_per_class == (F*K) // M
              containing indexes in [0..F-1]; you can use (indexes_in % B) for indexing
              if you want, in your computation, as the padding values do not matter.
              M is the number of classes; this is the leading tensor dimension for
              convenience of batched matrix multiply in whatever weighted computation
              you will be doing.
        weights: a Tensor of shape (M, samples_per_class), giving the weight corresponding to
              each item in the weighted sum (these should sum over the last dimension
              to a value quite close to 1).  The weights have the property that
              the *expected* values of the weights for a given index 0 <= b <= B
              equals logprobs.exp().
        indexes_out: a Tensor of shape (B, K), containing indexes in 0..M*samples_per_class - 1,
              which can be used to gather the weighted outputs of the computation,
              to be summed.  I.e. if the weighted output of the computation is a
              tensor of shape (M, N, K, *), you can reshape it to (M*N*K, *) and index
              it with this tensor to produce something of shape (B, K, *) that
              can be summed over the K dimension.
    """
    # Implementation notes:
    # (1) compute F >= B; pad logprobs to shape (F, M).

    (B, num_classes) = logprobs.shape

    if True:
        # This block computes the rounded-up and padded number of frames.
        assert num_classes % K == 0
        # the padded num-frames, F, must be an exact multiple of n so that the
        # average count of each class can be a whole number.
        N = num_classes // K

        # work out F >= B, the padded and rounded number of frames.
        F = N * (B + min_pad_items + (N - 1)) // N

    # pad `p` to F frames.  doesn't need to be normalized.  Anyway, we'll eventually be ignoring
    # the values of `probs` on padding frames.
    log_p = torch.cat(log_p, torch.zeros(F - B, num_classes,
                                         device=probs.device,
                                         dtype=probs.dtype))


    if True:
        # This block computes "padding_scale", which is a tensor of shape (F, 1) containing
        # 1 for non-padding frames and 0 for padding frames.
        if padding is None:
            padding_scale = torch.ones(B, 1, device=probs.device, dtype=torch.dtype)
        else:
            padding_scale = torch.logical_not(padding).to(probs.dtype).unsqueeze(-1)
        padding_scale = torch.cat((padding_scale,
                                   torch.zeros(F-B, 1, device=probs.device, dtype=probs.dtype)),
                                  dim=0)

    log_p = balance_target_marginals(log_p, padding_scale, K, num_iters=2)
    p = log_p.softmax(dim=-1)  # 'adjusted' probability.

    target_marginals = compute_target_marginals(p, K)





    # (2) approximately rebalance logprobs so target marginals are approx. equal; compute
    #     target marginals.




    # (2) do scheduled sampling to pick K symbols for each item 0 <= f < F.
    #      format the result as a matrix of counts (zero-one) of shape (F, M)
    #      Zero all the padding rows.
    # (3) do exc-cumsum over F and M axes to get counts_Fsum; counts_Msum.
    #
    #  (3.1) from last row of counts_Fsum, get counts for each class.
    #     use this to construct ragged array indexed [class][list-of-positions]
    #     with the num-counts per class as the lengths of the rows.
    #
    # (3.2) with the ragged array in 3.1, for each class, compute "indexes" which is a list of
    #      c indexes showing where that class is present.  can also store the k indexes
    #      here?  i.e. f*K + k?   can do this in a kernel?  i.e.
    #          if there's a 1 here, compute the location in the ragged
    #          array and set the value of f*K + k.
    #  (3.3)
    #      Compute "excess count" for each class, which is: (count - required_count).relu()
    #      We need to remove this many counts for those classes.
    #
    #      Using a copy of the ragged array in (3.2):
    #          [i] randomize the order using torch randn() and sort;
    #              ... can use the max size of the array (need to get item()) and
    #               just use torch sort.
    #          [ii] select just the number of these that we need to set to zero.
    #          [iii] set the appropriate counts in `counts` to zero.
    #
    # (3.4)
    #     [i] Compute "missing count" for each class, which is the number of
    #         items of this class that we need to make class-counts even;
    #         make ragged array from this.
    #     [ii] compute indexes (f*K + k) of all the item locations that we
    #         need to fill in randomly.
    #     [iii] randomize the order of the array from [ii]
    #    collate the randomized locations in [iii] and the class indexes from
    #    row-ids of ragged array in [i], and use them to atomically increment
    #    elements of the `class_count` (F,M) array.
    #
    #
    # (3.5) do, again, 2 cumulative sums on the class_count array,
    #  giving counts_Fsum, counts_Msum.  Use this info to produce
    #  an array of shape
    #       [num_classes, samples_per_class] containing indexes f.
    #  this is indexes_in, which we return.
    #
    # at the same time we compute indexes_out, which is of shape (B, K)
    # and containing indexes in 0..(num_classes*samples_per_class) - 1.
    #
    # as for `weights`: we can write them at this point.  first need to do
    # a computation for the expected weights.
    #
    #
    #(f*K + k) <-- do we need this, or
    #  do we just need f?
    #  saying which frame and which sample-index k it was.
    #
    #
    #
    #
    #
    #
    #
    #



    #
    #

    # (3) compute matrix of counts (zero-one for now) of shape (C,


    # (3)

    # (2) randomize order of logprobs using randperm.
    # (3) reshape randomized logprobs to shape (num_blocks, block_size, M)
    #     where num_blocks == C // block_size.  All further processing will bre
    #     on these blocks.
    # (4) within each block, for a couple of iterations, balance the target
    #     marginals so they are approximately equal (this will be done by
    #     adding an offset to logprobs).  we get approximately balanced
    #     target marginals from this.
    # (5) within each block, sample the indexes.. this is where
    #     most of the complexity is.
    #     The output of step (5) would be:
    #
    #     A tensor of shape (num_blocks, K, M, block_size//M), giving,
    #     for each block, each iteration of sampling and for each class,
    #     the indexes in [0..block_size-1] saying which item in the block
    #     has this index on this iteration of sampling.
    #
    #
    #        (i) a tensor of shape (num_blocks, block_size, K),
    #             containing indexes in 0..M-1.
    #
    # (6) compute the output weights as input-prob / target_marginal_prob.
    # ... format the output indexes ... return them.
    pass



def sample_balanced(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
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
            i.e. to the input product distribution (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
    """
    return SampleCombinedFunction.apply(p, K, input_is_log)


def _test_sample_combined_forward_compare():
    for _ in range(1000):
        B = random.randint(1, 1000)
        N = random.randint(1, 4)
        M = random.randint(16, 1024)
        T = random.randint(1, 2)
        while True:
            K = 2 ** (random.randint(0, 4))
            if K < M:
                break
        l = 6.0 * torch.randn(B, T, N, M)
        l = l.softmax(dim=-1)
        rand = torch.randint(2**63 - 1, (B, T), device=l.device, dtype=torch.int64)
        (indexes, indexes_combined, weights, epsilon) = sample_combined_forward(l, K, False, rand)
        print(f"B={B}, T={T}, N={N}, M={M}, K={K}")

        l_cuda = l.to(device='cuda')
        rand_cuda = rand.to(l_cuda.device)
        try:
            (indexes_cuda, indexes_combined_cuda, weights_cuda, epsilon_cuda) = sample_combined_forward(l_cuda, K, False, rand_cuda)
            assert torch.all((weights - weights_cuda.to('cpu')).abs() < 0.01)
            assert torch.all(indexes == indexes_cuda.to('cpu'))
            assert torch.all(indexes_combined == indexes_combined_cuda.to('cpu'))
            assert epsilon == epsilon_cuda.to('cpu')
        except:
            print("indexes = ", indexes)
            print("indexes_combined = ", indexes_combined)
            print("weights = ", weights)
            assert torch.all((weights.sum(dim=-1) - 1.0).abs() < 0.1)
            print("indexes_cuda = ", indexes_cuda)
            print("indexes_combined_cuda = ", indexes_combined_cuda)
            print("weights_cuda = ", weights_cuda)
            print("weights diff = ", weights - weights_cuda.to('cpu'))


def _test_sample_combined_forward_compare0():
    if True:
        B = 1
        N = 2
        M = 667
        K = 32
        l = 6.0 * torch.randn(B, N, M)
        l = l.softmax(dim=-1)
        rand = torch.randint(2**63 - 1, (B,), device=l.device, dtype=torch.int64)
        (indexes, indexes_combined, weights, epsilon) = sample_combined_forward(l, K, False, rand)
        print(f"B={B}, N={N}, M={M}, K={K}")

        l_cuda = l.to(device='cuda')
        rand_cuda = rand.to(l_cuda.device)

        (indexes_cuda, indexes_combined_cuda, weights_cuda,
         epsilon_cuda) = sample_combined_forward(l_cuda, K, False, rand_cuda)
        print("indexes = ", indexes)
        print("indexes_combined = ", indexes_combined)
        print("weights = ", weights)
        print("epsilon = ", epsilon)
        assert torch.all((weights.sum(dim=-1) - 1.0).abs() < 0.1)
        print("indexes_cuda = ", indexes_cuda)
        print("indexes_combined_cuda = ", indexes_combined_cuda)
        print("weights_cuda = ", weights_cuda)
        print("weights diff = ", weights - weights_cuda.to('cpu'))
        assert torch.all((weights - weights_cuda.to('cpu')).abs() < 0.01)
        assert torch.all(indexes == indexes_cuda.to('cpu'))
        assert torch.all(indexes_combined == indexes_combined_cuda.to('cpu'))
        assert epsilon == epsilon_cuda.to('cpu')

def _test_sample_combined_forward():
    for device in [torch.device('cpu'), torch.device('cuda')]:
        B = 2
        N = 2
        M = 16
        K = 4
        l = 3.0 * torch.randn(B, N, M, device=device)
        l = l.log_softmax(dim=-1)
        print("p = ", l.exp())
        (indexes, indexes_combined, weights, epsilon) = sample_combined_forward(l, K, True)
        print("indexes = ", indexes)
        print("indexes_combined = ", indexes_combined)
        print("weights = ", weights)
        print("epsilon = ", epsilon)
        assert torch.all((weights.sum(dim=-1) - 1.0).abs() < 0.1)


def _test_sample_combined_forward_average():
    B = 1
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
        indexes, indexes_combined, weights, epsilon = sample_combined_forward(l, K, True)
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
            indexes, combined_indexes, weights, epsilon = sample_combined_forward(p, K, True)
            sampled_p = torch.zeros_like(p)
            weights_expanded = weights.unsqueeze(-2).expand(*weights.shape[:-1], N, K)
            sampled_p.scatter_add_(dim=-1, index=indexes.transpose(-2, -1),
                                   src=weights_expanded)
            avg_p += sampled_p * (1.0/num_samples)
        print("sample_combined_mean(): N = ", N, ", p = ", p.exp())
        print("avg_p = ", avg_p)

        print("max err = ", (p.exp()-avg_p).abs().max())

def _get_num_iters(func) -> int:
    # finds how many times we can call func() in 0.1 second.
    import time
    max_time = 0.2
    num_iters = 0
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < max_time:
        func()
        num_iters = num_iters + 1
    return num_iters

def _test_sample_combined_speed():
    B = 512
    M = 256
    N = 2
    K = 16
    l = 3.0 * torch.randn(B, N, M)
    l = l.log_softmax(dim=-1)  # normalize.

    sample_combined_forward_test = lambda : sample_combined_forward(l, K, True)

    num_iters = _get_num_iters(sample_combined_forward_test)
    print(f"Num-iters in 0.2 sec on CPU with B={B}, M={M}, N={N}, K={K} {num_iters}")
    l = l.to('cuda')
    num_iters = _get_num_iters(sample_combined_forward_test)
    print(f"Num-iters in 0.2sec on GPU with B={B}, M={M}, N={N}, K={K} {num_iters}")

    import sampling_ref
    l = l.to('cpu')
    sample_combined_forward_ref_test = lambda : sampling_ref.sample_combined(l, K, True)
    num_iters = _get_num_iters(sample_combined_forward_ref_test)
    print(f"[sampling_ref.py]: Num-iters in 0.2sec on CPU with B={B}, M={M}, N={N}, K={K} {num_iters}")
    l = l.to('cuda')
    num_iters = _get_num_iters(sample_combined_forward_ref_test)
    print(f"[sampling_ref.py]: Num-iters in 0.2sec on GPU with B={B}, M={M}, N={N}, K={K} {num_iters}")



def _test_sample_from_target_marginals():
    num_frames = 100
    num_classes = 20
    probs = torch.randn(num_frames, num_classes).softmax(dim=1)
    K = 4
    target_marginals = compute_target_marginals(probs, K)

    samples = sample_from_target_marginals(target_marginals, K)

    assert torch.all(samples.sum(dim=1) == K)


if __name__ == '__main__':
    _test_sample_from_target_marginals()


    _test_sample_combined_forward_compare0()
    _test_sample_combined_speed()
    _test_sample_combined_forward_compare()
    _test_sample_combined_forward()
    _test_sample_combined_forward_average()
    _test_sample_combined_mean()
