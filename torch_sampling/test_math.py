#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Tuple

# This file is not part of the implementation; it exists to test that the
# algorithms described in NOTES.md are correct.



def compute_k_largest(X, K):
    """
    Returns, for each row of X, the values and indexes of the K largest elements,
    sorted from largest to smallest.
    Args:
      X: Tensor of any type, of shape (*, M) with M > K
      K: an integer with 0 < K <= M
    Returns (values, indexes), with
       values: K most positive values of each row of X, shape (*, K)
       indexes: indexes [on last axis] of K most positive values of each row of X,
            shape (*, K)
    """
    values, indexes = torch.sort(X, dim=-1, descending=True)
    return values[...,:K], indexes[...,:K]

def get_combined_cumsums(P,
                         P_cumsum_exclusive_scaled,
                         combined_indexes):
    """
    This is a function called while sampling from a distribution that's a product of
    N categorical distributions each of size M.

    Args:
       P: Tensor of int64 of shape (*, N, M), containing the individual integerized
          probabilities of classes.
      P_cumsum_exclusive_scaled: scaled exclusive-sum version of P_cumsum which is cumulative
           sum of P along M dimension, equal to
              (P_cumsum - P) * prod_prev_totals
          where prod_prev_totals is the product of the largest, final elements of P_cumsum
           over previous n indexes)
        combined_indexes: A tensor of int64 of shape (*, K, N), containing the top-K combinations
          of indexes in {0,1,..,M-1} that have the most probability mass, from greatest to least.
          We are interested in the (exclusive) cumulative sum at these points, i.e.  for each index
          in `combined_indexes` we are interested in the sum of all prior items.

      Returns:
          Returns a Tensor of int64 of shape (*, K), returning the cumulative sum of all
          combinations of indexes preceding each of the ones in 'combined_indexes'.

          We assign probability mass to combinations of indexes over the N axes, by
          multipliying these integerized probabilities, and we're interested in the cumulative
          sum of these products, assuming that index 0 varies fastest.  So the (inclusive) cumsum of the
          combination with indexes m0, m1, m2, would have a value given by:
              P_cumsum[..., 0, m0] + P_cumsum[..., 1, m1] * sum0 + P_cumsum[..., 2, m2] * sum0 * sum1
          where sum0 is the total sum from cumsum0 (its last element), and so on.
    """
    M = P.shape[-1]
    N = P.shape[-2]
    K = combined_indexes.shape[-2]
    assert combined_indexes.shape[-1] == N
    assert combined_indexes.shape[:-2] == P.shape[:-2]

    # ans: shape (*, K)
    ans = torch.zeros(*combined_indexes.shape[:-1], dtype=P.dtype, device=P.device)

    # P_cumsum_selected_scaled, of shape (*, N, K), contains the individual looked-up
    # exclusive-cumulative-sum values, i.e. the cumulative sum within the
    # individual softmax/distribution, of all preceding items;
    # these are pre-scaled by the product of total sum [P_sum] over previous
    # n indexes.
    P_cumsum_selected_scaled = P_cumsum_exclusive_scaled.gather(dim=-1, index=combined_indexes.transpose(-2, -1))

    # P_selected, of shape (*, N, K) contains the individual probability values
    # [corresponding to the indexes we want for the cumulative sum]
    P_selected = P.gather(dim=-1, index=combined_indexes.transpose(-2, -1))

    P_selected_cumprod = torch.cumprod(P_selected, dim=-2)
    # P_selected_laterprod, of shape (*, N, K), contains the sum of
    # P values for *later* n.
    P_selected_laterprod = P_selected_cumprod[...,N-1:N,:] // P_selected_cumprod
    print("P_selected_laterprod = ", P_selected_laterprod)


    # answer is sum over the N dimension, multipliying the
    # indexes for n>0 by P_prev_sum_product, i.e. the product over previous
    # sums.  [Earlier indexes are considered to vary fastest, this was easiest
    # to implement.]
    # Shape: (*, K)
    ans = (P_cumsum_selected_scaled * P_selected_laterprod).sum(dim=-2)
    print("Ans = ", ans)
    return ans


def compute_products(values, indexes):
    """
    This is intended to be called on the outputs of compute_k_largest().  It computes the
    products of different combinations of `values`, as follows:

     values: Tensor of shape (*, N, K)
    indexes: Tensor of shape (*, N, K)

    The K refers to the K-best, e.g. K=4, which will have been computed by
    compute_k_largest.  `values` contains the K largest elements per row, of a source
    tensor.  We are computing all products of these, over the N axis.

   Returns:  (values, indexes), where:
        prod_values: Tensor of shape (*, K**N)  containing the products of elements of `values`,
                    treating the dimensions in `*` as batch dimensions and taking products
                    along the N axis.
        prod_indexes: Tensor of shape (*, K**N, N)  containing the indexes of the original
                    elements that we took products of.
    """
    assert values.shape == indexes.shape
    K = values.shape[-1]
    N = values.shape[-2]

    # assume (*) == (B,) and N==3 for example shapes.
    # e.g. (B, 1, 1, 1)
    unit_shape = list(values.shape[:-2]) + ([1] * N)
    # e.g. (B, K, K, K)
    full_shape = list(values.shape[:-2]) + ([K] * N)
    # e.g. (B, K, K, K, N)
    indexes_shape = list(values.shape[:-2]) + ([K] * N) + [N]

    prod_values = 1
    prod_indexes = torch.empty(*indexes_shape, dtype=indexes.dtype,
                               device=indexes.device)


    for n in range(N):
        shape = list(unit_shape)  # copy it
        shape[-N + n] = K   # e.g. if n==1, shape might be (B, K, 1, 1)
        this_values = values.select(dim=-2, index=n).reshape(shape)
        this_src_indexes = indexes.select(dim=-2, index=n).reshape(shape)
        this_dest_indexes = prod_indexes.select(dim=-1, index=n) # e.g. (B, K, K, K)

        this_dest_indexes[:] = this_src_indexes # will broadcast
        prod_values = prod_values * this_values # will broadcast


    values_shape = list(values.shape[:-2]) + [K**N]
    indexes_shape = values_shape + [N]
    return prod_values.reshape(values_shape), prod_indexes.reshape(indexes_shape)



def compute_beta(P, K):
    """
    See: ComputeBeta function [practical version] in NOTES.md.
    Args:
        P: a tensor of shape (*, M), in practice containing integers in {1,2,..2**31+1},
           but can be any integers >0 as far as this function is concerned, provided the
           cumsum does not overflow.
        K: an integer 0 < K < M
    Returns a tensor of integers B of shape (*, 1) such that:
        sum(min(P, B)) == K*B
    [It will subtract a number in {0,1,..K-1} from one element of each row of P
    to make this sum exact.]
    """
    M = P.shape[-1]
    R, R_indexes = torch.sort(P, dim=-1)  # (*, M)
    Q = torch.cumsum(R, dim=-1)
    # Reference pseudocode was:
    #for k in 0,1,...K-1, in any order:
    #  # B_k is the value of B if k indexes take the l.h.s. of the "min" expression in min(B, P)
    #  B_k = (Q[M-1-i]  + K - k - 1) / (K - k)   # the "+ K - k - 1" is to ensure we round up
    #  if R[M-1-k] >= B_k and P[I-2-k] <= B_k:
    #     return B_k

    temp = torch.arange(K+1, dtype=R.dtype, device=R.device)
    # Kk, of shape (K,), contains [1, 2, ..., K], representing K-k for k = [K-1, K-2, ..., 0]
    Kk = temp[1:K+1]
    # Kk1 of shape (K,), contains [0, 1, ..., K-1], representing K-k-1 for k = [K-1, K-2, ..., 0]
    Kk1 = temp[0:K]

    Q_part = Q[...,M-K:M]   # represents: Q[...,M-1-k] for k = K-1,K-2,...,1,0

    B_k = Q_part // Kk  # shape (*, K)
    remainder_k = Q_part - (B_k * Kk)   # shape (*, K)

    large_int = (2**32 - 1)
    R_part1 = torch.cat((R[...,M-K+1:M], torch.full((*R.shape[:-1], 1), large_int)), dim=-1)
    R_part2 = R[...,M-K:M]

    # is_ok corresponds to: "(k==0 or R[M-k] > B_k) and R[M-1-k] <= B_k" in NOTES.md
    is_ok = (torch.logical_and(R_part1 > B_k, R_part2 <= B_k))  # shape: (*, K)

    assert torch.all(torch.max(is_ok, dim=-1)[0] == 1)
    B, indexes = torch.max(B_k * is_ok, dim=-1, keepdim=True)  # shape: (*, 1)
    remainder = torch.gather(remainder_k, dim=-1, index=indexes)

    remainder = torch.max(remainder_k * is_ok, dim=-1, keepdim=True)[0]  # shape: (*, 1)
    index = torch.max(R_indexes[...,M-K:M] * is_ok, dim=-1, keepdim=True)[0]
    P_index = torch.gather(R_indexes[...,M-K:M], dim=-1, index=indexes)
    P_val = torch.gather(P, dim=-1, index=P_index)
    P_val -= remainder
    P.scatter_(dim=-1, index=P_index, src=P_val)

    P_min = torch.minimum(P, B)

    P_min_sum = P_min.sum(dim=-1, keepdim=True)
    assert torch.all(K * B == P_min_sum)
    return B

def compute_beta_prods(Psum, Ptop):
    """
    Version of compute_beta() with a different interface, which is intended to work with
    products of softmaxes.  We are still assuming an integerized representation.

    Args:
      Psum: Tensor of shape (*,), treated as the batch dimension, which contains,
           as torch.int64, the total integerized probability mass taken as a product
           along all dimension, e.g. for a tensor of shape (*, N, K) containing integerized
           probabilities, we'd sum along the K dimension and take a product along the N
           dimension.
      Ptop: Tensor of shape (*, K), containing the probabilities for the top-K
           possible outputs (each possible output is a combination of N indexes in
           [0..M-1]).  The sum of Ptop must be less than Psum.

     Returns: (B, delta_P)
          beta: Tensor of shape (*) containing integers B satisfying:
                 sum(min(P, B)) == K*B                    (eqn:b1)
             ... where conceptually, P is a matrix of shape (*, K**N)
             that we do not materialize.
             What this condition amounts to in terms of args of this function,
             is that:
                 Psum + delta_P.sum(-1) = B*K
             [Caution: the exact equality in (eqn:b1) is only true
             once we subtract a small number in [0..K-1] from the next-largest
             element of P that is not >B, to correct for rounding error;
             this is accounted for in delta_P.
          delta_P: of shape (*, K), this contains the change, if any, that we have
             to make to the top-K elements of the distribution before sampling.
             Satisfies delta_P <= 0.  This combines two things: the
             differences (min(P[i], B) - P[i]); and the values in [-(K-1)..0]
             that we add to the largest item that's less than P to account
             for rounding effects.
    """
    K = Ptop.shape[-1]
    assert Psum.shape == Ptop.shape[:-1]

    Ptop_cum = torch.cumsum(Ptop, dim=-1)  # cumsum of Ptop, i.e. inclusive-sum.  Shape (*, K)

    # add zero first element per row, so Ptop_cum_shift[...,0] is all-zeros and
    # Ptop_cum_shift[...,1] contains the top-1.  The idea is that
    # Ptop_cum_shift[...,k] contains the sum of the top k items.
    Ptop_cum_shift = torch.cat((torch.zeros(*Ptop.shape[:-1], 1, dtype=Ptop.dtype,
                                       device=Ptop.device),
                                Ptop_cum[...,:K-1]), dim=-1)
    # S1[...,k] contains, for each batch element, the sum of all but the k largest
    # items.  It corresponds to s-1 in the math of NOTES.md, see "ComputeBeta function
    # [mathematical version].
    # Shape is (*, K)
    S1 = Psum.unsqueeze(-1) - Ptop_cum_shift

    temp = torch.arange(K, -1, -1)  # [K, K-1, ..., 0]
    # Kk, of shape (K,), contains [K, K-1, ..., 1], representing K-k for k = [0, 1, ..., K-1]
    Kk = temp[0:K]
    # Kk1 of shape (K,), contains [K-1, K-2, ..., 0], representing K-k-1 for k = [0, 1, ..., K-1]
    Kk1 = temp[1:K+1]

    # The following corresponds to:
    #    beta = (1 - s_k) / (K-k)
    # in NOTES.md.  This is integer division, we are rounding down.
    # B_k[...,k] is the beta value if k values are >= beta.
    B_k = S1 // Kk  # shape (*, K)
    remainder_k = S1 - (B_k * Kk)   # shape (*, K)
    print("B_k = ", B_k)
    print("remainder_k = ", remainder_k)

    large_int = (2**63 - 1)
    # Ptop_shifted is Ptop shifted right with a large value put first, i.e.
    # instead of [top1, top2, top3, top4] we have [inf, top1, top2, top3]
    Ptop_shifted = torch.cat((torch.full((*Ptop.shape[:-1], 1), large_int),
                              Ptop[...,:K-1]), dim=-1)

    print("Ptop= ", Ptop)
    print("Ptop_shifted= ", Ptop_shifted)

    # is_ok corresponds to: "(k==0 or R[M-k] > B_k) and R[M-1-k] <= B_k" in NOTES.md
    # It is true only for the "correct" k for each batch element, that corresponds
    # to the number of values greater than B_k.
    is_ok = (torch.logical_and(Ptop_shifted > B_k, Ptop <= B_k))  # shape: (*, K)

    # `indexes` are the values of k.
    B, indexes = torch.max(B_k * is_ok, dim=-1)  # shape: (*,)
    print("B = ", B)

    delta_P = (torch.minimum(Ptop, B.unsqueeze(-1)) - Ptop) - (remainder_k * is_ok)

    print("B_k = ", B_k)
    print("is_ok == ", is_ok)
    print("delta_P = ", delta_P)

    err = Psum + delta_P.sum(dim=-1) - B * K
    print("Err = ", err)
    assert torch.all(err == 0)
    assert torch.all(torch.sum(is_ok, dim=-1)[0] == 1)

    return B, delta_P

def compute_shifted_samples(combined_cumsums_mod: Tensor,
                            delta_P: Tensor,
                            samples: Tensor) -> Tensor:
    """
    Modified randomly sampled values by adding values to correct for "disallowed regions",
    i.e. parts of probability space that we skip because they correspond to a probability
    mass greater than beta [or because they correspond to small padding for roundoff].

      combined_cumsums_mod:  Modified cumulative sums which when they were "combined_cumsums"
                 can be thought of as points in probability space, but when they become
                 "modified" are reduced to account for "disallowed regions" that
                 we cannot sample.  The shape is (*, K) where `*` is the batch dimension
                 and K is the maximum number of "disallowed regions"
        delta_P: negative values that correspond to the amount of probability mass we
                 removed for each "disallowed region", i.e. the size of those
                 regions, as a negative number.  The shape is (*, K).
        samples: The samples that we have to modify by adding values corresponding to
                 the widths of the appropriate disallowed regions.  The shape is (*, K);
                 but this K is not the "same K"
     Returns: shifted_samples, which will be the same shape as `samples`, but possibly
                 with larger values, i.e. shifted_samples >= samples
    """
    samples = samples.unsqueeze(-1)
    combined_cumsums_mod = combined_cumsums_mod.unsqueeze(-2)
    delta_P = delta_P.unsqueeze(-2)

    # of shape (*, K, K), is_ge is True if sample k1 is >= combined_cumsum k2,
    # meaning we need to add the corresponding delta_p.
    is_ge = (samples >= combined_cumsums_mod)

    shifted_samples = samples - (is_ge * delta_P).sum(dim=-1, keepdim=True)
    shifted_samples = shifted_samples.squeeze(-1)
    return shifted_samples

def check_shifted_samples(combined_cumsums: Tensor,
                          delta_P: Tensor,
                          shifted_samples: Tensor,
                          prod_cumsum: Tensor):
    """
    Checks samples as modified by `compute_shifted_samples`: specifically, checks
    that they are not in the "disallowed regions" that we are supposed to skip over.

    combined_cumsums: Cumulative sums which can be thought of as the start of
                 "disallowed regions" in probability space.  Shape is (*, K)
             delta_P: the negative of the size of "disallowed regions".  Shape is (*, K)
     shifted_samples: The samples as modified by `compute_shifted_samples`.  None
                 of these should be within the "disallowed regions".  Shape is (*, K);
                 but note, this K does not have a correpondence with the K in the
                 other two args' shapes.
       prod_cumsum:  The product of sums/normalizers of the different softmaxes, of
                 shape (*,); this can be thought of as the total size of the probability
                 space, including "disallowed regions".    This is to check that
                 `shifted_samples` are less than this value.
    """
    assert torch.all(torch.logical_and(shifted_samples >= 0,
                                       shifted_samples < prod_cumsum.unsqueeze(-1)))

    shifted_samples = shifted_samples.unsqueeze(-1)
    combined_cumsums = combined_cumsums.unsqueeze(-2)
    delta_P = delta_P.unsqueeze(-2)

    disallowed_regions_start = combined_cumsums
    disallowed_regions_end = combined_cumsums - delta_P  # delta_p is <= 0.

    # in_disallowed_region is of shape (*, K, K)
    in_disallowed_region = torch.logical_and(shifted_samples >= disallowed_regions_start,
                                             shifted_samples < disallowed_regions_end)
    assert torch.all(torch.logical_not(in_disallowed_region))



def get_indexes_for_samples(P: Tensor,
                            P_cumsum: Tensor,
                            P_cumsum_exclusive: Tensor,
                            shifted_samples: Tensor) -> Tensor:
    """
    From K `shifted_samples` which are in the joint probability-space of N softmaxes
    of size M, figure out which sample indexes they correspond to.
    Args:
      P:  of shape (*, N, M), the original integerized probabilities we
          are interested in the products over [i.e. over the N dimension],
          e.g. N=2, M=128.
      P_cumsum:  Of shape (*, N, M), this is the (inclusive) cumulative sum of
          the original integerized probabilities P.  Conceptually, the entire
          probability space is over all possible products, over the N
          dimension, of different choices of m, arranged so that m-indexes for
          the earlier n indexes vary fastest, like [000,100,200,010,110,210, ... ].
      P_cumsum_exclusive:  Of shape (*, N, M), the exclusive-sum version of
          P_cumsum, equivalent to P_cumsum - P.
      shifted_samples:  Of shape (*, K), contains the random samples we want
          to find indexes for, "shifted" means we have skipped over "disallowed regions"
          corresponding to combinations of indexes that had too much probability mass.
          Will satisfy:
          0 <= shifted_samples < P_cumsum[...,-1].prod(dim=-1, keepdim=True)
    Returns:
        indexes: Of shape (*, K, N), the N-tuples of indexes in {0,1...M-1}
          corresponding to each of the K samples.
    """

    # P_sum_cumprod is the cumulative product of the total sum of the original
    # integerized probabilities P, of shape (*, M)
    P_sum_cumprod = torch.cumprod(P_cumsum[...,-1], dim=-1)
    print("P_sum_cumprod = ", P_sum_cumprod)
    M = P.shape[-1]
    N = P.shape[-2]

    ans_indexes_shape = list(shifted_samples.shape) + [N]  # (*, K, N)
    ans_indexes = torch.empty(*ans_indexes_shape, dtype=P.dtype,
                              device=P.device)

    cur_samples = shifted_samples  # (*, K)
    for n in range(N-1, -1, -1): # [N-1, N-2, ..., 0]
        this_samples = cur_samples  # (*, K)
        print("this_samples = ", this_samples)
        print("n=", n)
        if n > 0:
            # divide by the total product of probs *previous* indexes n,
            # so we can compare directly with P_cumsum.
            this_samples = this_samples // P_sum_cumprod[...,n-1:n]
        # right=True means we find
        # P_cumsum[...,index-1] <= this_samples[...,k] < P_cumsum[...,index],
        # which is what we want, as opposed to ... < ... <= (i.e. swap < and <=)
        idx = ans_indexes[...,n] = torch.searchsorted(P_cumsum[...,n,:], # (*, M)
                                                      this_samples, # (*, K)
                                                      right=True)
        print("idx = ", idx)
        this_P = torch.gather(P[...,n,:], dim=-1, index=idx)  # shape: (*, K)
        print("this_P = ", this_P)

        if n == 0:
            break

        # get cumsum corresponding to the indexes we just computed, we need
        # to subtract the start of the region corresponding to this index.
        # need exclusive-sum here..
        cur_cumsum = torch.gather(P_cumsum_exclusive[...,n,:], dim=-1, index=idx)
        print("cur_cumsum = ", cur_cumsum)
        # account for the product of previous dims' total sums...
        # TODO: multiply P_cumsum by P_sum_cumprod
        cur_cumsum *= P_sum_cumprod[...,n-1:n]
        print("cur_cumsum = ", cur_cumsum)
        # Get the remainder after subtracting the indexes we just worked out,
        # this will be used to get previous indexes, i.e. for lower n.
        remainder = cur_samples - cur_cumsum
        print("remainder = ", remainder)
        # Also divide by this_P, since all probability masses corresponding
        # to this index we just worked out will be scaled by this amount.
        remainder = remainder // this_P
        cur_samples = remainder

    print("ans_indexes = ", ans_indexes)
    return ans_indexes



def soft_sample_forward(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor]:
    """
    Forward function for soft sampling.
    Args:
      p: Tensor of shape (*, M)
      K: number of samples, 1 <= K < M
      input_is_log: if true, p must be probabilities in [0..1] that sum to one;
          if false, p must be logprobs (that sum to one after exp())
   Returns: (indexes, y), where:
        indexes: shape (*, K), a LongTensor containing elements in [0..M-1], distinct
           along the K axis
        y: shape (*, K), a Tensor containing values in [0..1], which sum to 1 along the
           K axis.

    Search for "def soft_sample" in NOTES.md to understand this.
    """
    if input_is_log:
        p = p.exp()
    M = p.shape[-1]
    two31 = 2 ** 31 # TEMP for testing, should be 2**31
    # to(dtype=this rounds toward 0, which is good enough
    P = (p*two31 + 1).to(dtype=torch.long)
    print("P = ", P)
    B = compute_beta(P, K)
    beta = B / two31
    print("B = ", B, ", beta = ", beta)
    t = torch.randint(M//2, p.shape[:-1] + (1,))  # shape: *, 1
    s = t * 2 + 1
    #s = torch.ones_like(t)

    # turns out we don't need inv_s.
    inv_s = (s ** (M//2 - 1)) % M
    assert torch.all((s * inv_s) % M == 1)  # if this fails, check that M is a power of 2

    # R = pseudo-random re-ordering of p.
    R = torch.minimum(torch.gather(P, dim=-1, index=(s * torch.arange(M)) % M),
                      B)
    # S = inclusive-sum of R
    S = torch.cumsum(R, dim=-1)

    # Let b be a random integer drawn uniformly from {0, 1, ..., B-1}.
    b = torch.randint((2**63 - 1), B.shape) % B

    print("R = ", R)
    print("S = ", S)
    print("b = ", b)

    S_prev = torch.cat((torch.zeros(*S.shape[:-1], 1), S[...,:-1]), dim=-1)

    k_prev = (S_prev + b) // B
    k_cur = (S + b) // B
    # if S_prev >= b and k_cur > k_prev:.. don't need S_prev >= b because rounded down.
    is_ok = (k_cur > k_prev)

    print("is_ok = ", is_ok, ", sum = ", is_ok.sum(dim=-1))
    print("k_cur = ", k_cur)
    # sort so the "false" goes first and the "true" goes in last K indexes.
    values, indices = is_ok.sort(dim=-1)
    i = indices[...,M-K:M]
    i = (i * s) % M  # Reverse the pseudo-random reordering
    print("beta = ", beta)
    y = torch.maximum(torch.gather(p, dim=-1, index=i), beta)
    print("i = ", i, ", y = ", y)
    assert torch.all(is_ok.sum(dim=-1) == K)
    assert torch.all((y.sum(dim=-1) - 1.0).abs() < 0.01)



class SoftSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p: Tensor, K: int, input_is_log: bool):
        """
        Forward function.
        Args:
          p: Tensor of shape (*, M)
          K: number of samples, 1 <= K < M
          input_is_log: if true, p must be probabilities in [0..1] that sum to one;
              if false, p must be logprobs (that sum to one after exp())
        """
        pass

def _test_compute_beta():
    # use a small M-- 8 here-- because it's more likely to
    # choose k != 0 in compute_beta(), giving a more complete test.
    a = torch.randint(low=1, high=65535, size=(9, 16))
    K = 4
    beta = compute_beta(a, K)  # it checks its own answer..
    print("beta = ", beta)


def _test_soft_sample():
    l = 2 * torch.randn(6, 64)
    p = torch.softmax(l, dim=-1)
    soft_sample_forward(p, K=4, input_is_log=False)

def _test_combined():
    N = 2
    K = 4
    M = 8

    P = ((5 * torch.randn(2, N, M)).softmax(dim=-1) * 16 + 1).to(dtype=torch.int64)

    print("P = ", P)
    values, indexes = compute_k_largest(P, K)
    print("largest values = ", values)
    print("largest indexes = ", indexes)
    prod_values, prod_indexes = compute_products(values, indexes)
    assert prod_values.shape == prod_indexes.shape[:-1]
    print("prod_values = ", prod_values)
    print("prod_indexes = ", prod_indexes)

    # combined_values, combined_indexes: (B, K) these are the top-K
    # most-probable combinations of (integerized_ probabilities and their
    # indexes, from best to worst.
    combined_values, combined_indexes = compute_k_largest(prod_values, K)

    combined_indexes_shape = list(combined_indexes.shape) + [N]
    # combined_indexes: (B, K, N)
    combined_indexes = torch.gather(prod_indexes, dim=-2,
                                    index=combined_indexes.unsqueeze(-1).expand(combined_indexes_shape))

    print("combined_values = ", combined_values)
    print("combined_indexes = ", combined_indexes)


    P_cumsum = torch.cumsum(P, dim=-1) # (B, N, M)
    P_cumsum_cat = torch.cat((torch.zeros(*P_cumsum.shape[:-1], 1, dtype=P_cumsum.dtype,
                                          device=P_cumsum.device),
                              P_cumsum), dim=-1)
    P_cumsum_exclusive = P_cumsum_cat[...,:-1]
    P_cumsum = P_cumsum_cat[...,1:]

    # P_sum is the total sum of the individual softmaxes/distributions.
    # Shape: (*, N)
    P_sum = P_cumsum[..., M-1]
    # P_prev_sum_product, of shape (*, N) contains the product of all the P_sum
    # values for the *previous* indexes n, i.e, over n_prev < n.  We divide by
    # P_sum to make it an exclusive, not an inclusive, product.
    P_prev_sum_product = torch.cumprod(P_sum, dim=-1) // P_sum


    P_cumsum_exclusive_scaled = P_cumsum_exclusive * P_prev_sum_product.unsqueeze(-1)

    # combined_cumsums: (B, K)
    combined_cumsums = get_combined_cumsums(P,
                                            P_cumsum_exclusive_scaled,
                                            combined_indexes)
    print("combined_cumsums = ", combined_cumsums)
    print("combined_cumsums + combined_values= ", combined_cumsums + combined_values)



    # prod_cumsum is the total sum over the M axis [i.e. the last element of cumsum],
    # multiplied along the N axis, so it can be thought of as the total probability mass,
    # or the probability's normalizer, of the joint distribution.  Shape: (B,)
    prod_cumsum = P_cumsum[...,-1].prod(dim=-1)  # (B,)
    print("prod_cumsum = ", prod_cumsum)

    assert torch.all(prod_cumsum.unsqueeze(-1) > combined_cumsums)

    assert torch.all(prod_cumsum.unsqueeze(-1) >= combined_cumsums + combined_values)

    B, delta_P = compute_beta_prods(prod_cumsum, combined_values)

    assert torch.all(combined_values + delta_P > 0)


    # reorder combined_cumsums from smallest to largest, which we'll require
    # when interpolating the "skipped regions" into the random numbers.
    combined_cumsums, reorder_indexes = torch.sort(combined_cumsums, dim=-1)
    # also reorder delta_P [so that delta_P and combined_cumsums are reordered
    # in the same way]
    delta_P = torch.gather(delta_P, dim=-1, index=reorder_indexes)

    print("combined_cumsums, reordered, = ", combined_cumsums)
    print("delta_P, reordered, = ", delta_P)

    # delta_P_exclusive, of shape (*, K), is the exclusive cumulative sum of
    # delta_P, containing negative values.
    delta_P_cumsum = torch.cumsum(delta_P, dim=-1)
    delta_P_exclusive = delta_P_cumsum - delta_P
    print("delta_P_exclusive = ", delta_P_exclusive)

    # combined_cumsums_mod is combined_cumsums modified by adding the product
    # of previous delta_P's (which will be negative).  This compensates for
    # the fact that the random numbers in "sampled_values" are in a compressed
    # space where we "skip over" regions of size -delta_P.
    #
    # These are the cutoffs for subtracting the delta_P's
    # from sampled_values
    combined_cumsums_mod = combined_cumsums + delta_P_exclusive
    print("combined_cumsums_mod = ", combined_cumsums_mod)


    # CAUTION: if the product of sums is too large, this rand_values
    # will not be sufficiently
    # random!!  We need to leave some headroom.
    # rand_values are random in {0, 1, ..., B-1}
    rand = torch.randint((2**63 - 1), B.shape) % B
    # rand, rand + B, rand + 2B, ...., rand + (K-1)B
    samples = rand.unsqueeze(-1) + B.unsqueeze(-1) * torch.arange(K)
    print("rand = ", rand)
    print("sampled = ", samples)

    shifted_samples = compute_shifted_samples(combined_cumsums_mod,
                                              delta_P,
                                              samples)
    print("shifted_samples = ", shifted_samples)

    check_shifted_samples(combined_cumsums,
                          delta_P,
                          shifted_samples,
                          prod_cumsum)

    indexes = get_indexes_for_samples(P, P_cumsum, P_cumsum_exclusive, shifted_samples)


if __name__ == '__main__':
    _test_combined()
    _test_compute_beta()
    _test_soft_sample()
    #test_normalizer()
