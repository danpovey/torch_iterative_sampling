#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Tuple

# This file is not part of the implementation; it exists to test that the
# algorithms described in NOTES.md are correct.



def compute_beta(P, K):
    """
    See: ComputeBeta function [practical version] in NOTES.md.
    Args:
        P: a tensor of shape (*, M), in practice containing integers in {1,2,..2**31+1},
           but can be any integers >0 as far as this function is concerned, provided the
           cumsum does not overflow.
        K: an integer 0 < K < M
    Returns a tensor of integers B of shape (*, 1) such that:
        sum(min(P, B)) <= K*B < sum(min(P, B)) + K
    """
    M = P.shape[-1]
    R, _ = torch.sort(P, dim=-1)  # (*, M)
    Q = torch.cumsum(R, dim=-1)
    # Reference pseudocode was:
    #for k in 0,1,...K-1, in any order:
    #  # B_k is the value of B if k indexes take the l.h.s. of the "min" expression in min(B, P)
    #  B_k = (Q[M-1-i]  + K - k - 1) / (K - k)   # the "+ K - k - 1" is to ensure we round up
    #  if R[M-1-k] >= B_k and P[I-2-k] <= B_k:
    #     return B_k


    temp = torch.arange(K+1)
    # Kk, of shape (K,), contains [1, 2, ..., K], representing K-k for k = [K-1, K-2, ..., 0]
    Kk = temp[1:K+1]
    # Kk1 of shape (K,), contains [0, 1, ..., K-1], representing K-k-1 for k = [K-1, K-2, ..., 0]
    Kk1 = temp[0:K]

    Q_part = Q[...,M-K:M]   # represents: Q[...,M-1-k] for k = K-1,K-2,...,1,0

    B_k = Q_part // Kk  # shape (*, K)

    large_int = (2**32 - 1)
    R_part1 = torch.cat((R[...,M-K+1:M], torch.full((*R.shape[:-1], 1), large_int)), dim=-1)
    R_part2 = R[...,M-K:M]

    # is_ok corresponds to: "(k==0 or R[M-k] > B_k) and R[M-1-k] <= B_k" in NOTES.md
    is_ok = (torch.logical_and(R_part1 > B_k, R_part2 <= B_k))  # shape: (*, K)

    assert torch.all(torch.max(is_ok, dim=-1)[0] == 1)
    B = torch.max(B_k * is_ok, dim=-1, keepdim=True)[0]  # shape: (*, 1)

    P_min = torch.minimum(P, B)

    P_min_sum = P_min.sum(dim=-1, keepdim=True)
    assert torch.all(K * B <= P_min_sum)
    assert torch.all(P_min_sum - K < K*B)
    return B

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
    a = torch.randint(low=1, high=65535, size=(9, 8))
    K = 4
    beta = compute_beta(a, K)  # it checks its own answer..
    print("beta = ", beta)


def _test_soft_sample():
    l = 2 * torch.randn(6, 64)
    p = torch.softmax(l, dim=-1)
    soft_sample_forward(p, K=4, input_is_log=False)

if __name__ == '__main__':
    _test_compute_beta()
    _test_soft_sample()
    #test_normalizer()
