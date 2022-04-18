#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Tuple



class _ComputeNormalizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, K: int) -> Tensor:
        # Please see compute_normalizer() below for documentation of the interface.
        # We are solving
        # \sum_{i=0}^{N-1}  1 - (1-x_i) ** alpha == K
        # i.e.:
        # \sum_{i=0}^{N-1}  1 - exp(alpha * log(1-x_i)) == K
        #
        # Let us define y_i = log(1-x_i).  Then we are solving:
        #
        # \sum_{i=0}^{N-1}  1 - exp(alpha * y_i) == K
        #
        #  K-N + \sum_{i=0}^{N-1} exp(alpha * y_i)  == 0  (eqn. 1)
        # d(LHS)/d(alpha) where LHS means left hand side of (eqn. 1) is:
        #  d = \sum_{i=0}^{N-1} y_i exp(alpha * y_i)
        #
        # An iterative solution (we'll see whether this converges) is:
        #    alpha := alpha - err / d
        # where err is the LHS of (eqn. 1).

        requires_grad = x.requires_grad
        x = x.detach()
        y = (1 - x).log()

        N = x.shape[-1]
        alpha = torch.empty(list(x.shape)[:-1], device=x.device,
                            dtype=x.dtype).fill_(K)

        for i in range(3):
            exp = torch.exp(alpha.unsqueeze(-1) * y)
            err = exp.sum(dim=-1) + (K - N)
            d = (exp * y).sum(dim=-1)
            alpha -= err / d
            if __name__ == '__main__':
                print(f"Iter {i}, alpha={alpha}, exp={exp}, err={err}, d={d}")

        # in:
        #  K-N + \sum_{i=0}^{N-1} exp(alpha * y_i) == 0  (eqn. 1),
        # d(LHS)/dy_i = alpha * exp(alpha * y_i).
        # d(alpha)/d(LHS) = -1/d = -1 / (sum_{i=0}^{N-1} (y_i * exp(alpha * y_i)))
        # ... so
        # but we want d(alpha)/d(x_i), which is:
        # d(alpha)/d(LHS) d(LHS)/dy_i  dy_i/dx_i.            (2)
        # dy_i/dx_i is: -1/(1-x).
        # So we can write (2) as:
        #    (alpha * exp(alpha * y_i)) / (d * (1-x))
        if requires_grad:
            ctx.deriv = (alpha.unsqueeze(-1) * exp) / (d.unsqueeze(-1) * (1 - x))
        return alpha

    @staticmethod
    def backward(ctx, alpha_grad: Tensor) -> Tuple[Tensor, None]:
        return alpha_grad.unsqueeze(-1) * ctx.deriv, None




def compute_normalizer(x: Tensor, K: int) -> Tensor:
    """
    Args:
     x: a Tensor of float with shape (*, N), interpreted as
     the probabilities of categorical distributions over N classes (should
     sum to 1 and be nonnegative).

      K: an integer satifying 0 < K < N.

    Returns a Tensor alpha of shape (*), satisfying:

        (1 - exp(-x * alpha.unsqueeze(-1))).sum(dim=-1) == K

    I.e., that:

          \sum_{i=0}^{N-1}  1 - exp(-alpha * x_i) == K.

    This will satisfy alpha >= K.  alpha, if an integer, would be the
    number of draws from the distribution x, such that the
    expected number of distinct classes chosen after that many draws
    would equal approximately K.  We can get this formula by using a Poisson assumption,
    with alpha * x_i being the expectation (the lambda parameter of the
    Poisson); another way to formulate this is to say that the probability of never
    choosing class i is 1 - (1 - x_i) ** alpha (this is equivalent
    for small alpha, using -x_i as an approximation of log(1-x_i)).  The
    version with exp() is more straightforward to differetiate though.  This
    does not really need to be super exact for our application, just fairly
    close.  Anyway, the two versions get very similar as K gets larger, because
    then alpha gets larger and the x_i's that are large make less
    difference.
    """
    return _ComputeNormalizer.apply(x, K)

    N = x.shape[-1]
    alpha_shape = list(x.shape)[:-1]

    alpha = torch.empty(alpha_shape, device=x.device, dtype=x.dtype).fill_(K)


    print(f"x = {x}")
    for i in range(3):
        exp = torch.exp(-alpha.unsqueeze(-1) * x)
        err = exp.sum(dim=-1) + (K - N)
        minus_d = (exp * x).sum(dim=-1)
        alpha += err / minus_d
        print(f"Iter {i}, alpha={alpha}, exp={exp}, err={err}, minus_d={minus_d}")
        # d2/d(alpha2) of LHS is:
        #  d1 = \sum_{i=0}^{N-1} x_i^2 exp(-alpha * x_i)
    return alpha



def test_normalizer():
    dim = 20
    K = 5
    torch.set_default_dtype(torch.double)
    for i in range(5):  # setting i too large will cause test failure, because
                        # the iterative procedure converges more slowly when the
                        # probs have a large range, and we use i below as a
                        # scale.
        B = 5  # Batch size
        probs = (torch.randn(B, dim) * i * 0.5).softmax(dim=-1)
        probs.requires_grad = True
        alpha = compute_normalizer(probs, K)
        print(f"Grad check, scale={i}")
        # atol=1.0 might seem large, but the gradients used by torch.autograd.gradcheck

        alpha_grad = torch.randn(*alpha.shape)
        (alpha * alpha_grad).sum().backward()
        probs_grad = probs.grad

        probs_delta = torch.randn(*probs.shape) * 0.0001

        alpha_delta = compute_normalizer(probs + probs_delta, K) - alpha

        observed_delta = (alpha_delta * alpha_grad).sum()
        predicted_delta = (probs_delta * probs_grad).sum()
        # Caution: for i=4, the difference can sometimes be large.  These will sometimes even be
        # the opposite sign, if it happened that alpha_delta is nearly orthogonal to the change
        # in alpha.
        print(f"For i={i}, observed_delta={observed_delta} vs. predicted_delta={predicted_delta}")


        #torch.autograd.gradcheck(compute_normalizer, (probs, K), eps=1.0e-06, rtol=0.025, atol=1.0)



if __name__ == '__main__':
    test_normalizer()
