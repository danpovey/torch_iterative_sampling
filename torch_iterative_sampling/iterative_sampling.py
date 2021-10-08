import os

import torch
from torch import Tensor
from typing import Tuple, Optional
from torch.utils.cpp_extension import load

VERBOSE = False


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_iterative_sampling_cpu
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_iterative_sampling_cpu')
    torch_iterative_sampling_cpu = load(
        name='torch_iterative_sampling_cpu',
        sources=[
            _resolve('iterative_sampling_cpu.cpp'),
        ],
        verbose=VERBOSE,
    )


try:
        import torch_iterative_sampling_cuda
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_iterative_sampling_cuda')
    torch_iterative_sampling_cuda = None
    if torch.cuda.is_available():
        torch_iterative_sampling_cuda = load(
            name='torch_iterative_sampling_cuda',
            sources=[
                _resolve('iterative_sampling_cuda.cpp'),
                _resolve('iterative_sampling_cuda_kernel.cu'),
            ],
            verbose=VERBOSE,
        )



def _iterative_sampling_forward_dispatcher(
        cumsum: torch.Tensor, rand: torch.Tensor, interp_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (output, output_indexes)
    where output is the sample, and output_indexes is a torch.Tensor of dtype=int32 and
    shape [output.shape[0], 2] that is required by the backward-pass code.
    """
    if cumsum.is_cuda:
        if torch_iterative_sampling_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return tuple(torch_iterative_sampling_cuda.iterative_sampling_cuda(
            cumsum, rand, interp_prob))
    else:
        return tuple(torch_iterative_sampling_cpu.iterative_sampling_cpu(
            cumsum, rand, interp_prob))

def _iterative_sampling_backward_dispatcher(
        cumsum: torch.Tensor,
        rand: torch.Tensor,
        ans_indexes: torch.Tensor,
        ans_grad: torch.Tensor,
        interp_prob: float,
        straight_through_scale: float) -> torch.Tensor:
    if cumsum.is_cuda:
        if torch_iterative_sampling_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_iterative_sampling_cuda.iterative_sampling_backward_cuda(
            cumsum, rand, ans_indexes, ans_grad, interp_prob,
            straight_through_scale)
    else:
        return torch_iterative_sampling_cpu.iterative_sampling_backward_cpu(
            cumsum, rand, ans_indexes, ans_grad, interp_prob,
            straight_through_scale)


class IterativeSamplingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                logits: torch.Tensor,
                rand: torch.Tensor,
                interp_prob: float,
                straight_through_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Forward propagation for flow-based algorithm for differentiably
        sampling from a categorical distribution.  The forward algorithm
        is relatively easy to explain.

        Args:
            logits:  A tensor of size (B, N) where B might be the batch size and N is
              the number of classes to sample from.  These will be interpreted as
              un-normalized log probabilities, known informally as logits.
              The softmax of `logits` is the distribution we sample from.
            rand:  A random tensor of i.i.d uniformly distributed random values of
              size (B, 3).  We make this an explicit function of this object for
              clarity, although we could just as easily compute it internally.
              It will be used in both forward and backward passes.  We require that
              this has requires_grad = False (we don't compute its derivative).
            interp_prob:  A value which must satisfy 0 < interp_prob <= 1.  It is
              the probability with which the output will be interpolated between
              two one-hot vectors, instead of being a single one-hot vector
              (actually the proportion of interpolated one-hot vectors will be
              smaller than this because we sometimes interpolate between two
              instances of the same class).  Smaller interp_prob gives output that
              is closer to really being one-hot, but with spikier derivatives.
            straight_through_scale:  With straight_through_scale = 0.0, the backprop
              returns the correct derivative; with straight_through_scale = 1.0,
              the backpropagated derivative is the one you would
              get if this function had returned logits.softmax(dim=1), and we had
              then replaced the output with this function's real output without
              informing the backprop machinery.  You can use nonzero straight_through_scale values,
              particularly early in training, to get a derivative that is biased
              but with a smaller variance.

          Return:
              Returns a tensor `result` of shape (B, N), like `logits`.  Will satisfy
              torch.all(result.sum(dim=1) == 1), i.e. each row sums to 1 and each row
              has exactly 1 or 2 columns nonzero.  The result is as if you had
              done as follows, which randomly selects two independent elements from the
              categorical distribution, and interpolates them with probability
              `interp_prob`, else picks one of the two.


            .. code-block::

               (B, N) = logits.shape
               probs = logits.softmax(dim=1)
               cum_probs = torch.cumsum(probs, dim=1)
               indexes1 = torch.searchsorted(cum_probs, rand[:,0:1])
               indexes2 = torch.searchsorted(cum_probs, rand[:,1:2])
               ans = torch.zeros(B, N)
               for b in range(B):
                 lower_bound = (1 - interp_prob) * 0.5:
                 upper_bound = (1 + interp_prob) * 0.5:
                 r = rand(b, 2)
                 if r < lower_bound or indexes1[b] == indexes2[b]:
                    ans[indexes1[b]] = 1.0
                 elif r > upper_bound:
                    ans[indexes2[b]] = 1.0
                 else:
                    e = (r - lower_bound) / interp_prob  # 0 <= e <= 1
                    ans[indexes1[b]] = 1.0 - e
                    ans[indexes2[b]] = e
                return ans

            However, the derivatives are not the same as if you had done the above, even
            if straight_through_scale == 0.0.  In fact, the code above does not compute any derivatives
            at all for `logits`; and it is not even possible to compute derivatives w.r.t
            `logits` for this result that would be meaningful for a fixed value of `rand`,
            because the output varies discontinuously as the logits change.
            Nevertheless, the derivatives we do return (assuming straight_through_scale==0.0) are valid
            in a suitable sense defined on the output distribution, assuming the random numbers are
            differently generated each time.

            See comment block labeled BACKPROP NOTES in iterative_sampling_cpu.cpp, and
            "NOTES ON PROOF" for discussion of the sense in which the algorithm is correct.
        """
        (B, N) = logits.shape
        assert rand.shape == (B, 3)
        assert not rand.requires_grad
        with torch.no_grad():
            cumsum = torch.cumsum(torch.softmax(logits, dim=1), dim=1)

        assert 0 < interp_prob and interp_prob <= 1.0
        assert 0 <= straight_through_scale and straight_through_scale <= 1.0
        ctx.interp_prob = interp_prob
        ctx.straight_through_scale = straight_through_scale
        (ans, ans_indexes) = _iterative_sampling_forward_dispatcher(cumsum, rand, interp_prob)
        if logits.requires_grad:
            ctx.save_for_backward(cumsum, rand, ans_indexes)
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[torch.Tensor, None, None, None]:
        (cumsum, rand, ans_indexes) = ctx.saved_tensors
        logits_grad = _iterative_sampling_backward_dispatcher(
            cumsum, rand, ans_indexes, ans_grad,
            ctx.interp_prob, ctx.straight_through_scale)
        return (logits_grad, None, None, None)



def iterative_sample(logits: torch.Tensor,
                interp_prob: float,
                dim: int = -1,
                straight_through_scale: float = 0.0,
                rand: Optional[Tensor] = None) -> torch.Tensor:
    """Forward propagation for flow-based sampling algorithm.  The forward
     algorithm is quite easy to explain.

      Args:
          logits:  A tensor of arbitrary shape, which will be interpreted as
            un-normalized; dimension `dim` will be treated as the class
            dimension. The softmax of `logits`
            is the distribution we (differentiably) sample from.
          interp_prob:  A value which must satisfy 0 < interp_prob <= 1.  It is
            the probability with which the output will be interpolated between
            two one-hot vectors, instead of being a single one-hot vector
            (actually the percentage of interpolated one-hot vectors will be
            smaller than this because we sometimes interpolate between two
            instances of the same one-hot output).  Smaller interp_prob gives
            output that is closer to really being one-hot, but spikier
            derivatives.
          straight_through_scale:  With straight_through_scale = 0.0, the backprop
            returns the correct derivative; with straight_through_scale = 1.0, the
            backpropagated derivative is the one you would
            get if this function had returned logits.softmax(dim=1), and we had
            then replaced the output with this function's real output without
            informing the backprop machinery.  You can use nonzero straight_through_scale
            values, particularly early in training, to get a parameter derivative that is biased
            but likely has lower variance.
          rand:  If provided, must be a uniformly distributed random tensor
            with shape: (B, 2) where B = logits.numel() / logits.shape[dim].
            This is intended to be used for testing only.

        Return:
            Returns a tensor `result` of shape (*, N), like `logits`.  Will satisfy
            torch.all(result.sum(1) == 1), i.e. each row sums to 1 and each row
            has exactly 1 or 2 columns nonzero.  Briefly, the forward part of the
            algorithm can be summarized as follows:

            With probability (1 - interp_prob), set result(b, :) to a one-hot
            vector of size N with probabilities given by logits.softmax(dim=1).

            Otherwise: independently choose two one-hot vectors of size N with
            probabilities given by logits.softmax(dim=1), and return a point
            randomly and uniformly chosen on the line between the two one-hot
            vectors.

           The backprop for this is a little complicated and involves the concept
           of flow of probability mass; we'll describe it in the C++ code.
           See comment block labeled BACKPROP in iterative_sampling_cpu.cpp.
        """
    ndim = logits.ndim
    if dim < 0:
        dim += ndim
    assert dim < ndim
    if dim != ndim - 1:
        logits = logits.transpose(dim, ndim - 1)
    shape = logits.shape
    logits = logits.reshape(-1, shape[-1])
    B = logits.shape[0]
    if rand is None:
        rand = torch.rand(B, 3, dtype=logits.dtype, device=logits.device)
    ans = IterativeSamplingFunction.apply(logits, rand, interp_prob,
                                     straight_through_scale)
    ans = ans.reshape(shape)
    if dim != ndim - 1:
        ans = ans.transpose(dim, ndim - 1)
    return ans
