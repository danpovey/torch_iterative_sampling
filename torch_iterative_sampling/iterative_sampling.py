import os

import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional, Union
from torch.utils.cpp_extension import load

VERBOSE = False


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    pass # TEMP
    #import torch_iterative_sampling_cpu
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




def _iterative_sample_dispatcher(
        cumsum: torch.Tensor,
        rand: torch.Tensor,
        seq_len: int) -> torch.Tensor:
    """
    Dispatcher for iterative
    """
    if cumsum.is_cuda:
        if torch_iterative_sample_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_iterative_sample_cuda.iterative_sample_cuda(
            cumsum, rand, seq_len)
    else:
        return torch_iterative_sample_cpu.iterative_sample_cpu(
            cumsum, rand, seq_len)



def iterative_sample(probs: torch.Tensor,
                     num_seqs: int,
                     seq_len: int) -> torch.Tensor:
    """Sample repeatedly from the categorical distribution in `probs`, each time
      only selecting from classes that were not previously seen.

      Args:
          probs:  A tensor of probabilities of discrete classes, of shape (*, N)
                  where N is the number of classes.
                  Is expected to sum to one (over the N indexes), and be nonnegative.
       num_seqs:  The number of parallel sequences to sample; must be > 0.
       seq_len:   The length of the sequences of sample; must be strictly between
                  9 and N.
       Returns:
                  Returns a LongTensor of shape (*, S, K), containing the sampled
                  indexes, with elements in the range [0..N-1].  Each element
                  of each sequence is sampled with probability proportional to
                  `probs`, excluding from consideration classes already sampled
                  within that sequence.
    """
    N = probs.shape[-1]
    rest_shape = probs.shape[:-1]
    probs = probs.reshape(-1, N)
    cumsum = exclusive_cumsum(probs, dim=-1)
    B = probs.shape[0]
    rand = torch.rand(B, num_seqs, dtype=probs.dtype, device=probs.device)
    indexes = _iterative_sample_dispatcher(cumsum, rand, seq_len)


    return indexes


def exclusive_cumsum(x: Tensor, dim: int) -> Tensor:
    """
    Exclusive cumulative sum, i.e. cumulative sum minus the current element.
    """
    return torch.cumsum(x, dim) - x


class PredictorInputParams(nn.Module):
    def __init__(self, num_classes: int, predictor_dim: int,
                 seq_len: int) -> None:
        """
        This module stores some embedding parameters that are part of how we predict the
        probabilities of the classes and weights.

        num_classes:  Number of classes in the discrete distribution that we are modeling,
                e.g. 512 (think of this as a hidden dimension).  Referred to elsewhere as N.
        predictor_dim:  Dimension of the input to the predictor that predicts log-probs
               of classes and weights.  This predictor will be a sum of various inputs.
               E.g. 512.
        seq_len:  The length K of the random sequences of classes.
        """

        # Initialize various embeddings.
        # self.embed is the embedding used for both the encoder and decoder.

        # All embeddings have a scale, this is intended to make them learn fast enough.
        self.embed_scale = predictor_dim ** 0.5


        def Embedding(num_embeddings, embedding_dim):
            return nn.Embedding(
                num_embeddings=num-embeddings, embedding_dim=embedding_dim,
                _weight=torch.randn(num_embeddings, embedding_dim) * (1 / self.embed_scale)
            )

        # Embedding by which we indicate that a class is "present" at a previous
        # position in the sequence (this gets set independently of its weight)
        self.class_present_embed = Embedding(num_classes, predictor_dim)
        # Embedding by which we indicate the (input) value of a class, this gets scaled
        # by `value`.
        self.class_value_embed = Embedding(num_classes, predictor_dim)

        # Embedding by which we indicate, when querying a weight, the identity of the class whose
        # weight we are querying.
        self.class_query_embed = Embedding(num_classes, predictor_dim)

        # position_embed is the embedding by which we indicate our current position in the sequence (k=0,1,..,K-1).
        self.position_embed = nn.Parameter(torch.randn(seq_len, predictor_dim) * (1 / self.embed_scale))

        # This will get multiplied by 'alpha' and added to the embeddings.
        self.alpha_embed = nn.Parameter(torch.randn(predictor_dim) * (1 / self.embed_scale))

        # This will get multiplied by the sum of previously emitted values (input versions,
        # not scaled-up versions).
        self.tot_values_embed = nn.Parameter(torch.randn(predictor_dim) * (1 / self.embed_scale))



    def forward(self, values: Tensor, indexes: Tensor, alpha: Tensor,
                base_predictor: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
         values:  values is the same-named input to iterative_sample(), of shape
                (*, N)
         indexes:  The same-named output of iterative_sample(), a LongTensor of
              shape (*, S, K); its elements are in [0..N-1]
         alpha:  The same-named output of iterative_sample(), a Tensor of
                 shape (*, S)
         base_predictor:  A Tensor of shape (*, predictor_dim) that encodes a
                prediction of the distribution, e.g. derived from previous
                frames' embeddings.   Will be combined with other embeddings
                owned in this class.

        Returns (class_predictor, weight_predictor), where:

           class_predictor: of shape (*, S, predictor_dim), this is to be used
               as the input to a feedforward network that predicts the next
               class in the sequence.
           weight_predictor: of shape (*, S, predictor_dim), this is to be used
               as the input to a feedforward network that predicts the (discretized)
               weight for the class that we just saw.
        """
        class_present_embedding = self.class_present_embed(indexes)  # (*, S, K, predictor_dim)
        class_present_embedding_cumsum = torch.cumsum(class_present_embedding, dim=-2)

        # class_value_embedding has shape (*, S, K, predictor_dim)
        selected_values = self._get_selected_values(values, indexes)  # (*, S, K)
        class_value_embedding = self.class_value_embed(indexes) * selected_values.unsqueeze(-1)
        class_value_embedding = exclusive_cumsum(class_value_embedding, dim=-2)


        # class_query_embedding has shape (*, S, K, predictor_dim)
        class_query_embedding = self.class_query_embed(index)

        # alpha_embedding has shape (*, S, 1, predictor_dim)
        alpha_embedding = self.alpha_embed * alpha.unsqueeze(-2)

        # tot_values_embedding has shape (*, S, K, predictor_dim)
        tot_values_embedding = exclusive_cumsum(selected_values, dim=-2).unsqueeze(-1) * self.tot_values_embed


        common_embedding = (class_value_embedding +
                            class_present_embedding_cumsum +
                            self.position_embed +
                            alpha_embedding +
                            tot_values_embedding)


        # reshape base_predictor to (*, 1, 1, predictor_dim)
        base_predictor = base_predictor.unsqueeze(-2).unsqueeze(-2)

        # we have to subtract class_present_embedding in order to exclude the
        # current class from class_present_embedding_cumsum (otherwise it's
        # cheating).
        class_predictor = base_predictor + self.embed_scale * (common_embedding - class_present_embedding)

        # We don't need to subtract the current class any more because by the time we predict the
        # weight, the class is known.  However we do need to add class_query_embedding, because
        # otherwise it wouldn't be able to tell which class was the one whose weight was being queried.
        weight_predictor = base_predictor + self.embed_scale * (common_embedding + class_query_embedding)

        return (class_predictor, weight_predictor)


    def _get_selected_values(values: Tensor, indexes: Tensor) -> Tensor:
        """
        Return the elements of `values` that are indicated by the `indexes`
        tensor.  Args:
         values:  values is the same-named input to iterative_sample(), of shape
                (*, N)
         indexes:  The same-named output of iterative_sample(), a LongTensor of
              shape (*, S, K); its elements are in [0..N-1]
        Returns selected_values, which is a Tensor of shape (*, S, K) containing
              the elements of `values` indicated by `indexes`.
        """
        # Reshape indexes to (*, S * K)
        indexes_shape = list(indexes.shape)
        S = indexes_shape[-2]
        K = indexes_shape[-1]
        indexes_shape[-2] = S * K
        indexes_shape = indexes_shape[:-1]
        indexes_reshaped = indexes.reshape(indexes_shape)

        # use torch.gather to get elements from `values`, of shape (*, S * K)
        selected_values = torch.gather(values, indexes_reshaped, dim=-1)
        # reshape to (*, S, K)
        selected_values = selected_values.reshape(indexes.shape)
        return selected_values



class _ClassesTotalLogprob(torch.autograd.Function):
    # see get_classes_total_logprob for more info on the interface
    @staticmethod
    def forward(ctx,
                class_prediction: Tensor,
                probs: Tensor,
                scales: Tensor,
                indexes: Tensor) -> Tensor:
        """
        Quick recap of sizes:
          class_prediction: un-normalized logprobs of shape (*, S, K, N)
          probs: (normalized) probabilities of shape (*, N)
          scales: a tensor of shape (*, S, K) containing, at each point in the
                sequence (k=0,1..), 1.0 / (1 - sum of the probs of previously
                selected classes).  Values are >= 1.0.  Used to renormalize probs
                after masking.
          indexes: a LongTensor of shape (*, S, K) containing values in [0..N-1]
                which are distinct along the K axis.
        """
        # Caution: we expect that class_prediction is the only input which
        # would have requires_grad == True.
        assert not probs.requires_grad and not scales.requires_grad
        ctx.save_for_backward(class_prediction.detach(), probs,
                              scales, indexes)

        with torch.no_grad():
            ans = _compute(class_prediction, probs, scales, indexes)
            return ans

    @staticmethod
    def backward(ctx: Tensor, ans_grad: Tensor) -> Tuple[Tensor, None, None, None]:
        """
        Backward method which only returns a derivative for class_probs; obtaining
        derivatives w.r.t. the other elements is rather nontrivial as we have to consider
        the larger context, not just this one operation.
        """
        (class_prediction, probs, scales, indexes) = ctx.saved_tensors

        class_prediction.requires_grad = True
        ans_temp = _compute(class_prediction, probs, scales, indexes)
        (ans_temp * ans_grad.detach()).sum().backward()
        return class_prediction.grad, None, None, None

    @staticmethod
    def _compute(class_prediction: Tensor,
                 probs: Tensor,
                 scales: Tensor,
                 indexes: Tensor) -> Tensor:
        N = class_prediction.shape[-1]
        # mask is of shape (*, S, K, N), with True in masked
        # positions
        mask = _get_mask(indexes, N)

        S = class_prediction.shape[-3]
        K = class_prediction.shape[-2]
        N = class_prediction.shape[-1]
        rest_shape = list(class_prediction.shape)[:-3]

        # clone may not be necessary.
        class_prediction = class_prediction.clone()
        class_prediction.masked_fill_(mask, float('-infinity'))
        class_logprobs = torch.log_softmax(class_prediction, dim=-1)

        # make scales of shape (*, S, K, 1)
        scales = scales.unsqueeze(-1)

        # logically the masking and the scale should be on the probs, not the
        # logprobs, but doing it this way is likely more memory efficient, since
        # the logprobs already have more dimensions; and the result is the same
        scaled_logprobs = (class_logprobs * scales)
        scaled_logprobs.masked_fill_(mask, float('0'))

        # make probs of shape (*, 1, 1, N)
        probs = probs.unsqueeze(-2).unsqueeze(-2)
        # divide by the number of sequences S because we want to average over
        # the alternative sequences (different samples)
        return torch.dot(scaled_logprobs.reshape(-1), probs.reshape(-1)) / S



    @staticmethod
    def _get_mask(indexes: Tensor, N: int) -> Tensor:
        """
        Args:
         indexes: a LongTensor with shape (*, S, K) containing class indexes in the range [0..N-1]
               N: the number of classes
         Returns: a Tensor with shape (*, S, K, N) and dtype=torch.bool containing True for masked
               positions (where there is a previous class) and False for un-masked positions).
        """
        rest_shape = list(indexes.shape)[:-2]
        S = indexes.shape[-2]
        K = indexes.shape[-1]
        ans = torch.zeros(*rest_shape, S, K, N, dtype=torch.bool)
        ans.scatter_(-1, indexes.unsqueeze(-1), True)
        mask = (torch.cumsum(ans, dim=-2, dtype=torch.int8) - ans.to(dtype=torch.int8)).to(dtype=torch.bool)



def get_classes_total_logprob(class_prediction: Tensor,
                              probs: Tensor,
                              scales: Tensor,
                              indexes: Tensor) -> Tensor:
    """
    Get the total logprob for prediction of the classes.  We try to do this in a
    memory efficient way.  Note: it would be possible to just use:
      get_weights_total_logprob(class_prediction, indexies)
    but what this function implements is an expectation of the logprob of the class,
    over the possible choices of class at each point in the randomly chosen sequence.
    This will give a derivative with a lower variance than using the classes we actually
    chose at each point.

      Args:
        class_prediction: A Tensor of un-normalized logprobs of shape (*, S, K, N) i.e.
             (*, num_seqs, seq_len, num_classes).
         probs:  A Tensor containing the class probabilities (used for sampling),
              of shape (*, N) where N is the number of classes.  These are the
              same at each point in the sequence, except that we have to exclude previously
              sampled classes and renormalize.
        scales:  A Tensor of shape (*, S, K) containing the scale that we'll have to
              scale up `probs` to make it sum to 1 after excluding previously chosen
              classes
       indexes: A LongTensor of shape (*, S, K) containing selected class indexes in
              the range [0..N-1].  Because we compute the total logprob via an
              expectation and not according to the actually selected class indexes,
              these indexes are only used to define which prior classes to
              exclude when normalizing `class_prediction`, and zeroing elements of
              `probs`.

    Returns: a scalar Tensor containing the logprob averaged over the S axis and
             summed over other dimensions.  This can be interpreted as the logprob
             of samples, but modified to use expectations instead of single
             samples wherever it is practical to do so, to minimize variance.
    """
    return _ClassesTotalLogprob.apply(class_prediction, probs,
                                      scales, indexes)


def get_weights_total_logprob(weight_prediction: Tensor,
                              samp_values: Tensor) -> Tensor:
    """
    Args:  weight_prediction: a Tensor of un-normalized logprobs of shape (*, S, K, M), i.e.
              (*, num_seqs, seq_len, num_discretization_levels)
           samp_values:  The discretized weights at the output of iterative_sample(), which
               is a LongTensor of shape (*, S, K) containing integers in [0..M-1].

    Returns:  a scalar Tensor containing the logprob averaged over the S axis and
           sumed over other dimensions.
    """
    samp_values = samp_values.unsqueeze(-1) #  (*, S, K, 1)

    # Note: we could replace this operation with an expectation, adding more args to this
    # function; and this would slightly reduce the variance of the derivatives, but since
    # we always assign nonzero probability to at most two of the discrete weight values,
    # the difference wouldn't really be significant.

    # chosen_values is of shape (*, S, K, 1).
    chosen_values = torch.gather(weight_prediction, dim=-1, index=samp_values)

    S = weight_prediction.shape[-3]

    return chosen_values.sum() / S



class _FakeParameterizedDropout:
    # Please see the function fake_parameterized_dropout() for a description of
    # the interface.
    @staticmethod
    def apply(ctx, probs: Tensor, mask: Tensor,
              values: Tensor, random_rate: float = 0.5,
              epsilon: float = 0.1) -> Tensor:
        probs = probs.detach()
        values = values.detach()

        C = probs.shape[-1]
        rest_shape = list(probs.shape)[:-1]

        # frame_mask is a bool tensor of shape (*, 1)
        frame_mask = (torch.randn(*rest_shape, device=probs.device) < random_rate).unqueeze(-1)
        ctx.saved_for_backward(probs, mask, values, frame_mask)
        ctx.epsilon = epsilon

        actual_mask = (frame_mask * mask + (1-frame_mask) * probs)
        ans = values * actual_mask
        return ans


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, None, Tensor, None, None]:
        (probs, mask, values, frame_mask) = ctx.saved_tensors
        epsilon = ctx.epsilon

        # `actual_mask` is what we multiplied the values by in the forward pass.
        actual_mask = (frame_mask * mask + (1-frame_mask) * probs)

        values_grad = ans_grad * actual_mask

        # See get_derivative_scales() to understand the function of
        # epsilon_tensor, it's the epsilon arg to that function.
        #
        # epsilon_tensor is close to epsilon when an element of values >>
        # epsilon, but approaches 1 when values << epsilon, assuming epsilon is
        # small.  We can use large elements in epsilon_tensor for small elements
        # of `values` because if an element `values` is small it's more valid to
        # treat the loss function as being linear in `ans`, so we can set
        # epsilon to a large value which gives lower-variance derivatives.
        # Actually we could perhaps separate this formula to use two constants.
        epsilon_tensor = epsilon * (1 + 1/(values + epsilon))

        s1, s2 = get_derivative_scales(probs, epsilon_tensor)

        grad_factor_random = (mask * s1) + ((1-mask) * s2)
        grad_factor_deterministic = 1.0
        grad_factor = (frame_mask * grad_factor_random + (1-frame_mask) * grad_factor_deterministic)
        actual_mask_grad = ans_grad * values
        probs_grad = grad_factor * actual_mask_grad

        return probs_grad, None, values_grad, None, None



def get_derivative_scales(probs: Tensor,
                          epsilon: Union[Tensor, float]) -> Tuple[Tensor, Tensor]:
    """
    This function, which is intended to be used without gradient, returns scales
    s1 and s2 which are to be used in backprop when we are applying a dropout-like
    zero-or-one mask.  Here is the scenario: we have some probabilities `probs`, with
    0 <= probs <= 1, and we randomly generate a zero-or-one mask, like:

              mask = (torch.rand_like(probs) < probs).

    We are going to get an output by masking some `values`, as in:

              output = values * mask,   (1)

    and we want to know how to propagate the derivative of `output` back to `probs`.
    (Note that in normal dropout, the dropout prob is just a constant and there is
    no derivative).  A simple way to do this would be to treat (1) the same as:
                output = values * probs     (2)
    .. the output in (2) is the same as the expected value of (1).   This amounts
    to backprop like:
             mask_grad = output_grad * values.
              probs_grad = mask_grad                  (3)
    If the loss function were linear in `output`, the derivative in (3) would
    be "correct", in the sense that it would be the derivative of the expected
    value of the loss function.  Of course,
    the loss function won't, in general, be linear in `output`.  For arbitrary loss functions,
    there's no way to get "correct" derivatives unless we measure derivatives
    at all points of `mask` between zero and one, which would require changing
    the forward pass.  But we can do a little better than (2): if we assume
    the loss function is quadratic in `output`, then the loss derivative w.r.t.
    the mask would vary linearly between mask==0 and mask==1, and we can
    tread the derivative of `prob` as being: (half the derivative at 0) plus
    (half the derivative at 1).  Here, the average derivative at 1 is just
    (mask_grad * mask / probs), and the average derivative at 0 is
    (mask_grad * (1 - mask) / (1 - probs)).   (There is some analysis required here
    to formalize this argument).  The gradient according to this approach would be:

        probs_grad = mask_grad * 0.5 * (mask / probs + (1-mask) / (1-probs))  (4).

    A difficulty with (4) is that we are dividing by values that get close to zero,
    which can cause gradient blow-up.  We need to introduce an epsilon value to
    prevent this.  Let us rewrite (4) as:

        probs_grad = mask_grad * (s1 * mask + s2 * (1-mask))  (5).

    If we treat the loss function as linear in `output`, then the requirement
    for correctness of the derivative would be:
          (s1 * probs + s2 * (1-probs)) = 1.0              (6)
    (bear in mind that "mask" is 1 with probability "probs"; this expression just
    gives the expected value of the scale in parentheses in (5).  Our proposed
    value for s1 and s2 is as follows:

         s1 = 0.5 * (1+epsilon)/(probs+epsilon) + epsilon/(1+epsilon-probs)    (7)
         s2 = 0.5 * (1+epsilon)/(1+epsilon-probs) + epsilon/(probs+epsilon)

    where epsilon > 0; actually epsilon does not have to be less than 1; as epsilon
    gets large, s1 and s2 both approach 0.5). You can verify that the formula above
    satisfies (6), e.g. type the following into
    wolframalpha.com:
      p * 0.5 * (1.1/(p+0.1) + 0.1/(1.1-p))  +  (1-p) * 0.5 * (1.1/(1.1-p) + 0.1/(p+0.1))
    The advantage of (7) is that s1 and s2 never get larger than 0.5/epsilon, but
    for probs between about [epsilon..1-epsilon], it is nearly equivalent to (4).


    Args:
         probs: a Tensor of arbitrary shape with elements in the interval [0,1],
              representing the probability of a mask value being 1 (so similar to
              1-dropout_rate).
         epsilon:  A smoothing value greater than zero.  This can be either
              a float (e.g. 0.1 or 0.2), or it can be any tensor that broadcasts
              with `probs`.  The idea is that you might want epsion to vary
              with the `values` in (1): if an element of `value` is close to zero,
              then the linear assumption is closer to being correct, and we might
              want a larger epsilon value.
     Returns:
         Returns a pair of tensors (s1, s2), intended to be applied similar to eqn. (5)
         above.
    """
    inv_p1 = 0.5 / (probs + epsilon)
    inv_p2 = 0.5 / ((1.0 + epsilon) - probs)
    common = epsilon * (inv_p1 + inv_p2)
    s1 = inv_p1 + common
    s2 = inv_p2 + common
    return s1, s2


def fake_parameterized_dropout(probs: Tensor,
                               mask: Tensor,
                               values: Tensor,
                               random_rate: float = 0.5,
                               epsilon: float = 0.1) -> Tensor:
    """
    This function returns (values * mask) if random_rate == 1.0 and
    (values * probs) if random_rate == 0.0 or if we are in eval mode
    (self.training == false).  Otherwise, it randomly selects on frame-by-frame
    / vector-by-vector basis, which of the two to use.  The main point of this
    function is that it intelligently backpropagates derivatives in such a way
    that you can meaningfully train `probs`.  See the function `get_derivative_scales()`
    to understand the central point of how we get derivatives w.r.t. `probs`.


    Args:
       probs: the probabilities with which the `mask` vector was chosen; we'll be able
             to compute derivatives w.r.t. this.  A Tensor of shape (*, C) where C is
             interpreted as the channel dimension. These must be in the interval [0,1].
       mask: A (possibly boolean) Tensor of shape (*, C) and values 0/False or 1/True,
             True/1 if this value is to be "passed through".
             The caller asserts that these values have been chosen with probabilities
             equal to `probs`, e.g. as:
                mask = (torch.rand_like(probs) < probs)
             (In practice we may be sampling with a more complicated method which has
             marginal probabilities equal to `probs`; the correctness of the derivatives
             becomes a little weaker in that case).
     values: A Tensor of shape (*, C), the same as probs_and mask; these are the
             values that are to be multiplied by a mask (or sometimes scaled by `probs`,
             if random_rate < 1).  The derivatives backpropagated to here are exact,
             i.e. just output_grad * mask.  We require that elements of values
             be in the interval [0,1] (needed for a formula involving epsilon).
  random_rate:  A float value that determines how often we use the zero-one mask; the
             rest of the time, we use the expected value (probs).
    epsilon:  A float value used to prevent division by zero in backprop; controls
             a bias-variance tradeoff in derivatives (small->lower bias, higher
             variance).

    Returns: A Tensor with the same shape as `probs`, `mask` and `values`, i.e.
           (*, C), which is randomly somewhere between values * mask and values * probs.

    """
    return _FakeParameterizedDropout.apply(probs, mask, values, random_rate, epsilon)



class _WithGradOf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        """
        Returns x but will assign the gradient to y.
        """
        return x

    @staticmethod
    def backward(ctx, ans_grad) -> Tuple[None, Tensor]:
        return None, ans_grad


def discretize_values(values: Tensor,
                      num_discretization_levels: int) -> Tuple[Tensor, Tensor]:
    """
    Pseudo-randomly discretize an input tensor, whose elements must be in the
    interval [0,1], into a fixed number of levels, e.g. 128.  The pseudo-random
    part has to do with rounding.

    Args:
       values: a Tensor of arbitrary shape
       num_discretization_levels:  The number of discrete values that we divide
           the interval [0,1] into, e.g. 128.
    Returns (y, indexes), where:
         y: a randomly discretized version of `values`, whose elements will
           differ from the corresponding elements of `values` by no more than
           1/(num_discretization_levels - 1).  Derivatives go "straight through"
           from this to `values`.
         indexes: a LongTensor containing the discrete indexes corresponding
           to `y`, in the range [0..num_discretization_levels-1].
    """
    indexes = (values * (num_discretization_levels - 1) + torch.rand_like(values)).to(dtype=torch.long)
    ans = indexes * (1.0 / (num_discretization_levels - 1))
    return _WithGradOf.apply(ans, values), indexes



class SamplingBottleneckModule(nn.Module):
    def __init__(self, dim: int , num_classes: int,
                 seq_len: int = 8,
                 num_discretization_levels: int = 128,
                 random_rate: float = 0.5,
                 epsilon: float = 0.1) -> None:
        """
    Create sampling bottleneck module.  This uses an iterative sampling algorithm to
    represent the hidden feature by a fixed number of randomly chosen classes (e.g. 8
    classes drawn from 512 possible classes), together with values in the range [0,1]
    for all of the randomly chosen classes.  The basic idea is that we turn the
    hidden feature into a categorical distribution over a number of classes, and
    we transmit that distribution in a randomized, lossy way that focuses modeling
    power on the classes that dominate the distribution.  So it's somewhat like
    a discrete sampling operation-- in fact, it is such an operation-- but we allow
    much more information to pass through than just a single class label.

    Args:
      dim: feature dimension before and after this module, e.g. 512.
    num_classes:  The number of discrete classes we form a distribution over, e.g. 512.
    seq_len:  The number of (distinct) classes we sample from the distribution when
           transmitting information over this channel
    num_discretization_levels:  The number of levels we discretize the interval [0,1]
           into when transmitting values, e.g. 128.
    random_rate:  The probability that we randomize a particular frame, versus
           using the expectation.
    epsilon: A value used in the backprop that affects derivatives w.r.t. probabilities;
           a value close to zero is more theoretically accurate but may lead to
           some derivatives being quite large.
        """
        super(SamplingBottleneckModule, self).__init__()
        self.dim = dim
        self.K = seq_len
        self.N = num_classes
        self.M = num_discretization_levels
        self.random_rate = random_rate
        self.epsilon = epsilon


        # We assume there is a layer-norm just prior to this module, so we don't
        # include layer norm on the input.

        # to_both_softmax is a linear projection that will go to a softmax to
        # the probs and values
        self.to_both_softmax = nn.Linear(dim, num_classes, bias=False)
        # the output of to_values_softmax gets added to the output of to_both_softmax,
        # and it becomes the values (so the values are not just copies of the probs).
        # This would be zero if probs == values, we initialize it to zero.
        self.to_values_softmax = nn.Linear(dim, num_classes, bias=False)


        self.to_output = nn.Linear(num_classes, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.to_values_softmax.weight, 0.)

    def forward(self, x: Tensor, num_seqs: int = 1) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward function.
        Args:
            x: a Tensor of shape (*, F) where F is the number of input features/channels,
               equal to `dim` arg to constructor.
         num_seqs:  The number of parallel sequences (S).  Should probably be 1 unless
               you are planning to model these probabilities
        Returns (y, probs, class_indexes, value_indexes), where:

           y: a Tensor of shape (*, F), like x, where F is the `dim` arg to this class's
               constructor.  This is the main output, to be given to
               later modules' forward function.
          probs: a Tensor of shape (*, N) where N is the number of classes
               (`num_classes` to constructor), giving the probabilities with which we
               sampled the classes in `class_indexes`.  Currently without gradient,
               to save a little memory.  This will be used when predicting
               the classes in `class_indexes`, as a way to replace samples with
               expectations to reduce the variance of the derivatives.
          class_indexes:  a LongTensor of shape (*, num_seqs, seq_len) containing
               the randomly sampled classes in the range [0..num_classes-1] == [0..N-1]
               Will be useful if you want to model the probabilities of the output
               of this layer.
          value_indexes:  a LongTensor of shape (*, num_seqs, seq_len) containing
               the values of the randomly sampled classes, whose elements are in
               the range [0..num_discretization_levels-1] == [0..M-1]
               Will be useful if you want to model the probabilities of the output
               of this layer.
        """
        # logprobs has shape (*, C); it is the input to the softmax that determines the
        # probabilities of sampling different classes on each iteration of sampling.
        # (Not the same as the marginal probabilities).
        probs = self.to_both_softmax(x)
        probs = torch.softmax(probs, dim=-1)

        # values also has shape (*, C); it is expected to be similar to `probs`,
        # since we want to bias towards transmitting the larger values.
        values = probs + self.to_values_softmax(x)
        values = torch.softmax(values, dim=-1)

        # compute marginal probabilities of selecting any given class at any point
        # in the sequence of K distinct samples.
        marginals = compute_marginals(probs, self.K)

        # indexes shape is (*, S, K)
        class_indexes = iterative_sample(probs, num_seqs)

        discrete_values, value_indexes = discretize_values(values, self.M)

        if not self.training:
            # in eval mode, don't use the discretized values.
            discrete_values = values


        class_indexes_0 = class_indexes.select(dim=-2, index=0)
        mask = torch.zeros_like(values)
        # Put 1.0 in the mask vector at positions specified by `class_indexes_0`.
        mask.scatter_(-1, class_indexes_0.unsqueeze(-1), 1.0)

        random_rate = 0.0 if not self.training else self.random_rate

        y = fake_parameterized_dropout(probs, mask, values,
                                       random_rate=random_rate,
                                       epsilon=self.epsilon)

        y = self.to_output(y)
        y = self.layer_norm(y)

        return y, class_indexes, value_indexes




def compute_marginals(probs: Tensor, K: int) -> Tensor:
    """
    Args:
     probs: a Tensor of float with shape (*, N), interpreted as
       the probabilities of categorical distributions over N classes (should
       sum to 1 and be nonnegative).
     K: the number of times to (conceptually) sample from the distribution
      `probs`, always excluding previously-chosen classes.

    Returns: the marginal probability, for each class, of selecting
       that class at any point during K rounds of selecting a new, previously
       unchosen class with probability proportional to `probs`.
       The result will be of shape (*, N), and the sum of the result over
       then N axis will equal K (or be very close to K).  Returned
       elements will be in the interval [0,1].
    """
    alpha = compute_normalizer(probs, K)
    return 1 - ((1 - probs) ** alpha.unsqueeze(-1))

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



def _test_normalizer():
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

def _test_compute_marginals():
    probs = (torch.randn(10, 20, 30) * 2.0).softmax(dim=-1)
    K = 8
    marginals = compute_marginals(probs, K)
    err = marginals.sum(dim = -1) - K
    avg_err = (err * err).mean().sqrt()
    print("avg_err of marginals is ", avg_err)
    assert avg_err < 0.2

def _test_discretize_values():
    values = torch.rand(10, 20, 30)
    values.requires_grad = True
    M = 32
    discrete_values, indexes = discretize_values(values, M)

    grad = torch.rand_like(values)

    # These tests will work with very high but not 1 probability.
    assert torch.min(discrete_values).item() == 0
    assert torch.min(indexes).item() == 0
    print("max is", torch.max(indexes).item())
    assert torch.max(indexes).item() == M - 1
    assert torch.max(discrete_values).item() == 1.0

    discrete_values.backward(grad)

    assert torch.allclose(values.grad, grad)


def _test_sampling_bottleneck():
    dim = 256
    num_classes = 512
    m = SamplingBottleneckModule(dim, num_classes)
    feats = torch.randn(30, 4, dim)

    y, class_indexes, value_indexes = m(feats)

    print(f"Shapes of: y={y.shape}, class_indexes={class_indexes.shape}, value_indexes={value_indexes.shape}")

    print("y part = ", y[0])

if __name__ == '__main__':
    _test_sampling_bottleneck()
    _test_discretize_values()
    _test_compute_marginals()
    _test_normalizer()
