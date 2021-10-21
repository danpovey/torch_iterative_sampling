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




def _iterative_sample_dispatcher(
        cumsum: torch.Tensor,
        rand: torch.Tensor,
        seq_len: int) -> torch.Tensor:
    """
    Dispatcher for iterative
    """
    if cumsum.is_cuda:
        if torch_iterative_sampling_cuda is None:
            raise EnvironmentError(f'Failed to load native CUDA module')
        return torch_iterative_sampling_cuda.iterative_sample_cuda(
            cumsum, rand, seq_len)
    else:
        return torch_iterative_sampling_cpu.iterative_sample_cpu(
            cumsum, rand, seq_len)



def ensure_nonzero(probs: torch.Tensor) -> torch.Tensor:
    """
    Return a version of `probs` that lacks zeros and ones.
    Args:
      probs: a Tensor of probabilities of shape (*, N), where N is the number of classes
    Return:
      Returns a modified version of probs without exact zeros or ones.
    """
    N = probs.shape[-1]
    assert probs.dtype in [torch.float32, torch.float64, torch.float16]
    epsilon = (1.2e-07 if probs.dtype == torch.float32 else
               (2.3e-16 if probs.dtype == torch.float64 else
                9.8e-04)) # <-- assume float16, if supported.
    return (probs * (1-N*epsilon)) + epsilon

def iterative_sample(probs: torch.Tensor,
                     num_seqs: int,
                     seq_len: int,
) -> torch.Tensor:
    """Sample repeatedly from the categorical distribution in `probs`, each time
      only selecting from classes that were not previously seen.

      Args:
          probs:  A tensor of probabilities of discrete classes, of shape (*, N)
                  where N is the number of classes.
                  Is expected to sum to one (over the N indexes), and be nonnegative.
                  We advise to pass it through ensure_nonzero(probs) before
                  this function.
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
    probs = probs.to(dtype=torch.float32)
    N = probs.shape[-1]
    rest_shape = probs.shape[:-1]
    probs = probs.reshape(-1, N)
    B = probs.shape[0]

    rand_int32 = torch.randint(0, (2**31)-1, (B, num_seqs), dtype=torch.int32,
                               device=probs.device)
    indexes = _iterative_sample_dispatcher(probs, rand_int32, seq_len)
    indexes = indexes.view(*rest_shape, num_seqs, seq_len)
    return indexes


def exclusive_cumsum(x: Tensor, dim: int) -> Tensor:
    """
    Exclusive cumulative sum, i.e. cumulative sum minus the current element.
    """
    return torch.cumsum(x, dim) - x


class PredictorInputParams(nn.Module):
    def __init__(self, num_classes: int, predictor_dim: int,
                 num_discretization_levels: int,
                 seq_len: int) -> None:
        """
        This module stores some embedding parameters that are part of how we predict the
        probabilities of the classes and weights.  Its forward

        num_classes:  Number of classes in the discrete distribution that we are modeling,
                e.g. 512 (think of this as a hidden dimension).  Referred to elsewhere as N.
        predictor_dim:  Dimension of the input to the predictor that predicts log-probs
               of classes and weights.  This predictor will be a sum of various inputs.
               E.g. 512.
        num_discretization_levels: the number of discretization
               levels from the SamplingBottleneckModule, dictates
               range of `value_indexes`
        seq_len:  The length K of the random sequences of classes.
        """
        super(PredictorInputParams, self).__init__()

        # Initialize various embeddings.

        # All embeddings will be returned after multiplying by this scale, this
        # is intended to make them learn fast enough.  We divide by this scale
        # when initializing.
        self.embed_scale = predictor_dim ** 0.5
        self.num_discretization_levels = num_discretization_levels

        def Embedding(num_embeddings, embedding_dim):
            return nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim,
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
        # shape: (K, N)
        self.position_embed = nn.Parameter(torch.randn(seq_len, predictor_dim) * (1 / self.embed_scale))


    def forward(self, class_indexes: Tensor,
                value_indexes: Tensor,
                base_predictor: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
           class_indexes: the same-named output from
              SamplingBottleneckModule.forward(), a LongTensor of
              shape (*, num_seqs, seq_len).
           value_indexes: the same-named output from
              SamplingBottleneckModule.forward(), a LongTensor of
              shape (*, num_seqs, seq_len).
           base_predictor:  A Tensor of shape (*, predictor_dim) that encodes a
              prediction of the distribution, e.g. derived from previous
              frames' embeddings.   Will be combined with other embeddings
              owned in this class.

        Returns (class_predictor, value_predictor), where:

           class_predictor: of shape (*, S, predictor_dim), this is to be used
               as the input to a feedforward network that predicts the next
               class in the sequence.
           value_predictor: of shape (*, S, predictor_dim), this is to be used
               as the input to a feedforward network that predicts the (discretized)
               weight for the class that we just saw.
        """
        # N == predictor_dim
        class_present_embedding = self.class_present_embed(class_indexes)  # (*, S, K, N)
        # class_present_embedding_cumsum includes the current class.  This is OK when
        # predicting the value, but for predicting the class itself we'll have to subtract the
        # current class.
        # class_present_embedding_cumsum shape: (*, S, K, predictor_dim)
        class_present_embedding_cumsum = torch.cumsum(class_present_embedding, dim=-2)

        selected_values = value_indexes * (1.0 / (self.num_discretization_levels - 1))  # (*, S, K)

        # class_value_embedding will be of shape (*, S, K, N).  Caution: this could be on
        # the large size if S*K is large, memory might be an issue.

        class_value_embedding = self.class_value_embed(class_indexes) * selected_values.unsqueeze(-1)
        # So do exclusive-cumsum so that for each point k in the sequence, the model
        # is aware of the values of all previously emitted classes and their values (but not the
        # current value, which is not known yet).
        class_value_embedding_cumsum = exclusive_cumsum(class_value_embedding, dim=-2)

        # class_query_embedding has shape (*, S, K, N)
        class_query_embedding = self.class_query_embed(class_indexes)


        common_embedding = (class_value_embedding_cumsum +
                            class_present_embedding_cumsum +
                            self.position_embed)

        # reshape base_predictor to (*, 1, 1, predictor_dim)
        base_predictor = base_predictor.unsqueeze(-2).unsqueeze(-2)

        # we have to subtract class_present_embedding in order to exclude the
        # current class from class_present_embedding_cumsum (otherwise it's
        # cheating, as the thing we're predicting would be known).
        class_predictor = base_predictor + self.embed_scale * (common_embedding - class_present_embedding)

        # For predicting the weight, we don't need to subtract the current class
        # any more because by the time we predict the weight, the class is
        # known.  However we do need to add class_query_embedding, because
        # otherwise the model wouldn't easily be able to tell which class was
        # the one whose weight was being queried.
        value_predictor = base_predictor + self.embed_scale * (common_embedding + class_query_embedding)

        return class_predictor, value_predictor

class BottleneckPredictor(nn.Module):
    def __init__(self, num_classes: int,
                 predictor_dim: int,
                 num_discretization_levels: int,
                 seq_len: int,
                 hidden_dim: int,
                 num_hidden_layers: int) -> None:
        """
        This module is used to predict the discrete symbols (classes and weights)
        of the discrete bottleneck
        in the SamplingBottleneckModule.  It handles only the prediction within
        individual frames; any cross-frame aspect of the prediction (i.e. taking
        care of the larger sequence across time) will be handled outside this module,
        likely by something like a transformer.  The forward function of this module
        accepts a predictor that would be the output that that transformer (or other
        sequential model).

        Args:
            num_classes:  The number of classes used in the SamplingBottleneckModule,
                          e.g. 512.
            num_discretization_levesl:  The number of discretization levesl in
                          the SamplingBottleneckModule, e.g. 256.
            seq_len:      The seq_len given to the SamplingBottleneckModule, which
                          is the number of symbols to sample from the distribution
                          each time; e.g. 8 or 16.
            hidden_dim:   The hidden dimension to use in the two feedforward networks owned
                          by this class, e.g. 512 or 1024.
            num_hidden_layers:  The number of hidden layers with ReLU activations to
                          use in the two feedforward networks owned
                          by this class, e.g. 1 or 2.
        """
        super(BottleneckPredictor, self).__init__()
        self.input_params = PredictorInputParams(num_classes, predictor_dim,
                                                 num_discretization_levels, seq_len)
        def create_predictor(output_dim):
            layers = []
            cur_dim = predictor_dim
            for i in range(num_hidden_layers):
                if i != 0:
                    layers.append(nn.LayerNorm(cur_dim))
                layers.append(nn.Linear(cur_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                cur_dim = hidden_dim
            layers.append(nn.Linear(cur_dim, output_dim))
            return nn.Sequential(*layers)

        self.class_predictor_module = create_predictor(num_classes)
        self.value_predictor_module = create_predictor(num_discretization_levels)

    def forward(self,
                probs: Optional[Tensor],
                class_indexes: Tensor,
                value_indexes: Tensor,
                base_predictor: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the predicted total log-probs of the classes and values selected
        by the SamplingBottleneckModule.

        Args:
           probs: the same-named output from SamplingBottleneckModule.forward(),
             a Tensor of shape (*, num_classes).  You can provide None if you want
             class_indexes to be used instead of `probs`, but using `probs`
             should give a lower-variance estimate of the derivative.  We use
             this without grad (we detach it), for a couple reasons.
              - only a difference of log-likes would make sense to train the
                bottleneck and its input;
              - It wouldn't be "all the derivative" anyway, the real
                mechanics of backprop to get the correct derivatives w.r.t. `probs`
                are much more complicated than just enabling grad here.
           class_indexes: the same-named output from
              SamplingBottleneckModule.forward(), a LongTensor of
              shape (*, num_seqs, seq_len).
           value_indexes: the same-named output from
              SamplingBottleneckModule.forward(), a LongTensor of
              shape (*, num_seqs, seq_len).
           base_predictor:  A Tensor of shape (*, predictor_dim) that encodes a
              prediction of the distribution, e.g. derived from previous
              frames' embeddings via some kind of sequential model.
        Returns (class_logprobs, value_logprobs), where:
            class_logprobs: a Tensor of shape (*) [matching the inputs];
              this gives the UNNORMALIZED logprob of the class indexes,
              summed over
              the sequence (seq_len) and averaged over the parallel
              sequences (num_seqs).
            value_logprobs: a Tensor of shape (*) [matching the inputs];
              this gives the UNNORMALIZED logprob of the discretized weights,
              summed over
              the sequence (seq_len) and averaged over the parallel
              sequences (num_seqs).
        """
        if probs is not None:
            probs = probs.detach()
        # class_predictor: (*, num_seqs, seq_len, predictor_dim)
        # value_predictor: (*, num_seqs, seq_len, predictor_dim)
        class_predictor, value_predictor = self.input_params(class_indexes,
                                                             value_indexes,
                                                             base_predictor)
        # class_prediction: (*, num_seqs, seq_len, num_classses)
        # value_prediction: (*, num_seqs, seq_len, num_discretization_levels)
        # both of these are unnormalized logprobs.
        class_prediction = self.class_predictor_module(class_predictor)
        value_prediction = self.value_predictor_module(value_predictor)

        class_prediction, mask = self.mask_prev_classes(class_prediction,
                                                        class_indexes)

        class_prediction = class_prediction.log_softmax(dim=-1)
        value_prediction = value_prediction.log_softmax(dim=-1)

        # class_all_logprobs and value_all_logprobs are of shape
        # (*, num_seqs, seq_len)
        # Even if probs is supplied, in training mode once in every 20 or so minibatches we use
        # the sampled classes instead of `probs`.  This seems to allow the training to
        # get started faster than it otherwise would.
        if probs is None or (self.training and random.random() < 0.05):
            class_all_logprobs = torch.gather(class_prediction, dim=-1,
                                              index=class_indexes.unsqueeze(-1)).squeeze(-1)
        else:
            probs_expanded = probs.unsqueeze(-2).unsqueeze(-2).expand(mask.shape).contiguous()
            # mask out probs of previously seen classes to zero; these are no longer possible,
            # so distribution at each point should exclude these prior classes.
            probs_expanded.masked_fill_(mask, 0.0)
            class_prediction = class_prediction.clone()
            class_prediction.masked_fill_(mask, 0.0)
            # Re-normalize probs to sum to one after the last dim.
            probs_expanded = probs_expanded / probs_expanded.sum(dim=-1).unsqueeze(-1)
            class_all_logprobs = (probs_expanded * class_prediction).sum(dim=-1)

        value_all_logprobs = torch.gather(value_prediction, dim=-1,
                                          index=value_indexes.unsqueeze(-1)).squeeze(-1)

        num_seqs = class_indexes.shape[-2]

        # class_logprobs and value_logprobs are of shape (*).
        class_logprobs = torch.sum(class_all_logprobs, dim=(-2,-1)) * (1 / num_seqs)
        value_logprobs = torch.sum(value_all_logprobs, dim=(-2,-1)) * (1 / num_seqs)

        if random.random() < 0.0001:
            class_seq = torch.mean(class_all_logprobs,
                                   dim=tuple(range(class_all_logprobs.ndim - 1)))
            value_seq = torch.mean(value_all_logprobs,
                                   dim=tuple(range(class_all_logprobs.ndim - 1)))
            print(f"Class/value logprobs, as seqs, are: {class_seq}/{value_seq}")

        return class_logprobs, value_logprobs


    def mask_prev_classes(self, class_logprobs: Tensor, class_indexes: Tensor) -> Tensor:
        """
        Replaces the logprobs in `class_logprobs` that correspond to classes
        that were previously seen in a sequence (and are therefore now disallowed),
        with -infinity.  This means that we don't have to waste modeling power
        learning the fact that classes cannot be seen more than once.

          Args:
             class_logprobs: a Tensor of shape (*, seq_len, num_seqs, num_classes),
                   containing un-normalized logprobs of the classes.
                   WARNING: used destructively (actually operates in-place)
               class_indexes: a LongTensor of shape (*, seq_len, num_seqs), containing
                   class indexes in {0..num_classes-1}.
        Returns: (class_logprobs, mask)
            class_logprobs:  An in-place modified version of class_logprobs with
               elements corresponding to previously seen classes in the sequence
               replaced with -infinity.
            mask: a BoolTensor with the same shape as `class_logprobs`, i.e.
               (*, seq_len, num_seqs, num_classes),with True in the places
               where we put -infinity.
        """
        counts = torch.zeros_like(class_logprobs, dtype=torch.int16)
        counts.scatter_(-1, class_indexes.unsqueeze(-1), 1)
        mask = (exclusive_cumsum(counts, dim=-2) != 0)
        # use -1e+20 instead of -infinity for the mask, because otherwise when
        # we multiply by zero later we'll get nan's.
        class_logprobs.masked_fill_(mask, -1e+20)
        return class_logprobs, mask

    def get_prob_scales(self, probs: Tensor, class_indexes: Tensor) -> Tensor:
        """
        Returns some scaling factors >= 1 that compensate for the fact that we will
        be masking out elements in `probs` that correspond to previously emitted
        classes in the sequence: the scales will be those that would cause `probs`
        to sum to one after such masking.

        Args:
          probs: A Tensor of shape (*, num_classes), which sums to one along
                dim=-1, containing class probabilities, as returned from
                SamplingBottleneckModule.
          class_indexes:  a LongTensor of shape (*, num_seqs, seq_len) as returned
                from a SamplingBottleneckModule, containing elements in
                {0..num_classes-1}.
        Return:  Returns a Tensor of shape (*, num_seqs, seq_len), containing
                 scaling factors >= 1.
        """
        num_seqs = class_indexes.shape[-2]
        num_classes = probs.shape[-1]
        probs_temp = probs.unsqueeze(-2).expand(probs.shape[:-1], num_seqs, num_classes)
        # probs_temp now of shape (*, num_seqs, num_classes).
        selected_probs = torch.gather(probs_temp, dim=-1, index=class_indexes)
        # selected_probs is now of shape (*, num_seqs, seq_len)
        selected_probs_cumsum = exclusive_cumsum(selected_probs, dim=-1)
        # selected_probs_cumsum is of shape (*, num_seqs, seq_len), containing
        # the exclusive cumulative sum of selected_probs
        # epsilon is the floating point epsilon.. we'll be dividing by inv_scale, so
        # must be very careful about roundoff
        epsilon = (1.2e-07 if probs.dtype == torch.float32 else
                   (2.3e-16 if probs.dtype == torch.float64 else
                    9.8e-04)) # <-- assume float16, if supported.
        inv_scale = (1 - selected_probs_cumsum).clamp(min=epsilon)


class _ParameterizedDropout(torch.autograd.Function):
    # Please see the function parameterized_dropout() for a description of
    # the interface.
    @staticmethod
    def forward(ctx, probs: Tensor, mask: Tensor,
                values: Tensor, random_rate: float = 0.5,
                epsilon: float = 0.1) -> Tensor:
        probs = probs.detach()
        values = values.detach()

        C = probs.shape[-1]
        rest_shape = list(probs.shape)[:-1]

        # frame_mask is a bool tensor of shape (*, 1), that's True on
        # frames that will be random (i.e. use "mask" and not "probs").
        frame_mask = (torch.rand(*rest_shape, device=probs.device) < random_rate).unsqueeze(-1)
        ctx.save_for_backward(probs, mask, values, frame_mask)
        ctx.epsilon = epsilon

        actual_mask = (frame_mask * mask + torch.logical_not(frame_mask) * probs)
        ans = values * actual_mask
        return ans


    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tuple[Tensor, None, Tensor, None, None]:
        (probs, mask, values, frame_mask) = ctx.saved_tensors
        epsilon = ctx.epsilon

        # `actual_mask` is what we multiplied the values by in the forward pass.
        actual_mask = (frame_mask * mask + torch.logical_not(frame_mask) * probs)

        mask_weight = values / (values + epsilon)   # <-- close to 1 if an element of values >> epsilon
        probs_weight = 1 - mask_weight   # <-- close to 1 if an element of values << epsilon

        # The formula is an attempt to reduce the variance of the derivatives.  The assumption
        # is that if a 'value' is small, then the derivative of the output w.r.t. the
        # (value * mask) will be about the same whether the mask is 0 or 1, so we can just use
        # the element of 'probs', treating it as an expectation.
        # whereas if value >> epsilon, we should use the actual mask, for accuracy.
        values_grad = ans_grad * (mask_weight * actual_mask  + probs_weight * probs)

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

        grad_factor_random = (mask * s1) + (torch.logical_not(mask) * s2)
        grad_factor_deterministic = 1.0
        grad_factor = (frame_mask * grad_factor_random + torch.logical_not(frame_mask) * grad_factor_deterministic)
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


def parameterized_dropout(probs: Tensor,
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
             i.e. just output_grad * mask.  We currently require that elements of values
             be in the interval [0,1] (this is needed for a formula involving epsilon).
  random_rate:  A float value that determines how often we use the zero-one mask; the
             rest of the time, we use the expected value (probs).
    epsilon:  A float value used to prevent division by zero in backprop; controls
             a bias-variance tradeoff in derivatives (small->lower bias, higher
             variance).

    Returns: A Tensor with the same shape as `probs`, `mask` and `values`, i.e.
            (*, C), which is randomly somewhere between values * mask and
            values * probs.
    """
    return _ParameterizedDropout.apply(probs, mask, values, random_rate, epsilon)



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
    # the 0.999 is to ensure we don't get exactly one.  Caution: this won't work
    # in half precision, so we use an assert for now (otherwise we'd later get
    # an error in a scatter kernel)
    assert values.dtype != torch.float16
    indexes = (values * (num_discretization_levels - 1) + 0.999*torch.rand_like(values)).to(dtype=torch.long)
    ans = indexes * (1.0 / (num_discretization_levels - 1))
    y = _WithGradOf.apply(ans, values)
    return y, indexes



class SamplingBottleneckModule(nn.Module):
    def __init__(self, dim: int , num_classes: int,
                 seq_len: int = 8,
                 num_discretization_levels: int = 128,
                 random_rate: float = 0.0,
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
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_discretization_levels = num_discretization_levels
        self.random_rate = random_rate
        self.epsilon = epsilon
        assert epsilon > 0

        self.input_scale = nn.Parameter(torch.tensor([3.0]))

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

    def forward(self, x: Tensor, num_seqs: int = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward function.
        Args:
            x: a Tensor of shape (*, F) where F is the number of input features/channels,
               equal to `dim` arg to constructor.
         num_seqs:  The number of parallel sequences (S).  Should probably be 1 unless
               you are planning to model these probabilities

        Returns (y, probs, class_indexes, value_indexes, class_entropy, frame_entropy), where:

           y: a Tensor of shape (*, F), like x, where F is the `dim` arg to this class's
               constructor.  This is the main output, to be given to
               later modules' forward function.
          probs: a Tensor of shape (*, N) where N is the number of classes
               (`num_classes` to constructor), giving the probabilities with which we
               sampled the classes in `class_indexes`.  Currently without gradient,
               to save a little memory.  This will be used when predicting
               the classes in `class_indexes`, replacing samples with
               expectations in order to reduce the variance of the resulting
               derivatives.
          class_indexes:  a LongTensor of shape (*, num_seqs, seq_len) containing
               the randomly sampled classes in the range [0..num_classes-1].
               Will be useful if you want to model the probabilities of the output
               of this layer.
          value_indexes:  a LongTensor of shape (*, num_seqs, seq_len) containing
               the values of the randomly sampled classes, whose elements are in
               the range [0..num_discretization_levels-1] == [0..M-1]
               Will be useful if you want to model the probabilities of the output
               of this layer.
          class_entropy: A scalar Tensor containing the entropy over classes,
               of the distribution summed over all frames; will be close to
               log(num_classes) if all classes are equally likely overall.
               Might be useful if you want to ensure this stays large
               (might help optimization by ensuring we don't waste classes).
          frame_entropy: A scalar Tensor containing the per-frame entropy over
               classes, reflecting the average uncertainty in the distribution over
               classes on individual frames.  Will be less than class_entropy.
               Might be useful if you want to ensure this doesn't get too
               small (this might help optimization).

        """
        # logprobs has shape (*, N); it is the input to the softmax that determines the
        # probabilities of sampling different classes on each iteration of sampling.
        # (Not the same as the marginal probabilities).
        x = x * self.input_scale
        probs = self.to_both_softmax(x)
        values = probs + self.to_values_softmax(x)

        probs = torch.softmax(probs, dim=-1)

        avg_probs = probs.mean(dim=tuple(range(probs.ndim-1)))
        class_entropy = -(avg_probs * (avg_probs + 1.0e-20).log()).sum()
        frame_entropy = -(probs * (probs + 1.0e-20).log()).sum() / (probs.numel() / probs.shape[-1])
        if random.random() < 0.001:
            class_perplexity = class_entropy.exp().to('cpu').item()
            frame_perplexity = frame_entropy.exp().to('cpu').item()
            print(f"Class perplexity={class_perplexity}, frame perplexity={frame_perplexity}, vs. max possible {avg_probs.numel()}")

        # values also has shape (*, N); it is expected to be similar to `probs`,
        # since we want to bias towards transmitting the larger values.
        values = torch.softmax(values, dim=-1)

        probs = ensure_nonzero(probs)

        # compute marginal probabilities of selecting any given class at any point
        # in the sequence of K distinct samples.
        marginals = compute_marginals(probs, self.seq_len)

        # indexes shape is (*, S, K)
        class_indexes = iterative_sample(probs, num_seqs=num_seqs,
                                         seq_len=self.seq_len)


        # values_expanded is values expanded from (*, N) to (*, S, N)
        N = probs.shape[-1]
        values_expanded = values.unsqueeze(-2).expand(*probs.shape[:-1], num_seqs, N)
        chosen_values = torch.gather(values_expanded, dim=-1, index=class_indexes)

        # discrete_values and value_indexes have shape (*, S, K)
        discrete_values, value_indexes = discretize_values(chosen_values,
                                                           self.num_discretization_levels)

        # discrete_actual_values has shape (*, N), it is just the input `values`
        # discretized.
        discrete_actual_values, _ = discretize_values(values,
                                                      self.num_discretization_levels)

        if not self.training:
            # in eval mode, don't use the discretized values.
            discrete_values = values

        class_indexes_0 = class_indexes.select(dim=-2, index=0)
        mask = torch.zeros_like(values)
        # Put 1.0 in the mask vector at positions specified by `class_indexes_0`.
        mask.scatter_(-1, class_indexes_0, 1.0)

        random_rate = 0.0 if not self.training else self.random_rate

        y = parameterized_dropout(marginals, mask, discrete_actual_values,
                                  random_rate=random_rate,
                                  epsilon=self.epsilon)
        y = self.to_output(y)
        y = self.layer_norm(y)

        return y, probs, class_indexes, value_indexes, class_entropy, frame_entropy




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
    return 1 - (1 - probs) ** alpha.unsqueeze(-1)

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
    # just makes sure the forward function runs without crashing.
    # There is more extensive testing of this, including training in iterative_sampling_test.py
    dim = 256
    num_classes = 512
    num_discretization_levels = 128
    seq_len = 8
    m = SamplingBottleneckModule(dim, num_classes,
                                 num_discretization_levels=num_discretization_levels,
                                 seq_len=seq_len)

    predictor_dim = 128

    p = PredictorInputParams(num_classes, predictor_dim,
                             num_discretization_levels=num_discretization_levels,
                             seq_len=seq_len)

    hidden_dim = 256
    num_hidden_layers = 2
    b = BottleneckPredictor(num_classes, predictor_dim,
                            num_discretization_levels, seq_len, hidden_dim,
                            num_hidden_layers)

    feats = torch.randn(30, 4, dim)

    y, probs, class_indexes, value_indexes, class_entropy, frame_entropy = m(feats)

    print(f"Shapes of: y={y.shape}, probs={probs.shape}, class_indexes={class_indexes.shape}, value_indexes={value_indexes.shape}")

    base_predictor = torch.randn(30, 4, predictor_dim)
    (class_predictor, value_predictor) = p(class_indexes, value_indexes, base_predictor)
    print(f"class_predictor shape={class_predictor.shape}, value_predictor shape={value_predictor.shape}")
    print(f"class_predictor variance={(class_predictor**2).mean()} value_predictor variance={(value_predictor**2).mean()}")

    assert value_indexes.min() == 0 and value_indexes.max() < num_discretization_levels

    print("y part = ", y[0])

    (class_logprobs, value_logprobs) = b(None, class_indexes, value_indexes, base_predictor)
    assert class_logprobs.shape == (30, 4)
    assert value_logprobs.shape == (30, 4)
    (class_logprobs, value_logprobs) = b(probs, class_indexes, value_indexes, base_predictor)
    assert class_logprobs.shape == (30, 4)
    assert value_logprobs.shape == (30, 4)


def _compare_seen_expected_products(a: Tensor, b: Tensor, a_name: str = "seen", b_name: str = "expected",
                                    threshold: float = 0.02):
    """
    Compute and display products between a and b, and check that (a*b).sum() is close to (b*b).sum().
    """
    ab = (a * b).sum().to('cpu').item()
    aa = (a * a).sum().to('cpu').item()
    bb = (b * b).sum().to('cpu').item()
    a_flip_b = (a.flip(dims=(0,)) * b).sum().to('cpu').item()
    err = (1.0 - ab / (0.5 * (ab + bb)))
    print(f"{a_name}*{b_name}:{ab}, {a_name}*{a_name}:{aa}, {b_name}*{b_name}:{bb}, {a_name}-flipped*{b_name}:{a_flip_b}, rel_err={err}")

    assert abs(err) < threshold


def _test_iterative_sample():
    for device in 'cpu', 'cuda':
        print("device=", device)
        device = torch.device(device)
        B = 3000
        N = 256
        logprobs = 2 * torch.randn(B, N, device=device)
        probs = logprobs.softmax(dim=-1)

        num_seqs = random.randint(1, 8)

        seq_len = random.randint(5, 15)
        indexes = iterative_sample(probs, num_seqs=num_seqs, seq_len=seq_len)
        #print("indexes = ", indexes)

        indexes_0 = indexes[:,0,:]  # take 1st sequence.
        zero_ones = torch.zeros(B, N, device=device)
        zero_ones.scatter_(-1, indexes_0, 1.0)

        # Check all indexes in each sequence are distinct.
        assert zero_ones.sum().to('cpu').item() == indexes_0.numel()

        expected_marginals = compute_marginals(probs, seq_len)

        _compare_seen_expected_products(zero_ones, expected_marginals, "seen", "expected")


def _test_get_derivative_scales():
    probs = torch.rand(200, 300, 2)
    epsilon_tensor = 0.1 + torch.rand(200, 300, 2)
    for epsilon in [0.0, 0.1, 1.0, 2.0, epsilon_tensor]:
        s1, s2 = get_derivative_scales(probs, epsilon)
        assert torch.all(s1>=0) and torch.all(s2>=0)
        one = (s1 * probs) + (s2 * (1-probs))
        assert(torch.allclose(one, torch.ones_like(one)))


def _test_parameterized_dropout():
    probs = torch.rand(100, 200, 5)
    mask = (torch.rand_like(probs) < probs)
    values = torch.rand_like(probs)
    probs.requires_grad = True
    values.requires_grad = True

    output_grad = torch.randn_like(probs)
    quadratic_grad = torch.randn_like(probs)

    for random_rate in (0.0, 0.5, 1.0):
        for epsilon in (0.001, 0.1, 0.5, 1.0):
            for quadratic_term in (0.0, 1.0, 3.0):
                """
                The 'quadratic_term' part requires an explanation.  (we assume you've read the docuemntation
                for parameterized_dropout()).  We construct a loss function that is:

                   (output * output_grad.sum() + 0.5 * quadratic_term * (output * output).sum())   (eq. 1)

                (Remember, as epsilon -> 0, our backprop approach is supposed to approach exact
                derivatives for any quadratic loss function).

                What is the expected derivative contribution from the quadratic part of the
                loss function?  We'll first compute the expected loss, which is the thing we
                are supposed to be backpropping, and then compute the derivative of that.
                Again considering just the quadratic part in (eq. 1), the expected loss
                if random_rate == 1.0 (i.e. we use the random
                      0.5 * quadratic_term * (probs * values * values).sum()   (eq. 2).
                [note: with probability (1-probs), the output is zero so the squared
                output would also be zero.]
                If random_rate == 0.0, i.e. output == probs * values, it is:
                      0.5 * quadratic_term * (probs * probs * values * values).sum()   (eq. 3).
                In general, the quadratic part of the loss function is (expected value):
                    0.5 * random_rate * quadratic_term * (probs * values * values).sum()  +
                    0.5 * (1 - random_rate) * quadratic_term * (probs * probs * values * values).sum()  (eq. 4).
                The derivative of this w.r.t. 'probs' is:
                    (0.5 * random_rate * quadratic_term * values * values  +
                     (1 - random_rate) * quadratic_term * probs * values * values).sum()  (eq. 5).
                and the derivative of this w.r.t. 'values' is:
                       (random_rate * quadratic_term * probs * values  +
                     (1 - random_rate) * quadratic_term * probs * probs * values).sum()  (eq. 6).
                """

                probs.grad = None
                values.grad = None

                output = parameterized_dropout(probs, mask, values, random_rate, epsilon)
                expected_output = values * probs
                print(f"test_parameterized_dropout: random_rate={random_rate}, epsilon={epsilon}, quadratic_term={quadratic_term}:")
                _compare_seen_expected_products(output, expected_output)

                if random_rate == 0.0:
                    assert torch.allclose(output, values * probs) # deterministic in this case.


                (output * output_grad + 0.5 * quadratic_term * quadratic_grad * output * output).sum().backward()

                expected_probs_grad = output_grad * values
                # for next line, see (eq. 5) above
                expected_probs_grad += quadratic_term * quadratic_grad * (0.5 * random_rate * values * values +
                                                                          (1-random_rate) * probs * values * values)
                expected_values_grad = output_grad * probs
                # for next line, see (eq. 6) above
                expected_values_grad += quadratic_term * quadratic_grad * (random_rate * probs * values +
                                                                           (1-random_rate) * probs * probs * values)


                # if all three of quadratic_term, epsilon and random_rate are nonzero,
                # there is a risk of inaccuracy in expected vs. observed derivatives.
                threshold=0.015 + (0.075 * quadratic_term * epsilon * random_rate)
                if threshold > 0.02:
                    print(f"Threshold={threshold}, since quadratic_term={quadratic_term} and epsilon={epsilon} and random_rate={random_rate}")

                # Note: this test won't always succeed, and the threshold is heuristic, not based on
                # a properly derived formula.  The threshold above can be played with
                # if the failures bother you.
                _compare_seen_expected_products(probs.grad, expected_probs_grad, "probs_grad", "expected_probs_grad",
                                                threshold=threshold)
                _compare_seen_expected_products(values.grad, expected_values_grad, "values_grad", "expected_values_grad",
                                                threshold=threshold)

if __name__ == '__main__':
    _test_iterative_sample()
    _test_sampling_bottleneck()
    _test_parameterized_dropout()
    _test_get_derivative_scales()
    _test_discretize_values()
    _test_compute_marginals()
    _test_normalizer()
