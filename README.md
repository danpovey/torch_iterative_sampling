
# Scheduled sampling for PyTorch


This repository implements Python code and C++ and CUDA extensions for efficient
importance sampling from either a discrete distribution, or a product of
independent discrete distributions.  The main reason this is nontrivial
is that it ensures the samples are all distinct; and doing so is not
super-trivial for a product of discrete distributions.



## Usage

#### Installation

```shell script
pip install torch-discounted-cumsum
```

#### API

- `sample_combined(p, K, input_is_log)`.

see [torch_scheduled_sampling/sampling.py]:
<code>
def sample_combined(p: Tensor, K: int, input_is_log: bool) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample from a distribution that is the product of softmaxes.  We will sample
    K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
    for a computed beta.
    Args:
         p: A Tensor of shape (*, N, M): either normalized log-probs (if input_is_log==False),
             or normalized probabilities; normalized along the M axis.
             N must be in [1,2,3,4]; in the common case, N will be 1, you can use unsqueeze().

         K: An integer, the number of samples required, with 0 < K < M
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
            weights will sum to 1 along the K axis.  The class-inclusion
            probabilities in the sample would be given by (p / weights), assuming
            input_is_log == False, or (p.exp() / weights) otherwise; but
            we output them in this format as it is less likely to lead to
            large numbers in backprop.
     """
</code>

#### Example

```python
import torch
from torch_scheduled_sampling import sample_combined


logprobs = (2 * torch.randn(3, 1, 8)).log_softmax(dim=2)

indexes, _combined_indexes, weights = sample_combined(logprobs, K=2, input_is_log=True)

print(indexes.shape)
print(indexes.squeeze(-1))
print(weights.squeeze(-1))
indexes, _combined_indexes, weights = sample_combined(logprobs, K=2, input_is_log=True)
importance_logprobs = weights.log() - torch.gather(logprobs, dim=2, index=indexes.transpose(1, 2)).squeeze(1)
print(importance_logprobs)
```

Output:
```
torch.Size([3, 2, 1])  # indexes.shape
tensor([[0, 6],
        [1, 6],
        [2, 4]])    # indexes.squeeze(-1)
tensor([[0.7474, 0.2526],
        [0.1940, 0.8060],
        [0.5000, 0.5000]])  # weights.squeeze(-1)
tensor([[ 8.9407e-08,  1.7194e-01],
        [ 1.0342e+00, -2.9802e-08],
        [ 5.6074e-01,  5.0099e-01]])  # importance_logprobs
```
