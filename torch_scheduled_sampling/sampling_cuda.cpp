#include <torch/extension.h>


/*
  sampling forward function, CUDA version (the backward is implemented in python).  Please see
  `sample_combined_forward` in sampling_ref.py for a comparable PyTorch implementation
  in Python, that is easier to follow.

  Sample from a distribution that is the product of softmaxes.  We will sample
  K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
  for a computed beta.

    Args:
         p: A Tensor of shape (B, N, M): either normalized log-probs (if input_is_log==False),
             or normalized probabilities; normalized along the M axis.  B is the batch
             size; M will typically be a power of 2 but can be any number, ideally with only
             small prime factors; and N must be in [1,2,3,4].  The type of p can be half,
             float or double.

        rand: A Tensor of int64_t of shape (B * (N+1),) containing random numbers in [0..2**63-1]
           which is the largest int64_t.  Actually this could just as easily be randum
           numbers in 0..2**64-1, as we'll be interpreting this as uint64_t.  The reason
           we want a flat array rather than array of shape (B, N+1) is so that we
           can access it in a more memory-efficient way, avoiding scattered reads.

         K: An integer, the number of samples required, with 0 < K < M
   input_is_log:  True if p represents normalized log-probs, False if it represents
             probabilities.

    Returns: (indexes, combined_indexes, weights)
       indexes: of shape (B, K, N), for each of K samples from a distribution it contains
            an N-tuple of indexes saying which combination of indexes from the
            component distributions were sampled.
       combined_indexes: of shape (B, K), contains the N-tuples in `indexes` combined
           into a single integer in [0..(K**N)-1]
       weights: of shape (B, K), gives the weight associated with each sample,
            which will equal max(p, beta) for a beta specific to the batch element,
            i.e. to the product of the distributions (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
       epsilon: this is only needed by calling code if input_is_log == False.
            Of shape (,), i.e. a scalar, this is a small value that will be
            used in the backward pass to prevent division by zero; it corresponds
            to the amount we added to the distribution in the forward pass before sampling.
*/
std::vector<torch::Tensor>
sample_combined_forward_cuda(torch::Tensor probs, // [B][N][M]
                            torch::Tensor rand,   // [B]
                            int K, bool input_is_log);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_combined_forward_cuda", &sample_combined_forward_cuda,
        "Multi-softmax sampling function forward (CUDA)");
}
