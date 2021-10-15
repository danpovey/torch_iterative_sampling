#include <torch/extension.h>


/*
  iterative_sample function, CUDA version.

    cumsum: (exclusive) cumulative probabilities of input classes, of shape (B, N),
        where B is the batch size and N is the number of classes, so element
        cumsum[b][k] is the sum of probs[b][i] for 0 <= i < k.
        Implicitly the final element (not present) would be 1.0.
    rand:  random numbers uniformly distributed on [0,1], of shape (B,S), where
         S is the number of separate sequences of samples we are choosing from
         each distribution in `cumsum`.
       K: length of the random sequence to draw; must satisfy 0 < K < N.

  Returns:  Tensor of shape (B, S, K) and type torch::kInt64, containing,
            for each B, a squence of K distinct sampled integers (class
            labels) in the range [0..N-1], drawn with probability proportional
            to the differences between `cumsum` elements, but always excluding
            previously drawn classes within the current sequence.
*/
torch::Tensor iterative_sample_cuda(torch::Tensor cumsum,
                                    torch::Tensor rand,
                                    int K);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iterative_sample_cuda", &iterative_sample_cuda, "Iterative sampling function (CUDA)");
}
