#include <torch/extension.h>


/*
  sample function, CUDA version.

    cumsum: (exclusive) cumulative integerized probabilities of input classes,
        of shape (B, N),
        where B is the batch size and N is the number of classes, so element
        cumsum[b][k] is the sum of probs[b][i] for 0 <= i < k.  We require that
        the probs be in type int32_t, and they should be converted to this by
        multiplying by (1<<31) and then converting to int32_t.  Wrapping to
        negative does not matter as we'll be using unsigned types in the
        kernel.  We require that the values in `cumsum` all be distinct,
        i.e. implicitly that all the integerized `probs` are nonzero; you can
        apply a floor of 1 to ensure this.  This avoids problems where fewer
        than K classes have nonzero prob.

    rand:  random int32_t numbers uniformly distributed on {0..1<<31 - 1},
        of shape (B,S), where S is the number of separate sequences of samples
        we are choosing from each distribution in `cumsum`.

      K: length of the random sequence to draw; must satisfy 0 < K < N.

  Returns:  Tensor of shape (B, S, K) and type torch::kInt64, containing,
            for each B, a squence of K distinct sampled integers (class
            labels) in the range [0..N-1], drawn with probability proportional
            to the differences between `cumsum` elements, but always excluding
            previously drawn classes within the current sequence.
*/
torch::Tensor sample_cuda(torch::Tensor cumsum,
                                    torch::Tensor rand,
                                    int K);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_cuda", &sample_cuda, "Iterative sampling function (CUDA)");
}
