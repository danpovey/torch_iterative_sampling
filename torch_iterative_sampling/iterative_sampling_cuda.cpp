#include <torch/extension.h>


/*
  Forward of flow sampling, CUDA version.
  Please see the documentation for iterative_sampling_cpu() in iterative_sampling_cpu.cpp;
  the interface is the same.
  Returns (ans, ans_indexes).
*/
std::vector<torch::Tensor> iterative_sampling_cuda(torch::Tensor cumsum,
                                              torch::Tensor rand,
                                              float interp_prob);

/*
   backward of iterative_sampling.  Returns logits_grad; note, `logits` is the
   original log-probs from which `cumsum` was derived by softmax and then
   cumulative summation.
   We don't return a grad for `rand`; actually, such a thing is not even
   possible as the the forward function is a discontinuous function of the
   input.
   These derivatives are not correct from the point of view of a
   deterministic function of the original `logits` and `rand`.  They are
   only correct in a sense involving expectations.  See BACKPROP NOTES
   in iterative_sampling_cpu.cpp for a longer discussion.
*/
torch::Tensor iterative_sampling_backward_cuda(
    torch::Tensor cumsum,
    torch::Tensor rand,
    torch::Tensor ans_indexes,
    torch::Tensor ans_grad,
    float interp_prob,
    float straight_through_scale);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iterative_sampling_cuda", &iterative_sampling_cuda, "Flow sampling forward function (CUDA)");
  m.def("iterative_sampling_backward_cuda", &iterative_sampling_backward_cuda, "Flow sampling backward function (CUDA)");
}
