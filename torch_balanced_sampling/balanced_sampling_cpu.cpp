#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>




/*
  compute_count_indexes

  Compute the indexes at which counts are nonzero.

    Args:
          counts: a Tensor of shape (num_frames, num_classes), of type torch.int32,
                 containing count values like 0, 1, 2, but mostly 0 or 1).
           indexes: an output Tensor, also of type torch.int32, of shape
                (num_frames, max_count),
                 to which we write
                 indexes 0 <= c < num_classes.  At entry it is expected to be
                 initialized to all -1's.
     This function writes to 'indexes'; it writes indexes 0 <= c < num_classes,
     corresponding to the nonzero values in the original `counts` array.  If
     a class originally a count of 2 in that array, it would be repeated twice.

     If the number of classes on a given frame was larger than `max_count`, the
     extra classes won't be written.  If fewer, we'll leave the -1 values in the
     `indexes` array.
*/
void compute_count_indexes(torch::Tensor counts, // shape=(num_frames, num_classes)
                          torch::Tensor indexes) {   // shape=(num_frames, max_count)

  TORCH_CHECK(counts.dim() == 2, "counts must be 2-dimensional");
  TORCH_CHECK(indexes.dim() == 2, "indexes must be 2-dimensional");

  int32_t num_frames = counts.size(0),
    num_classes = counts.size(1),
    max_count = indexes.size(1);
  TORCH_CHECK(indexes.size(0) == num_frames);

  TORCH_CHECK(counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(indexes.scalar_type() == torch::kInt32);

  TORCH_CHECK(counts.device().is_cpu() && indexes.device().is_cpu(),
              "inputs must be CPU tensors");

  // torch::ScalarType::Int is int32.
  auto counts_a = counts.packed_accessor32<int32_t, 2>(),
    indexes_a = indexes.packed_accessor32<int32_t, 2>();

  for (int32_t f = 0; f < num_frames; f++) {
    auto this_counts_a = counts_a[f],
      this_indexes_a = indexes_a[f];
    int32_t cur_index = 0;
    for (int32_t c = 0; c < num_classes; c++) {
      int32_t this_count = this_counts_a[c];
      for (int32_t i = 0; i < this_count && cur_index < max_count; i++,cur_index++)
        this_indexes_a[cur_index] = c;
    }
  }
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_count_indexes", &compute_count_indexes,
        "Utility function to compute count indexes");
}
