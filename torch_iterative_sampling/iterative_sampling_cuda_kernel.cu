#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()
#include <cooperative_groups.h>
#include <cmath>  // for INFINITY
#include <stdio.h>

extern __shared__ int extern_buf[];
// `extern_buf` is general-purpose shared memory.


/*
  Return the index i into cumsum, such that cumsum[i - 1] <= r < cumsum[i]
  (this is the inclusive cumulative sum, which is why we have i - 1 and i,
  not i and i + 1).
  We take cumsum[-1] to be negative infinity (we could almost equivalently
  say 0, since we expect r >= 0).

  Note: if r is less than 0 or greater than cumsum[N-1], this function
  will return 0 or N-1 respectively; i.e. we won't go out of the available
  range.

  This is the same as the CPU version, except for __device__ __forceinline__.
 */
template <typename IterType, typename ScalarType>
__device__ __forceinline__ int find_class(
    IterType cumsum, ScalarType r) {
  // First search for the 'begin' element such that
  // cumsum[begin] <= r < cumsum[begin+1]
  int N = cumsum.size(0),
      begin = -1, end = N - 1;
  while (end > begin + 1) {
    int mid = begin + (end - begin) / 2;
    if (cumsum[mid] <= r)
      begin = mid;
    else
      end = mid;
  }
  return begin + 1;
}


/*
  Forward of flow_sampling.

  The thread-block will be of shape  (blockDim.x, blockDim.y)
  where blockDim.x, e.g. 16, corresponds to a group of threads assigned to process a
  single 'b' (batch) index, and blockDim.y varies with 'b'.
  We require that blockDim.x is no greater than the warp size, i.e.
  no greater than 32 (this is because we do __syncwarp() in the code, not
  __syncthreads()).

  On the grid, the threads only vary in y (gridDim.x==1), so the formula for b is:

    b = blockIdx.y * blockDim.y  +  threadIdx.y

  Template args:
      scalar_t: the floating-point type, e.g. float, double; maybe half

  Args:
      cumsum:  Accessor to the (inclusive) cumulative sum of probabilities,
             of shape (B, N)
      rand:  Accessor to a tensor of random numbers, of shape (B, 3).
           Probably the easiest way to see how they are used is to
           look at the CPU code in flow_sampling_cpu.cpp
      out: Accessor to the output (which will be empty/undefined at entry),
           of shape (B, N)

  When this kernel is invoked, the suer must specify the amount of shared memory
  to be allocated, via `extern_buf`.  This must be enough to store
    (N * blockDim.y * sizeof(scalar_t)
  bytes.
*/
template <typename scalar_t>
__global__
void flow_sampling_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> cumsum,   // B, N
    torch::PackedTensorAccessor32<scalar_t, 2> rand,   // B, 3
    torch::PackedTensorAccessor32<scalar_t, 2> ans,    // B, N
    torch::PackedTensorAccessor32<int32_t, 2> ans_indexes, // B, 2
    float interp_prob) {
  const int B = cumsum.size(0),
      N = cumsum.size(1);

  namespace cg = cooperative_groups;
  cg::thread_group group_tile = cg::tiled_partition(cg::this_thread_block(),
                                                    blockDim.x);

  const int b = threadIdx.y + blockIdx.y * blockDim.y;

  // group_cumsum_buf is a pointer to a buffer of shared memory, that is used by
  // this group `blockDim.x` threads, of size N, which will be used to
  // store this row of `cumsum`.  Each such group of threads points to
  // a different section of the buffer.
  scalar_t *group_cumsum_buf = ((scalar_t*)extern_buf) + (N * threadIdx.y);

  // Load 'cumsum'
  if (b < B) {
    for (int i = threadIdx.x; i < N; i += blockDim.x)
      group_cumsum_buf[i] = cumsum[b][i];
  }
  assert (blockDim.y <= 16);
  __shared__ int i_vec[16][2];

  int i1, i2;

  if (threadIdx.x < 2 && b < B) {
    scalar_t r = rand[b][threadIdx.x];
    ans_indexes[b][threadIdx.x] = i_vec[threadIdx.y][threadIdx.x] = find_class(cumsum[b], r);
  }

  group_tile.sync();

  if (b < B) {
    // set 'group_cumsum_buf' to zero; we'll later copy this to
    // our row of 'ans'.
    for (int i = threadIdx.x; i < N; i += blockDim.x)
      group_cumsum_buf[i] = 0.0;

    if (threadIdx.x == 0) {
      i1 = i_vec[threadIdx.y][0];
      i2 = i_vec[threadIdx.y][1];
      float lower_bound = (1.0 - interp_prob) * 0.5,
          upper_bound = (1.0 + interp_prob) * 0.5;
      scalar_t d = rand[b][2],
          e = (d < lower_bound ? 0.0 :
               (d > upper_bound ? 1.0 :
                ((d - lower_bound) / interp_prob)));
      if (i1 == i2) {
        group_cumsum_buf[i1] = 1.0;
      } else {
        group_cumsum_buf[i1] = 1.0 - e;
        group_cumsum_buf[i2] = e;
      }
    }
  }
  group_tile.sync();
  if (b < B) {
    // Copy 'group_cumsum_buf' to our row of 'ans'
    for (int i = threadIdx.x; i < N; i += blockDim.x)
      ans[b][i] = group_cumsum_buf[i];
  }
}


/*
  Summing reduction within a warp or part of a warp.
  This uses only the threadIdx.x, it assumes a group
  of threads with the same threadIdx.{y,z} are co-operating
  in the reduction.

  Args:
       buf:              Pointer to the start of a __shared__ buffer of size
                         blockDim.x, to be used as a temporary within this function.
                         We assume that each group of threads that
                         co-operates, provides a separate buffer pointer.
       val:              The value to be summed
  Return:
       All threads return the sum.
 */
template <typename scalar_t>
__forceinline__ __device__ scalar_t within_tile_reduce_sum(__volatile__ scalar_t *buf,
                                                           cooperative_groups::thread_group tile,
                                                           scalar_t val) {
  // Each iteration halves the number of active threads Each thread adds its
  // partial sum[i] to sum[lane+i]
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    buf[threadIdx.x] = val;
    tile.sync();
    if (threadIdx.x < i)
      val += buf[threadIdx.x + i];
  }
  if (threadIdx.x == 0) {
    buf[0] = val;
    tile.sync();
  }
  return buf[0];  // All threads return the summed value.
}



/*
  Backward kernel for flow sampling.  Please see the corresponding CPU code in
  flow_sampling_cpu.cpp to understand the algorithm and inputs.

  The block and grid dimensions are the same as we documented above for
  flow_sampling_kernel(); however, it needs a little more shared memory
  to be allocated.

  When this kernel is invoked, the suer must specify the amount of shared memory
  to be allocated, via `extern_buf`.  This must be enough to store
    ((N + blockDim.x) * blockDim.y) * sizeof(scalar_t)
  bytes.

 */
template <typename scalar_t>
__global__
void flow_sampling_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> cumsum,   // B, N
    torch::PackedTensorAccessor32<scalar_t, 2> rand,   // B, 3
    torch::PackedTensorAccessor32<scalar_t, 2> ans_grad,    // B, N
    torch::PackedTensorAccessor32<int32_t, 2> ans_indexes, // B, 2
    torch::PackedTensorAccessor32<scalar_t, 2> logits_grad,  // B, N.  This is an output.
    float interp_prob,
    float straight_through_scale) {

  const int B = cumsum.size(0),
      N = cumsum.size(1);

  const int b = threadIdx.y + blockIdx.y * blockDim.y;

  // group_probs_buf is a pointer to a buffer of shared memory, that is used by
  // this group `blockDim.x` threads, of size N.
  // Each such group of threads points to
  // a different section of the buffer.
  scalar_t *group_probs_buf = ((scalar_t*)extern_buf) + ((N + blockDim.x) * threadIdx.y);
  // group_reduce_buf is of size blockDim.x (e.g. 16); each group
  // of `blockDim.x` threads points to a different one.
  scalar_t *group_reduce_buf = group_probs_buf + N;

  namespace cg = cooperative_groups;
  cg::thread_group group_tile = cg::tiled_partition(cg::this_thread_block(),
                                                    blockDim.x);

  // N_ceil is N rounded up to a multiple of blockDim.x, so
  // all threads go round the loop below the same number of times,
  // because we have a sync() inside the loop.
  int N_ceil = ((N + blockDim.x - 1) / blockDim.x) * blockDim.x;


  if (b < B) {
    // Load 'group_probs_buf'.
    for (int i = threadIdx.x; i < N; i += blockDim.x)
      group_probs_buf[i] = cumsum[b][i];

    // Turn cumsum into probs by subtracting..
    // need to use N_ceil so all threads in the group do sync()
    for (int i = N_ceil - blockDim.x + threadIdx.x; i >= 0; i -= blockDim.x) {
      group_tile.sync();
      if (i < N) {
        scalar_t prob = group_probs_buf[i] - (i == 0 ? 0.0 : group_probs_buf[i-1]);
        group_probs_buf[i] = prob;
      }
    }
    group_tile.sync();
  }

  scalar_t z_grad = 0.0;
  // z_grad only needs to be set if straight_through_scale != 0.0
  if (b < B && straight_through_scale != 0.0) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      scalar_t prob = group_probs_buf[i],
          o_grad = ans_grad[b][i];
      // Implements: z_grad = - (\sum_i p(i) o_grad(i)
      z_grad -= prob * o_grad;
    }
    // all threads get the sum of their group.
    z_grad = within_tile_reduce_sum(group_reduce_buf, group_tile, z_grad);
  }

  int32_t i1, i2;
  scalar_t d;
  if (b < B) {
    i1 = ans_indexes[b][0];
    i2 = ans_indexes[b][1];
    d = rand[b][2];

    float lower_bound = (1.0 - interp_prob) * 0.5,
        upper_bound = (1.0 + interp_prob) * 0.5;

    scalar_t e_grad = ans_grad[b][i2] - ans_grad[b][i1],
        d_grad = (d < lower_bound || d > upper_bound ?
                  0.0 : e_grad / interp_prob),
        d2m1 = (2.0 * d - 1.0);

    for (int i = threadIdx.x; i < N_ceil; i += blockDim.x) {
      scalar_t l_grad_i;
      if (i < N) {
        scalar_t prob = group_probs_buf[i];
        // Don't bother loading ans_grad if straight_through_scale == 0.0
        scalar_t o_grad = straight_through_scale != 0.0 && b < B ? ans_grad[b][i] : 0.0;

        // Implements:
        // l_grad(i) := p(i) * (straight_through_scale * (o_grad(i) + z_grad) +
        //                     (1-straight_through_scale) * d_grad * (2d - 1))
        l_grad_i  = prob * (straight_through_scale * (o_grad + z_grad) +
                                  d_grad * (d2m1 * (1.0 - straight_through_scale)));
      }
      group_tile.sync();
      if (i < N) {
        // Really doing:
        // this_logits_grad_a[i] = l_grad_i
        // but re-using group_probs_buf for it.
        group_probs_buf[i] = l_grad_i;
      }
    }
    if (threadIdx.x == 0) {
      // Implements:
      // l_grad(i1) -= (1-straight_through_scale) * d_grad * d;
      // l_grad(i2) += (1-straight_through_scale) * d_grad * (1-d)
      // Note: there is no point in this code if i1 == i2 (in this case, anyway,
      // d_grad == 0); but it's harmless.
      group_probs_buf[i1] -= (1.0 - straight_through_scale) * d_grad * d;
      group_probs_buf[i2] += (1.0 - straight_through_scale) * d_grad * (1.0 - d);
    }

    // Write logits_grad
    for (int i = threadIdx.x; i < N; i += blockDim.x)
      logits_grad[b][i] = group_probs_buf[i];
  }
}




// See flow_sampling_cpu() in flow_sampline_cpu.cpp, for documentation.
std::vector<torch::Tensor> flow_sampling_cuda(torch::Tensor cumsum,
                                              torch::Tensor rand,
                                              float interp_prob) {
  TORCH_CHECK(cumsum.dim() == 2, "cumsum must be 2-dimensional");
  TORCH_CHECK(rand.dim() == 2, "rand must be 2-dimensional");
  TORCH_CHECK(cumsum.size(0) == rand.size(0) &&
              rand.size(1) == 3, "rand has unexpected shape");
  TORCH_CHECK(interp_prob > 0.0 && interp_prob <= 1.0);
  TORCH_CHECK(cumsum.device().is_cuda() && rand.device().is_cuda(),
              "inputs must be CUDA tensors");

  auto scalar_type = cumsum.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_type).device(cumsum.device());

  const int B = cumsum.size(0),
      N = cumsum.size(1);

  torch::Tensor ans = torch::empty({B, N}, opts);

  auto int32_opts = torch::TensorOptions().dtype(torch::kInt32).device(cumsum.device());

  torch::Tensor ans_indexes = torch::empty({B, 2}, int32_opts);


  // Always use 16 x 16 thread-blocks for now.
  int threads_per_group = 16,
      groups_per_block = 16,
      num_blocks = (B + groups_per_block - 1) / groups_per_block;

  dim3 blockDim(threads_per_group, groups_per_block, 1),
      gridDim(1, num_blocks, 1);

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "flow_sampling_cuda_stub", ([&] {
        int extern_memory_bytes = groups_per_block * N * sizeof(scalar_t);
        flow_sampling_kernel<scalar_t><<<gridDim, blockDim, extern_memory_bytes, at::cuda::getCurrentCUDAStream()>>>(
            cumsum.packed_accessor32<scalar_t, 2>(),
            rand.packed_accessor32<scalar_t, 2>(),
            ans.packed_accessor32<scalar_t, 2>(),
            ans_indexes.packed_accessor32<int32_t, 2>(),
            interp_prob);
      }));
  return std::vector<torch::Tensor>({ans, ans_indexes});
}



/*
  Backward of flow sampling, please see flow_sampling_cpu.cpp for more
  documentation.

  Returns logits_grad.
*/
torch::Tensor flow_sampling_backward_cuda(
    torch::Tensor cumsum,
    torch::Tensor rand,
    torch::Tensor ans_indexes,
    torch::Tensor ans_grad,
    float interp_prob,
    float straight_through_scale) {
  TORCH_CHECK(cumsum.dim() == 2, "cumsum must be 2-dimensional");
  TORCH_CHECK(rand.dim() == 2, "rand must be 2-dimensional");
  TORCH_CHECK(ans_indexes.dim() == 2, "ans_indexes must be 2-dimensional");
  TORCH_CHECK(ans_grad.dim() == 2, "ans_grad must be 2-dimensional");
  TORCH_CHECK(interp_prob > 0.0 && interp_prob <= 1.0);
  TORCH_CHECK(straight_through_scale >= 0.0 && straight_through_scale <= 1.0);


  auto scalar_type = cumsum.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_type).device(cumsum.device());

  const int B = cumsum.size(0),
      N = cumsum.size(1);

  TORCH_CHECK(rand.size(0) == B && rand.size(1) == 3);
  TORCH_CHECK(ans_indexes.size(0) == B && ans_indexes.size(1) == 2);
  TORCH_CHECK(ans_grad.size(0) == B && ans_grad.size(1) == N);

  TORCH_CHECK(cumsum.device().is_cuda() && rand.device().is_cuda() &&
              ans_indexes.device().is_cuda() && ans_grad.device().is_cuda());

  // We compute the derivative w.r.t. the original logits (meaning:
  // un-normalized logprobs), even though the input to the original function
  // was a processed form of the logits: the cumulative distribution of
  // the class probabilities, derived from the logits.
  torch::Tensor logits_grad = torch::empty({B, N}, opts);

  // Always use 16 x 16 thread-blocks for now.
  int threads_per_group = 16,
      groups_per_block = 16,
      num_blocks = (B + groups_per_block - 1) / groups_per_block;

  dim3 blockDim(threads_per_group, groups_per_block, 1),
      gridDim(1, num_blocks, 1);

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "flow_sampling_backward_cuda_stub", ([&] {
        int extern_memory_bytes =  groups_per_block * (N + threads_per_group) * sizeof(scalar_t);
        flow_sampling_backward_kernel<scalar_t><<<gridDim, blockDim, extern_memory_bytes, at::cuda::getCurrentCUDAStream()>>>(
            cumsum.packed_accessor32<scalar_t, 2>(),
            rand.packed_accessor32<scalar_t, 2>(),
            ans_grad.packed_accessor32<scalar_t, 2>(),
            ans_indexes.packed_accessor32<int32_t, 2>(),
            logits_grad.packed_accessor32<scalar_t, 2>(),
            interp_prob,
            straight_through_scale);
      }));
  return logits_grad;
}
