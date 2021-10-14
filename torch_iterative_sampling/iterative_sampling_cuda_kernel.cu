#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()
#include <cooperative_groups.h>
#include <cmath>  // for INFINITY
#include <stdio.h>

extern __shared__ int extern_buf[];
// `extern_buf` is general-purpose shared memory.



/*
  This device function, intended to be called with the same args among all threads
  in the tile `group_tile` (actually a group of blockDim.x threads),
  returns, in all threads in the tile, the index i into cumsum, with begin <= i < end,
  such that cumsum[i] <= r < cumsum[i + 1], where cumsum[i + 1] is never accessed
  and is treated as if it were +infinity.  (TODO: see if this is really necessary).

  Args:
         group_tile: represents the group of threads which will be calling this
                 function (for synchronization).
             cumsum: pointer to an array of ScalarType that we are searching in.
             begin: first index in `cumsum` that we might return
             end: one-past-the-last index in `cumsum` that we might return
               (cumsum[end] will never be accessed and is treated as +infinity).
               Must satisfy end > begin.
             r:  value whose interval we are searching for in `cumsum`
        shared_int:  A pointer to an int32_t in shared memory that is unique
               to this tile.



*/
template <typename ScalarType>
__forceinline__ __device__ int find_class(
    cooperative_groups::thread_group group_tile,
    ScalarType *cumsum, int begin, int end, ScalarType r, int32_t *shared_int) {

  group_tile.sync();
  int i = group_tile.thread_rank();  // Actually will equal threadIdx.x.
  if (begin + i < end) {


  }


  assert(end > begin);
  while (end > begin + 1) {
    int mid = begin + (end - begin) / 2;
    if (cumsum[mid] <= r)
      begin = mid;
    else
      end = mid;
  }
  return begin;
}



/*
  kernel for iterative_sampling.

  One kernel handles one batch index (b) of 'cumsum', including all 's' indexes
  i.e. all sequences that correspond to that batch index.  As s gets larger
  (more sequences), we compensate by using fewer threads per sequence.
  Of course we have to process the 'k' index (the position in the sequence)
  sequentially.

  The thread-block will be of shape  (blockDim.x, blockDim.y)
  where blockDim.x, a power of two (e.g. 16 or 32), is the size of the
  group of threads assigned to handle one sequence 0 <= s < S,
  and blockDim.y equals S.

  The grid size will be (gridDim.x), and we'll iterate over the
  batch index b.


  Template args:
      scalar_t: the floating-point type, e.g. float, double; maybe half

  Args:
      cumsum:  Accessor to the (exclusive) cumulative sum of probabilities,
            of shape (B, N), i.e. (batch_size, num_classes)
      rand:  Accessor to a tensor of random numbers, of shape (B, S),
         i.e. (batch_size, num_sequences)
      indexes (an output): Accessor to a LongTensor of chosen indexes, of
          shape (B, S, K), i.e. (batch_size, num_sequences, seq_len)

  When this kernel is invoked, the suer must specify the amount of shared memory
  to be allocated, via `extern_buf`.  The size of extern_buf must be:
  row of

     [TODO] * sizeof(scalar_t)
  bytes.
*/
template <typename scalar_t>
__global__
void iterative_sampling_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> cumsum,   // B, N
    torch::PackedTensorAccessor32<scalar_t, 2> rand,   // B, S
    torch::PackedTensorAccessor32<int64_t, 2> indexes, // S, K
    float interp_prob) {
  const int B = cumsum.size(0),
      N = cumsum.size(1),
      S = indexes.size(0),
      K = indexes.size(1);

  assert(S == blockDim.y);

  namespace cg = cooperative_groups;
  cg::thread_group group_tile = cg::tiled_partition(cg::this_thread_block(),
                                                    blockDim.x);


  int s = threadIdx.y;  // each block of "blockDim.x" threads handles one 's'
  // index (one sequence)

  // each block handling a different sequence s has a different 'cur_cumsum'
  // buffer, of size K.  This is a pointer to __shared__ memory.
  scalar_t *cur_cumsum = (cumsum_buf + N) + K * s;

  // each block handling a different sequence s has a different 'cur_classes'
  // buffer, of size (K + 1).  This is a pointer to __shared__ memory.
  int32_t *cur_classes = (int32_t*)(cumsum_buf + N + (K * S)) + (K + 1) * s;


  // `shared_int` is a pointer to a single __shared__ integer that is common to
  // each group of blockDim.x threads.
  int32_t *shared_int =  (int32_t*)(cumsum_buf + N + (K * S)) + (K + 1) * S + s;


  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    __syncthreads();
    scalar_t *cumsum_buf = ((scalar_t*)extern_buf);

    // load cumsum_buf
    for (int n = threadIdx.x + threadIdx.y * blockDim.x; n < N;
         n += blockDim.x * blockDim.y)
      cumsum_buf[n] = cumsum[b][n];

    __syncthreads();


    if (threadIdx.x < 2) {
      cur_cumsum[threadIdx.x] = 0.0;
      cur_classes[threadIdx.x] = (threadIdx.x == 0 ? -1 : N);
    }
    group_tile.sync();

    // iterative_sample_cpu() in iterative_sampling_cpu.cpp may be helpful for
    // understanding this code, where the contents of 'cur_classes' and
    // 'cur_cumsum' are documented.


    scalar_t chosen_sum = 0.0,
        r = rand[b][s];

    for (int k = 0; k < K; ++k) {


    }
  }
}


  // group_cumsum_buf is a pointer to a buffer of shared memory, that is used by
  // this thread-block to store a row of `cumsum`.
  group `blockDim.x` threads, of size N, which will be used to
  // store this row of `cumsum`.  Each such group of threads points to
  // a different section of the buffer.
  scalar_t *cumsum_buf = ((scalar_t*)extern_buf) + (N * threadIdx.y);

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
  iterative_sampling_cpu.cpp to understand the algorithm and inputs.

  The block and grid dimensions are the same as we documented above for
  iterative_sampling_kernel(); however, it needs a little more shared memory
  to be allocated.

  When this kernel is invoked, the suer must specify the amount of shared memory
  to be allocated, via `extern_buf`.  This must be enough to store
    ((N + blockDim.x) * blockDim.y) * sizeof(scalar_t)
  bytes.

 */
template <typename scalar_t>
__global__
void iterative_sampling_backward_kernel(
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
                                    int K) {
  TORCH_CHECK(cumsum.dim() == 2, "cumsum must be 2-dimensional");
  TORCH_CHECK(rand.dim() == 2, "rand must be 2-dimensional");

  int B = cumsum.size(0),  // batch size
      N = cumsum.size(1),  // num classes
      S = rand.size(1);    // num sequences

  TORCH_CHECK(K > 0 && K < N);  // K is sequence length
  TORCH_CHECK(rand.size(0) == B);

  TORCH_CHECK(cumsum.device().is_cuda() && rand.device().is_cuda(),
              "inputs must be CUDA tensors");


  auto scalar_type = cumsum.scalar_type();  // presumably float or double

  auto opts = torch::TensorOptions().dtype(scalar_type).device(cumsum.device()),
      long_opts = torch::TensorOptions().dtype(torch::kInt64).device(cumsum.device());

  torch::Tensor indexes = torch::empty({B, S, K}, long_opts);


  // Always use 16 x 16 thread-blocks for now.
  int threads_per_group = 16,
      groups_per_block = 16,
      num_blocks = (B + groups_per_block - 1) / groups_per_block;

  dim3 blockDim(threads_per_group, groups_per_block, 1),
      gridDim(1, num_blocks, 1);

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "iterative_sampling_cuda_stub", ([&] {
        int extern_memory_bytes = groups_per_block * N * sizeof(scalar_t);
        iterative_sampling_kernel<scalar_t><<<gridDim, blockDim, extern_memory_bytes, at::cuda::getCurrentCUDAStream()>>>(
            cumsum.packed_accessor32<scalar_t, 2>(),
            rand.packed_accessor32<scalar_t, 2>(),
            ans.packed_accessor32<scalar_t, 2>(),
            ans_indexes.packed_accessor32<int32_t, 2>(),
            interp_prob);
      }));
  return std::vector<torch::Tensor>({ans, ans_indexes});
}



/*
  Backward of flow sampling, please see iterative_sampling_cpu.cpp for more
  documentation.

  Returns logits_grad.
*/
torch::Tensor iterative_sampling_backward_cuda(
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

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "iterative_sampling_backward_cuda_stub", ([&] {
        int extern_memory_bytes =  groups_per_block * (N + threads_per_group) * sizeof(scalar_t);
        iterative_sampling_backward_kernel<scalar_t><<<gridDim, blockDim, extern_memory_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
