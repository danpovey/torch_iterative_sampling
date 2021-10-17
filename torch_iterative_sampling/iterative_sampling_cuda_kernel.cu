#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()
#include <cooperative_groups.h>
#include <cmath>  // for INFINITY
#include <stdio.h>

extern __shared__ char extern_buf[];
// `extern_buf` is general-purpose shared memory.



/*
  This device function, intended to be called with the same args among all threads
  in the tile `g` (actually a group of blockDim.x threads),
  returns, in all threads in the tile, the index i into cumsum, with begin <= i < end,
  such that cumsum[i] <= r < cumsum[i + 1], where cumsum[i + 1] is never accessed
  and is treated as if it were +infinity.  (TODO: see if this is really necessary).

  Args:
       g: represents the group of threads which will be calling this
                 function (for synchronization).
       shared_int:  A pointer to an int32_t in shared memory that is unique
              to this tile, to be used for inter-thread communication.
           cumsum: pointer to an array of scalar_t that we are searching in.
            begin: first index in `cumsum` that we might return
              end: one-past-the-last index in `cumsum` that we might return
                 (cumsum[end] will never be accessed and is treated as +infinity).
                 Must satisfy end > begin.
               r:  value whose interval we are searching for in `cumsum`

   Return:
       Returns the value of i, with begin <= i < end, such that
       cumsum[i] <= r < cumsum[i + 1], treating cumsum[end] as +infinity.
       This is undefined if r < cumsum[begin].

*/
template <typename scalar_t>
__forceinline__ __device__ int find_class(
    cooperative_groups::thread_group &g, int32_t *shared_int,
    scalar_t *cumsum, int begin, int end, scalar_t r) {
  int orig_begin=begin, orig_end=end;

  g.sync();
  int i = g.thread_rank(),  // Actually will equal threadIdx.x.
      tile_size = g.size();  // Actually will equal blockDim.x

  while (end > begin + 1) {
    // 'block_size' is the number of indexes that each thread is responsible for
    // at this stage of the computation.
    int block_size = (end - begin + tile_size - 1) / tile_size;

    // block_start and block_end are the (start,end) points of the
    // block of indexes that this thread is responsible for.
    int block_start = begin + i * block_size,
        block_end = block_start + block_size;
    if (block_start < end &&
        r >= cumsum[block_start] &&
        (block_end >= end || r < cumsum[block_end])) {
      // Exactly one thread will reach this point.
      *shared_int = block_start;
    }
    g.sync();
    // All other threads will read 'begin' from the thread that "succeeded".
    // Exactly one thread should "succeed", assuming the entry conditions for
    // this function are satisfied.  The key one is that r >= cumsum[begin],
    // at entry.
    begin = *shared_int;
    // we are only syncing within a tile, so the fact that different tiles
    // may go different numbers of times around this loop does not matter.
    end = min(end, begin + block_size);
  }
#if 0
  if (blockIdx.x == 0 && threadIdx.y == 0) {
    printf("blockIdx.x=%d, threadIdx.{x,y}=%d,%d, orig begin,end=%d,%d, returning begin=%d, x,r,y=%f,%f,%f\n", blockIdx.x, threadIdx.x, threadIdx.y,
           orig_begin, orig_end, begin,
           (float)cumsum[begin], (float)r, (float)cumsum[begin + 1]);
  }
#endif
  if (!(r >= cumsum[begin] && (begin + 1 == orig_end || r < cumsum[begin + 1]))) {
    printf("blockIdx.x=%d, threadIdx.{x,y}=%d,%d, search error:  begin,end=%d,%d, returning begin=%d, x,r,y=%f,%f,%f\n", blockIdx.x, threadIdx.x, threadIdx.y,
           orig_begin, orig_end, begin, (float)cumsum[begin], (float)r, (float)cumsum[begin + 1]);
  }

  return begin;
}


template <typename scalar_t>
__forceinline__ __device__ void wrap_if_outside_unit_interval(scalar_t *r) {
  if (*r > 1.0 || *r < 0.0) {
    // should be very rare.
    printf("iterative_sampling_cuda_kernel.cpp: warning: blockIdx.x=%d, threadIdx.{x,y}=%d,%d, wrapping %f\n",
           blockIdx.x, threadIdx.x, threadIdx.y, (float)(*r));
    // mathematically, r should still be in the range [0,1]; we wrap
    // around like this just in case of roundoff errors.
    if (*r < 0.0)
      *r = -*r;
    *r = (*r - (int)*r);
  }
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

  The grid size will be (gridDim.x <= B), and we'll iterate over the
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

  When this kernel is invoked, the user must specify the amount of shared memory
  to be allocated, via `extern_buf`.  The size of extern_buf, in bytes, must be:

      N * sizeof(scalar_t)           <-- for cumsum_buf
     (K + 2) * S * sizeof(scalar_t)  <-- for cur_cumsum
   + (K + 2) * S * sizeof(int32_t)   <-- for cur_classes
   + S * sizeof(int32_t)             <-- for shared_int

*/
template <typename scalar_t>
__global__
void iterative_sampling_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2> cumsum,   // B, N
    torch::PackedTensorAccessor32<scalar_t, 2> rand,     // B, S
    torch::PackedTensorAccessor32<int64_t, 3> indexes) { // B, S, K
  const int B = cumsum.size(0),
      N = cumsum.size(1),
      S = indexes.size(1),
      K = indexes.size(2);

  assert(S == blockDim.y);

  namespace cg = cooperative_groups;
  cg::thread_group g = cg::tiled_partition(cg::this_thread_block(),
                                           blockDim.x);

  // each block of "blockDim.x" threads handles one 's' index (one sequence)
  int s = threadIdx.y;
  assert(s < S);

  // This buffer stores one row of `cumsum`, indexed 0..N-1; it points to
  // __shared__ memory.
  scalar_t *cumsum_buf = (scalar_t*)extern_buf;

  // each block handling a different sequence s has a different 'cur_cumsum'
  // buffer, of size K + 2.  This is a pointer to __shared__ memory.
  // We store one more element in this buffer, vs. CPU code, that is never
  // actually needed but makes the code more similar to the 'cur_classes'
  // buffer and avoids if-statements.
  scalar_t *cur_cumsum = (cumsum_buf + N) + (K + 2) * s;

  // each block handling a different sequence s has a different 'cur_classes'
  // buffer, of size (K + 2).  This is a pointer to __shared__ memory.
  int32_t *cur_classes = (int32_t*)(cumsum_buf + N + ((K + 2) * S)) + (K + 2) * s;

  // `shared_int` is a pointer to a single __shared__ integer that is shared
  // within each tile of blockDim.x threads.
  int32_t *shared_int =  (int32_t*)(cumsum_buf + N + ((K + 2) * S)) + (K + 2) * S + s;

  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    // iterative_sample_cpu() in iterative_sampling_cpu.cpp may be helpful for
    // understanding this code.  This is essentially a translation of that code
    // to CUDA.

    __syncthreads();

    // load cumsum_buf
    for (int n = threadIdx.x + threadIdx.y * blockDim.x; n < N;
         n += blockDim.x * blockDim.y)
      cumsum_buf[n] = cumsum[b][n];

    __syncthreads();


    if (threadIdx.x < 2) {
      cur_cumsum[threadIdx.x] = (threadIdx.x == 0 ? 0.0 : 1.0);
      cur_classes[threadIdx.x] = (threadIdx.x == 0 ? -1 : N);
    }
    g.sync();

    scalar_t chosen_sum = 0.0,
        r = rand[b][s];

    auto indexes_a = indexes[b][s];

    for (int k = 0; k < K; ++k) {
      // Note: at this point, r is in the interval [0,1-chosen_sum]
      int i = find_class(g, shared_int, cur_cumsum, 0, k + 1, r);
      // i will now be (in all threadsin the tile), some value 0 <= i <= k, satisfying
      // cur_cumsum[i] <= r < cur_cumsum[i+1], where,
      // implicitly, cur_cumsum[k+1] == 1-chosen_sum, although
      // actually we never access that element and it is not present.

      // class_range_begin, class_range_end, are the (first,
      // one-past-the-last) class indexes of the range of classes know
      // the k'th randomly chosen class is in.  See the comment above
      // about the k+1 intervals.
      int class_range_begin = cur_classes[i] + 1,
          class_range_end = cur_classes[i + 1];

      if (!((class_range_begin >= 0 && class_range_begin < class_range_end &&
             class_range_end <= N))) {
        // It will be extremely rare to reach this point; it can happen due
        // to roundoff issues.

        // Will eventually delete this print statement.
        printf("[warning:]blockIdx.x=%d, threadIdx.{x,y}=%d,%d, class_range_begin=%d, class_range_end=%d, k=%d, i=%d, x=%g,r=%g,y=%g,r-x=%g,y-r=%g, (1-chosen_sum)-r=%g\n", blockIdx.x, threadIdx.x, threadIdx.y,
               class_range_begin, class_range_end, k, i,
               cur_cumsum[i], r, cur_cumsum[i+1], r-cur_cumsum[i], cur_cumsum[i+1]-r, (1-chosen_sum)-r);

        // Find the first position i that has a nonempty set of possible classes.
        for (i = 0; i <= k; i++) {
          if (cur_classes[i] + 1 < cur_classes[i + 1]) {
            r = 0.5 * (cur_cumsum[i] + cur_cumsum[i+1]);
            class_range_begin = cur_classes[i] + 1;
            class_range_end = cur_classes[i + 1];
            break;
          }
        }
      }

      // shift r by "adding back" the probability mass due to the subset
      // of previously chosen classes that were numbered less than
      // class_range_begin.  Now r can be compared to elements of
      // `cumsum`.
      scalar_t class_range_begin_cumsum = cumsum_buf[class_range_begin];
      scalar_t r_orig1 = r;
      r = r - cur_cumsum[i] + class_range_begin_cumsum;
      //assert(r >= 0 && r <= 1.0);

      assert(class_range_begin < class_range_end && class_range_end <= N);
      int c = find_class(g, shared_int,
                         cumsum_buf,
                         class_range_begin,
                         class_range_end, r);

      assert(c >= class_range_begin && c < class_range_end);

      // c is the class chosen, satisfying cumsum_buf[c] <= r <
      // cumsum_buf[c+1], where implicitly cumsum_buf[N] == 1.0.
      // It will be distinct from all previously chosen classes.
      if (threadIdx.x == 0) {
        indexes_a[k] = c;
      }

      scalar_t this_class_cumsum = cumsum_buf[c],
          this_class_prob = (c + 1 == N ? 1.0 : cumsum_buf[c + 1]) - this_class_cumsum;
      scalar_t r_orig = r;  //TEMP
      r = (r - this_class_cumsum) / this_class_prob;
      // mathematically, r should be in [0,1]; but make sure of this in case,
      // due to roundoff, it is just outside that interval.
      wrap_if_outside_unit_interval(&r);
      // We can now treat r as a "new" random value on [0,1].

      {
        // This block updates cur_classes and cur_cumsum.  Normally the loop
        // below will execute just once; it is a loop because we want to handle
        // the (unusual) case where K > blockDim.x.
        int num_threads_needed = k + 2 - i,
            num_iters = (num_threads_needed + blockDim.x - 1) / blockDim.x;

        for (int iter = num_iters - 1; iter >= 0; --iter) {
          // `this_k` covers at least [i..i+num_threads_needed-1] == [i..k+1],
          // although we may also encounter this_k > k+1.  If it loops more than
          // once, it will go from the higher values of `this_k` to the lower
          // ones.
          int this_k = i + (iter * blockDim.x) + threadIdx.x;

          // Caution: unlike most other variables in this code, c_temp and
          // cumsum_temp have different values in different threads in the tile g.
          int32_t c_temp;
          scalar_t cumsum_temp;

          if (this_k <= k + 1) {
            c_temp = cur_classes[this_k];
            cumsum_temp = cur_cumsum[this_k];
          }
          g.sync();
          if (this_k > i) {
            if (this_k <= k + 1) {
              cur_classes[this_k + 1] = c_temp;
              cur_cumsum[this_k + 1] = cumsum_temp - this_class_prob;
            }
          } else {  // this_k == i
            cur_classes[this_k + 1] = c;
            cur_cumsum[this_k + 1] = cumsum_temp + (this_class_cumsum - class_range_begin_cumsum);
          }
        }
      }

      chosen_sum += this_class_prob;
      // Reduce the random value r so that it is in the range
      // [0..1-chosen_sum], which is the size of the reduced interval
      // after subtracting the probability mass due to previously chosen
      // classes.  On the next iteration, we will search within this
      // reduced interval.
      r = r * (1.0 - chosen_sum);

#if 0
      if (blockIdx.x == 0 && threadIdx.y == 0) {
        printf("blockIdx.x=%d, threadIdx.{x,y}=%d,%d, r=%f, r_orig1=%f, r_orig=%f, chosen_sum=%f, this_class_prob=%f, this_class_cumsum=%f, class_range_begin_cumsum=%f, k=%d, i=%d, c=%d, class_range_begin=%d, class_range_end=%d\n", blockIdx.x, threadIdx.x, threadIdx.y,
               r, r_orig1, r_orig, chosen_sum, this_class_prob, this_class_cumsum, class_range_begin_cumsum, k, i, c, class_range_begin, class_range_end);
      }
#endif

    }
  }
}

// gpuErrchk is probably not ideal, likely Torch has something for this.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
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


  int32_t grid_dim_x = std::min<int>(B, 256),
      block_dim_y = S,
      block_dim_x = 32;

  // actually block_dim_x must be 32 because for now cooperative_groups
  // does not support tiles with size more than 32.
  /*
  while (block_dim_x * S < 256 && block_dim_x < N)
    block_dim_x *= 2;
  */


  TORCH_CHECK(K > 0 && K < N);  // K is sequence length
  TORCH_CHECK(rand.size(0) == B);

  TORCH_CHECK(cumsum.device().is_cuda() && rand.device().is_cuda(),
              "inputs must be CUDA tensors");


  auto scalar_type = cumsum.scalar_type();  // presumably float or double

  auto opts = torch::TensorOptions().dtype(scalar_type).device(cumsum.device()),
      long_opts = torch::TensorOptions().dtype(torch::kInt64).device(cumsum.device());

  torch::Tensor indexes = torch::empty({B, S, K}, long_opts);

  dim3 blockDim(block_dim_x, block_dim_y, 1),
        gridDim(grid_dim_x, 1, 1);

  //fprintf(stderr,"block_dim_x: %d, block_dim_y: %d, grid_dim_x: %d\n", block_dim_x,
  // block_dim_y, grid_dim_x);
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "iterative_sampling_cuda_stub", ([&] {
        // scalar_t is defined by the macro AT_DISPATCH_FLOATING_TYPES, should
        // equal scalar_type.
        int extern_memory_bytes = (N * sizeof(scalar_t) +
                                   (K + 2) * S * sizeof(scalar_t) +
                                   (K + 2) * S * sizeof(int32_t) +
                                   S * sizeof(int32_t));
        //        fprintf(stderr, "N = %d, K = %d, S = %d, extern_memory_bytes = %d\n", N, K, S, extern_memory_bytes);
        // extern_memory_bytes += 1024;

        iterative_sampling_kernel<scalar_t><<<gridDim, blockDim, extern_memory_bytes>>>(
            cumsum.packed_accessor32<scalar_t, 2>(),
            rand.packed_accessor32<scalar_t, 2>(),
            indexes.packed_accessor32<int64_t, 3>());
        cudaDeviceSynchronize();  // TEMP
        gpuErrchk(cudaGetLastError())
            }));
  return indexes;
}
