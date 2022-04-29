#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()

// We have included cub as a submodule as ../cub, and added the flag "-Icub" via
// setup.py.  This is fixed to the tag v1.8.0, since the current master gave us
// a compilation error; the choice of v1.8.0 was very random, just "an older
// version", hopefully old enough to be easy to compile but not so old as to be
// significantly worse.  In general it is not OK to use cub in a torch
// submodule, as it can lead to linking problems; however, I am hoping that
// since we only use block-level functionality (BlockScan), and not
// device-level, it will be used purely as a template library without causing
// any C-language global or namespace variables to be instantiated (I think
// those are what cause the compatibility problems with Torch).
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cmath>  // for INFINITY
#include <stdio.h>


extern __shared__ char extern_buf[];
// `extern_buf` is general-purpose shared memory.


/*
  Zoom into an interval.  What we are trying to emulate here is something like this:

  float cur_begin = 0.4, cur_end = 0.6,
         r = 0.423143, end = 0.7;
  r = (r - cur_begin) * end / (cur_end - cur_begin);
  .. that is, in the real-number version, we are taking some r with
  cur_begin <= r <= cur_end, and shifting and scaling it so that 0 <= r <= end.
  Look at this as "zooming into an interval", as in arithmetic coding.  This is
  all done in integer math, though.  Integer math is more convenient because
  we don't have to worry about roundoff here.

         r:       Random number with cur_begin <= r < cur_end.
    orig_r:       A number that we use as an auxiliary source of randomness, if
                  needed (e.g. if we have to zoom in "very far").
    cur_begin:    Beginning of interval that `r` is in.  We require
                     0 < cur_begin < (int)((1<<31) + 1.1)
    cur_end:      One-past-the-last of interval that `r` is in.  We require
                     cur_begin < cur_end < (int)((1<<31) + 1.1),
       end:      One past the last element of the interval that we are
                 shifting r to (first element is 0).

    Return:   Returns a number that, conceptually, is uniformly distributed on the
              interval [0..end-1], assuming r was originally uniformly distributed
              on the interval [cur_begin..cur_end-1].
 */
__forceinline__ __device__ uint32_t zoom(uint32_t r, uint32_t orig_r,
                                         uint32_t cur_begin, uint32_t cur_end,
                                         uint32_t end) {
  // prod will not overflow because none of these numbers are significantly larger
  // than 1 << 31.
  //
  // Adding ((orig_r + cur_begin) % end) is intended to provide randomness in
  // the lower bits, to avoid going to exactly zero when cur_begin - cur_end ==
  // 1, which could happen (extremely rarely)
  uint64_t prod = (uint64_t)(r - cur_begin) * (uint64_t)end  +  ((orig_r + cur_begin) % end);
  // We know that prod < (cur_end - cur_begin) * end, because
  // (uint64_t)(r - cur_begin) * (uint64_t)end <= (cur_end - 1 - cur_begin) * end,
  //                        and (orig_r % end) < 1 * end.

  // We know that ans < end, because prod < (cur_end - cur_begin) * end.
  uint32_t ans = prod / (cur_end - cur_begin);
  return ans;
}


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
__forceinline__ __device__ int find_class(
    cooperative_groups::thread_block_tile<32> &g,
    int32_t *shared_int,
    uint32_t *cumsum, int begin, int end, uint32_t r) {
  assert(end > begin);
  int orig_begin = begin, orig_end = end;  // debug

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

    if (!(begin >= orig_begin && begin < end && end <= orig_end)) {
      printf("[failure!]blockIdx.x=%d, threadIdx.{x,y}=%d,%d, orig begin,end=%d,%d, begin=%d, x,r,y=%d,%d,%d\n", blockIdx.x, threadIdx.x, threadIdx.y,
             orig_begin, orig_end, begin,
             (begin >= orig_begin && begin < orig_end ? cumsum[begin] : 10000), r,
             (begin + 1 >= orig_begin && begin + 1 < orig_end ? cumsum[begin + 1] : 100000));
      assert(0);
    }
  }
#if 0
  if (blockIdx.x == 0 && threadIdx.y == 0) {
    printf("blockIdx.x=%d, threadIdx.{x,y}=%d,%d, orig begin,end=%d,%d, returning begin=%d, x,r,y=%d,%d,%d\n", blockIdx.x, threadIdx.x, threadIdx.y,
           orig_begin, orig_end, begin,
           cumsum[begin], r, cumsum[begin + 1]);
  }
#endif
  if (!(r >= cumsum[begin] && (begin + 1 == orig_end || r < cumsum[begin + 1]))) {
    uint32_t y = (begin + 1 < orig_end ? cumsum[begin + 1] : 300000);
    printf("blockIdx.x=%d, threadIdx.{x,y}=%d,%d, search error:  begin,end=%d,%d, returning begin=%d, x,r,y=%d,%d,%d\n", blockIdx.x, threadIdx.x, threadIdx.y,
           orig_begin, orig_end, begin, cumsum[begin], r, y);
  }

  return begin;
}



/*
  kernel for sampling.

  One kernel handles one batch index (b) of 'cumsum', including all 's' indexes
  i.e. all sequences that correspond to that batch index.  As s gets larger
  (more sequences), we compensate by using fewer threads per sequence.
  Of course we have to process the 'k' index (the position in the sequence)
  sequentially.

  The thread-block will be (blockDim.x, blockDim.y, blockDim.z) = (32, X, 1) where X is 2, 4 or 8.
  32 is the size of the  group of threads assigned to handle one sequence 0 <= s < S,
  and blockDim.y will handle different sequences 's' in parallel.   We make
  BLOCK_DIM_Y a template arg so we can use BlockScan.


  The grid size will be (gridDim.x <= B), and we'll iterate over the
  batch index b.


  Args:
      probs:  Accessor to the probabilities,
         of shape (B, N), i.e. (batch_size, num_classes).

      rand:  Accessor to a tensor of random integers, in range {0..1<<31 - 1}
          The shape if (B, S), i.e. (batch_size, num_sequences)

      indexes (an output): Accessor to a LongTensor of chosen indexes, of
          shape (B, S, K), i.e. (batch_size, num_sequences, seq_len)

  When this kernel is invoked, the user must specify the amount of shared memory
  to be allocated, via `extern_buf`.  The size of extern_buf, in bytes, must be:

     (N + 1) * sizeof(uint32_t)       <-- for cumsum_buf
     (K + 2) * block_dim_y * sizeof(uint32_t)   <-- for cur_cumsum
   + (K + 2) * block_dim_y * sizeof(uint32_t)   <-- for cur_classes
   + block_dim_y * sizeof(uint32_t)             <-- for shared_int

*/
#define BLOCK_DIM_X 32
template <int BLOCK_DIM_Y>
__global__
void sampling_kernel(
    torch::PackedTensorAccessor32<float, 2> probs,   // B, N
    torch::PackedTensorAccessor32<int32_t, 2> rand,     // B, S
    torch::PackedTensorAccessor32<int64_t, 3> indexes) { // B, S, K
  using BlockScan = cub::BlockScan<uint32_t, 32, cub::BLOCK_SCAN_RAKING, BLOCK_DIM_Y>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int B = probs.size(0),
      N = probs.size(1),
      S = indexes.size(1),
      K = indexes.size(2);

  namespace cg = cooperative_groups;

  cg::thread_block_tile<32> g = cg::tiled_partition<32>(cg::this_thread_block());

  // each block of "blockDim.x" threads handles one 's' index (one sequence)
  int s = threadIdx.y;

  // This buffer stores 0 and then one row of `cumsum`, converted to uint32_t.
  // it is indexed 0..N; it points to __shared__ memory.
  uint32_t *cumsum_buf = (uint32_t*)extern_buf;

  // each block handling a different sequence s has a different 'cur_cumsum'
  // buffer, of size K + 2.  This is a pointer to __shared__ memory.
  // We store one more element in this buffer, vs. CPU code, that is never
  // actually needed but makes the code more similar to the 'cur_classes'
  // buffer and avoids if-statements.
  uint32_t *cur_cumsum = (cumsum_buf + N + 1) + (K + 2) * s;

  // each block handling a different sequence s has a different 'cur_classes'
  // buffer, of size (K + 2).  This is a pointer to __shared__ memory.
  int32_t *cur_classes = (int32_t*)((cumsum_buf + N + 1 + ((K + 2) * BLOCK_DIM_Y)) + (K + 2) * s);

  // `shared_int` is a pointer to a single __shared__ integer that is shared
  // within each tile of blockDim.x threads.
  int32_t *shared_int =  (int32_t*)((cumsum_buf + N + 1 + ((K + 2) * BLOCK_DIM_Y)) + (K + 2) * BLOCK_DIM_Y + s);

  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    // sample_cpu() in sampling_cpu.cpp may be helpful for
    // understanding this code.  This is essentially a translation of that code
    // to CUDA.

    __syncthreads();
    int thread_xy_idx = threadIdx.x + threadIdx.y * BLOCK_DIM_X;

    // load probs into cumsum_buf prior to doing exclusive-sum.
    for (int n = thread_xy_idx; n < N; n += BLOCK_DIM_X * BLOCK_DIM_Y) {
      // The + 1 is to ensure it's nonzero.  This is quite a bit smaller than
      // the floating point epsilon, so we're not concerned about the bias from
      // doing "+" instead of "max".
      cumsum_buf[n] = static_cast<uint32_t>((((uint32_t)1)<<31) * probs[b][n]) + 1;
    }


    __syncthreads();


    {
      // This block does the exclusive-sum of cumsum_buf.

      // Because N is probably a power of 2 and going over N for the exclusive sum might be
      // significantly wasteful, we do the exclusive sum up to N only, and treat the N'th
      // element specially at the end of this block.
      int items_per_thread =  N + (BLOCK_DIM_X * BLOCK_DIM_Y - 1) / BLOCK_DIM_X * BLOCK_DIM_Y;

      // Each thread is responsible for `items_per_thread` successive items;
      // first compute that partial sum.
      uint32_t start_idx = thread_xy_idx * items_per_thread,
          this_thread_tot = 0;
      for (int i = 0; i < items_per_thread; i++) {
        // j ranges over the same indexes as i but in a different order; this is
        // intended to avoid bank conflict for shared memory.
        int j = (i + thread_xy_idx) % items_per_thread,
            this_idx = start_idx + j;
        if (this_idx < N)
          this_thread_tot += cumsum_buf[this_idx];
      }

      BlockScan(temp_storage).ExclusiveSum(this_thread_tot, this_thread_tot);

      // OK, now 'this_thread_tot' contains the sum of the 'this_thread_tot'
      // values from all lower-indexed threads (i.e. those with lower
      // thread_xy_idx).
      int i;
      for (i = 0; i < items_per_thread && start_idx + i < N; i++) {
        uint32_t this_prob = cumsum_buf[start_idx + i];
        cumsum_buf[start_idx + i] = this_thread_tot;
        this_thread_tot += this_prob;
      }
      if (start_idx + i == N)
        cumsum_buf[N] = this_thread_tot;
      __syncthreads();
    }

    for (int s = threadIdx.y; s < S; s += BLOCK_DIM_Y) {
      g.sync();
      if (threadIdx.x < 2) {
        cur_cumsum[threadIdx.x] = (threadIdx.x == 0 ? 0 : cumsum_buf[N]);
        cur_classes[threadIdx.x] = (threadIdx.x == 0 ? -1 : N);
      }
      g.sync();

      uint32_t r = rand[b][s],  // 0 <= rand < (1 << 31)
          r_orig = r,
          remaining_prob = cumsum_buf[N];
      // at iteration k, remaining_prob contains the sum of the probabilities
      // of classes that have not so far been chosen.


      r = zoom(r, r, (uint32_t)0, ((uint32_t)1) << 31, remaining_prob);

      auto indexes_a = indexes[b][s];

      for (int k = 0; k < K; ++k) {
        assert(r < remaining_prob);

        int i = find_class(g, shared_int, cur_cumsum, 0, k + 1, r);
        assert(i >= 0 && i <= k);

        // i will now be (in all threadsin the tile), some value 0 <= i <= k, satisfying
        // cur_cumsum[i] <= r < cur_cumsum[i+1], where,
        // implicitly, cur_cumsum[k+1] == remaining_prob, although
        // actually we never access that element and it is not present.

        // class_range_begin, class_range_end, are the (first,
        // one-past-the-last) class indexes of the range of classes know
        // the k'th randomly chosen class is in.  See the comment above
        // about the k+1 intervals.
        int class_range_begin = cur_classes[i] + 1,
            class_range_end = cur_classes[i + 1];

        assert((class_range_begin >= 0 && class_range_begin < class_range_end &&
                class_range_end <= N));


        // shift r by "adding back" the probability mass due to the subset
        // of previously chosen classes that were numbered less than
        // class_range_begin.  Now r can be compared to elements of
        // `cumsum`.
        uint32_t class_range_begin_cumsum = cumsum_buf[class_range_begin];
        r = r - cur_cumsum[i] + class_range_begin_cumsum;

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

        uint32_t this_class_cumsum = cumsum_buf[c],
            this_class_next_cumsum = cumsum_buf[c + 1],
            this_class_prob = this_class_next_cumsum - this_class_cumsum;

        remaining_prob -= this_class_prob;

        r = zoom(r, r_orig, this_class_cumsum, this_class_next_cumsum, remaining_prob);

        {
          // This block updates cur_classes and cur_cumsum.  Normally the loop
          // below will execute just once; it is a loop because we want to handle
          // the (unusual) case where K > BLOCK_DIM_X.
          int num_threads_needed = k + 2 - i,
              num_iters = (num_threads_needed + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

          for (int iter = num_iters - 1; iter >= 0; --iter) {
            // `this_k` covers at least [i..i+num_threads_needed-1] == [i..k+1],
            // although we may also encounter this_k > k+1.  If it loops more than
            // once, it will go from the higher values of `this_k` to the lower
            // ones.
            int this_k = i + (iter * BLOCK_DIM_X) + threadIdx.x;

            // Caution: unlike most other variables in this code, c_temp and
            // cumsum_temp have different values in different threads in the tile g.
            int32_t c_temp;
            uint32_t cumsum_temp;

            if (this_k <= k + 1) {
              c_temp = cur_classes[this_k];
              assert(c_temp >= -1 && c_temp <= N);
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
      }
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
  sample function, CUDA version.

  probs: probabilities of input classes, of shape (B, N);
        where B is the batch size and N is the number of classes;
        must be in interval [0,1] and sum to (approximately) one.  Must be
        of float32 type, i.e. single precision float.

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

std::vector<torch::Tensor>
sample_combined_forward_cuda(torch::Tensor probs, // [B][N][M]
                             torch::Tensor rand,   // [B]
                             int K, bool input_is_log) {
  TORCH_CHECK(probs.dim() == 3, "probs must be 3-dimensional");
  TORCH_CHECK(rand.dim() == 1, "rand must be 1-dimensional");

  auto int64_type = torch::kInt64;
  TORCH_CHECK(rand.scalar_type() == int64_type);

  int B = probs.size(0),  // batch size
      N = probs.size(1),  // num distributions
      M = probs.size(2);  // num classes
  TORCH_CHECK(rand.size(0) == B);

  TORCH_CHECK(K > 0 && K < M && ((K&(K-1))==0));  // K is sequence length
  TORCH_CHECK(N >= 0 && N <= 4);
  TORCH_CHECK(rand.size(0) == B);

  TORCH_CHECK(probs.device().is_cuda() && rand.device().is_cuda(),
              "inputs must be CPU tensors");

  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(probs.device());
  auto real_opts = torch::TensorOptions().dtype(probs.dtype()).device(probs.device());

  // TODO: make empty
  torch::Tensor indexes = torch::empty({B, K, N}, long_opts),
      combined_indexes = torch::empty({B, K}, long_opts);

  torch::Tensor weights = torch::empty({B, K}, real_opts);


  int KpowN = K;
  for (int n = 1; n < N; n++)
    KpowN *= K;

  int32_t grid_dim_x = std::min<int>(B, 256),
      block_dim_x = std::max(M, KpowN);  // M will normally be larger.

  /*
  // HERE
  int extern_memory_bytes = ((N + 1) * sizeof(int32_t) +
                             (K + 2) * block_dim_y * sizeof(int32_t) +
                             (K + 2) * block_dim_y * sizeof(int32_t) +
                             block_dim_y * sizeof(int32_t));

  if (block_dim_y == 2) {
    sampling_kernel<2><<<gridDim, blockDim, extern_memory_bytes>>>(
        probs.packed_accessor32<float, 2>(),
        rand.packed_accessor32<int32_t, 2>(),
        indexes.packed_accessor32<int64_t, 3>());
  } else if (block_dim_y == 4) {
    sampling_kernel<4><<<gridDim, blockDim, extern_memory_bytes>>>(
        probs.packed_accessor32<float, 2>(),
        rand.packed_accessor32<int32_t, 2>(),
        indexes.packed_accessor32<int64_t, 3>());
  } else {
    sampling_kernel<8><<<gridDim, blockDim, extern_memory_bytes>>>(
        probs.packed_accessor32<float, 2>(),
        rand.packed_accessor32<int32_t, 2>(),
        indexes.packed_accessor32<int64_t, 3>());
  }
  cudaDeviceSynchronize();  // TEMP
  gpuErrchk(cudaGetLastError())
  */
  return std::vector<torch::Tensor>({indexes, combined_indexes, weights});
}
