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

template <typename IntT>
__device__ void print_array(IntT *buf, int num_items, const char *name) {
  if (threadIdx.x == 0) {
    printf("%s = [", name);
    for (int i = 0; i < num_items; i++) {
      printf("%f ",  (double)buf[i]);
    }
    printf("]\n");
  }
}


// c.f. https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
// A simple inclusive scan algorithm.  Require num_items <= blockDim.x.
template <typename IntT>
__device__ void simple_inclusive_scan(IntT *buf, int num_items) {
  print_array(buf, num_items, "simple-exclusive-scan-at-entry");
  for (int offset = 1; offset < num_items; offset *= 2) {
    IntT src_prev, src_cur;
    __syncthreads();
    if (threadIdx.x < num_items) {
      src_prev = (threadIdx.x >= offset ? buf[threadIdx.x - offset] : 0);
      src_cur = buf[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x < num_items) {
      buf[threadIdx.x] = src_prev + src_cur;
    }
  }
  __syncthreads();
  print_array(buf, num_items, "simple-exclusive-scan-at-exit");
  if (threadIdx.x < num_items && threadIdx.x > 0) {
    assert(buf[threadIdx.x-1] <= buf[threadIdx.x]);  //TEMP
  }
}




/*
  This function does a partial sort of an array, in reverse order, so that it's
  as if the array `start..start+input_size-1` is sorted, but we only care about
  the `start..start+num_keep-1` elements.

  see test_merge() in sorting_ref.py for the original Python code that this was
  based on.  This is not very elegant but is probably enough for now.

  Caution: this only works correctly if the elements x to be sorted satisfy
  x > (x-1) [not true, for example, for 0 in unsigned arithmetic];
  this is due to the subtraction of 1 in "x_val_mod" below.
 */
template <typename IntT>
__device__ void merge_based_partial_sort_reverse(
    IntT *buf, uint32_t num_keep, uint32_t num_items) {
  // num_keep == max_elements_needed in python.
  print_array(buf, num_items, "merge-sort-at-entry");
  __syncthreads();
  for (uint32_t new_sublist_size = 2;
       new_sublist_size <= num_items;
       new_sublist_size *= 2) {
    uint32_t old_sublist_size = new_sublist_size / 2;
    __syncthreads();
    uint32_t i = threadIdx.x,
        new_pos;
    IntT x_val;
    if (i < num_items) {
      x_val = buf[i];
      uint32_t offset_in_old = i & (old_sublist_size - 1);

      uint32_t new_sublist_start = i & ~(new_sublist_size - 1),
          is_rhs = (i & old_sublist_size), // equals old_sublist_size for right
                                                     // half of input
          other_list_start = new_sublist_start | (is_rhs ^ old_sublist_size),
          search_offset = other_list_start,
          search_begin = 0,
          search_end = std::min<uint32_t>(uint32_t(num_keep),
                                          old_sublist_size) + 1;

      IntT x_val_mod = x_val - (is_rhs != 0);
      while (search_begin + 1 < search_end) {
        uint32_t mid = (search_begin + search_end) / 2;
        // we are implementing reversed sorting, so replace the ">" in the
        // Python with "<".
        if (x_val_mod < buf[search_offset + mid - 1]) search_begin = mid;
        else search_end= mid;
      }
      new_pos = new_sublist_start + offset_in_old + search_begin;
    }
    __syncthreads();
    if (i < num_items) {
      buf[new_pos] = x_val;
    }
  }
  print_array(buf, num_items, "merge-sort-at-exit");
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

inline __device__ double Exp(double f) { return exp(f); }
template<typename Real>
inline __device__ Real Exp(Real f) { return Real(expf(float(f))); }

/*
  Return the index i into cumsum, with begin <= i < end,
  such that cumsum[i] <= r < cumsum[i + 1].
  We assume that cumsum[begin] <= r < cumsum[end], and we do not
  access cumsum[begin] or cumsum[end].
*/
template <typename IterType> __device__ int find_class_1thread(
    IterType cumsum, int begin, int end, uint32_t r) {
  TORCH_CHECK(end > begin);
  while (end > begin + 1) {
    int mid = begin + (end - begin) / 2;
    if (((uint32_t)cumsum[mid]) <= r)
      begin = mid;
    else
      end = mid;
  }
  return begin;
}



template<typename Real>
__global__
void sample_combined_forward_kernel(
    torch::PackedTensorAccessor32<Real, 3> probs,   // B, N, M
    torch::PackedTensorAccessor32<int64_t, 1> rand,     // B
    uint32_t K, bool input_is_log, uint32_t p_bits,
    uint32_t M_bits, uint32_t K_bits, uint32_t M_unique) {
  //__shared__ typename BlockScan::TempStorage temp_storage;
  int B = probs.size(0);  // batch size
  uint32_t N = probs.size(1),  // num distributions
      M = probs.size(2),  // num classes
      KpowN = 1 << (K_bits * N);

  // For now just use 1 thread, we'll gradually parallelize operations.  The
  // following arrays are all located in shared memory.  Trailing underscore
  // is for variables that, in class CombinedSampler in sampling_cpu.cpp, were
  // class member variables.
  uint64_t *topK_P_ = reinterpret_cast<uint64_t*>(extern_buf),  // [K]
      *topK_P_exclusive_sum_ = topK_P_ + K, // [K]
      *topK_delta_P_ = topK_P_exclusive_sum_ + K, // [K]
      *topK_cumsums_ = topK_delta_P_ + K, // [K]
      *sorted_topK_P_ = topK_cumsums_ + K, // [K]
      *sorted_topK_delta_P_ = sorted_topK_P_ + K, // [K]
      *sorted_topK_delta_P_cumsum_ = sorted_topK_delta_P_ + K, // [K]
      *sorted_topK_cumsums_ = sorted_topK_delta_P_cumsum_ + K, // [K]
      *sorted_topK_cumsums_reduced_ = sorted_topK_cumsums_ + K, // [K]
      *unreduced_samples_ = sorted_topK_cumsums_reduced_ + K; // [K]
  uint32_t *topK_indexes_ = reinterpret_cast<uint32_t*>(unreduced_samples_ + K), // [K*N]
      *P_cumsum_ = topK_indexes_ + (K*N); // [(M+1) * N]
  uint64_t *sort_buf64_ = reinterpret_cast<uint64_t*>(P_cumsum_ + (M+1) * N), // [K**N]
      *P_sum_cumprod_ = sort_buf64_ + KpowN,  // [N+1]
      *B_ptr_ = P_sum_cumprod_ + (N+1);  // [1]
  uint32_t *indexes_for_samples_ = reinterpret_cast<uint32_t*>(B_ptr_ + 1), // [K*N]
      *sort_buf32_ = indexes_for_samples_;  // [M+(N-1)*K], shares memory with indexes_for_samples

  if (threadIdx.x < N) {
    P_cumsum_[(M+1) * threadIdx.x] = 0;
  }

  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    // for now do everything in 1 thread, we'll gradually move to doing more
    // things in parallel to ease debugging.

    uint64_t rand_source = rand[b];


    { // LoadP()
      for (uint32_t n = 0; n < N; n++) {
        uint32_t multiple = 1 + M_unique * (rand_source >> (M_bits * n)) % (M / M_unique);
        // First load P linearly from global memory to shared memory.
        uint32_t m = threadIdx.x;
        if (m < M) {
          Real p = probs[b][n][m];
          if (input_is_log)
            p = Exp(p);
          // add 1 because if we allow zero probabilities, we get nasty edge cases.
          uint32_t P = uint32_t(1) + uint32_t((1 << p_bits) * p);
          P_cumsum_[n * (M+1) + m] = P;
        }
        // .. then pseudo-randomly reorder/shuffle P based on "multiple".
        __syncthreads();
        int32_t P;
        if (m < M) {
          uint32_t src_m = (m * multiple) % M;
          P = P_cumsum_[n * (M+1) + src_m];
        }
        __syncthreads();
        if (m < M) {
          P_cumsum_[n * (M+1) + m] = P;
        }
      }
    }

    for (uint32_t n = 0; n < N; n++) {
      print_array(P_cumsum_ + n*(M+1), M+1, "P_cumsum, prior to cumsum");
    }

    { // ComputeKLargest() in sampling_cpu.cpp.
      // This next loop populates sort_buf32_ with the top-K probabilities
      // for each source distribution.
      for (uint32_t n = 0; n < N; n++) {
        __syncthreads();

        uint32_t m = threadIdx.x;
        uint32_t P = P_cumsum_[n*(M+1) + m + 1];

        uint32_t *sort_buf = sort_buf32_ + (K*n);

        sort_buf[m] = (P << M_bits) + m;

        merge_based_partial_sort_reverse(sort_buf, K, M);

        // in CUDA we'll just sort the entire array.  Note, we don't need the
        // sorting algorithm to sort the indexes because we include them manually.
        //std::nth_element(sort_buf, sort_buf + K, sort_buf + M, std::greater<>());
        //std::sort(sort_buf, sort_buf + K, std::greater<>());
      }

      // Now we will iteratively compute the top-K *combined* probabilities,
      // starting by combining n=0 with n=1 and then including n=2 and so on one by one
      // if N>2.
      uint64_t *sort_combinations = sort_buf64_;
      uint32_t n = 0;
      __syncthreads();
      if (threadIdx.x < K) {
        uint32_t k = threadIdx.x,
            this_P = (sort_buf32_[K*n + k] >> M_bits); // one of the k-best probs for the n'th softmax
        // we include the index k, but leave space for N such indexes.
        sort_combinations[k] = (((uint64_t)this_P) << (N * K_bits)) | k;
      }
      print_array(sort_combinations, K, "sort-combinations-n=0");
      for (n = 1; n < N; n++) {
        __syncthreads();

        uint64_t K_mask = (uint64_t(1) << uint64_t(n * K_bits)) - 1;
        uint64_t new_S;
        if (threadIdx.x < K * K) {
          uint32_t best_k = threadIdx.x % K,
              new_k = threadIdx.x / K;
          // best_k is an index into the 1st K elements of array
          // `sort_combinations`
          uint64_t S = sort_combinations[best_k],
              P = S & ~K_mask,
              prev_ks = S & K_mask;
          uint64_t combined_k = prev_ks | (new_k << (n * K_bits));
          // one of the k-best probs for the n'th softmax
          uint32_t this_P = (sort_buf32_[K*n + new_k] >> M_bits);
          new_S = (P * this_P) | combined_k;
        }
        __syncthreads();
        if (threadIdx.x < K * K) {
          sort_combinations[threadIdx.x] = new_S;
        }
        merge_based_partial_sort_reverse(sort_combinations, K, K*K);
      }
      if (N == 1) {
        merge_based_partial_sort_reverse(sort_combinations, K, K);
      }

      print_array(sort_combinations, K, "sort-combinations-n=N");

      uint32_t M_mask = (1 << M_bits) - 1; // M may not be a power of 2, can't use M-1.
      if (threadIdx.x < K) {
        uint32_t k = threadIdx.x;
        uint64_t combination = sort_combinations[k],
            P = combination >> (K_bits * N);
        topK_P_[k] = P;
      }
      if (threadIdx.x < (K*N)) {
        uint32_t k = threadIdx.x % K,
            n = threadIdx.x / K;
        uint64_t combination = sort_combinations[k];
        // src_k is the k index among the top-K source items for this `n`.  We
        // need to look up the original 'm' index
        uint32_t src_k = (combination >> (n * K_bits)) & (K-1),
            src_m = sort_buf32_[K*n + src_k] & M_mask;
        topK_indexes_[k*N + n] = src_m;
      }

      if (threadIdx.x < K) {
        uint32_t k = threadIdx.x;
        topK_P_exclusive_sum_[k] = (k == 0 ? 0 : topK_P_[k-1]);
      }
      simple_inclusive_scan(topK_P_exclusive_sum_, K);

      print_array(topK_indexes_, K, "topK_indexes");
      print_array(topK_P_exclusive_sum_, K, "topK_P_exclusive_sum");
      print_array(topK_P_, K, "topK_P_");
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


// returns num_bits >= 1 such that (1 << num_bits) >= n.
inline int FindNumBitsFor(int n) {
  int num_bits = 1;
  while ((int(1) << num_bits) < n)
    num_bits++;
  return num_bits;
}


uint32_t FindProdUniquePrimeFactors(uint32_t i) { // returns smallest number coprime to
  TORCH_CHECK(i != 0);
  uint32_t ans = 1;
  for (uint32_t p = 2; i != 1; p++) {
    if (i % p == 0) {
      ans *= p;
      i /= p;
      while (i % p == 0)
        i /= p;  // divide by duplicates of this factor.
    }
  }
  return ans;  // product of unique prime factors of i, e.g. 4 -> 2, 18 -> 6.
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


  // work out p_bits, etc., on the CPU as it's a slightly complicated formula
  int M_bits = FindNumBitsFor(M),
      K_bits = FindNumBitsFor(K),
      p_bits = std::min(54 / N, // so product of N of these is
                                         // comfortably less than 64, search for
                                         // "headroom"
                        std::min((63/N) - K_bits,  // for when we sort `sort_combinations`.
                                 31 - M_bits)), // for when we sort `sort_buf` in ComputeKLargest()
      M_unique(FindProdUniquePrimeFactors(M));


  // TODO: allocate buffers..
  int KpowN = 1 << (K_bits * N),
      size64 = sizeof(uint64_t),
      size32 = sizeof(uint32_t);
  int grid_dim_x = std::min<int>(B, 256),
      block_dim_x = std::max(M, KpowN);  // M will normally be larger.



  // the order in which we list these buffers differs from their declaration ordere,
  // because we are trying to keep the expressions for working out addresses, generally
  // as simple as possible (since the CUDA code would have to do this).
  int extern_memory_bytes = size64 * K + // topK_P_
      size64 * K + // topK_P_exclusive_sum_
      size64 * K + // topK_delta_P_
      size64 * K + // topK_cumsums_
      size64 * K + // sorted_topK_P_
      size64 * K + // sorted_topK_delta_P_
      size64 * K + // sorted_topK_delta_P_cumsum_
      size64 * K + // sorted_topK_cumsums_
      size64 * K + // sorted_topK_cumsums_reduced_
      size64 * K + // unreduced_samples_
      size32 * (K*N) + // topK_indexes_
      size32 * (M+1) * N + // P_cumsum_
      size64 * KpowN + // sort_buf64_, because double the element size
      size64 * (N+1) +  // P_sum_cumprod_
      size64 * 1 + // B_.
      size32 * std::max<int>((M+(N-1)*K), // sort_buf32_,
                             K*N); // indexes_for_samples_


  AT_DISPATCH_FLOATING_TYPES(probs.scalar_type(), "sample_combined_cpu_forward_dispatch", ([&] {
        // scalar_t is defined by the macro AT_DISPATCH_FLOATING_TYPES
        sample_combined_forward_kernel<scalar_t><<<grid_dim_x, block_dim_x, extern_memory_bytes>>>(
            probs.packed_accessor32<scalar_t, 3>(),
            rand.packed_accessor32<int64_t, 1>(),
            K, input_is_log, p_bits, M_bits, K_bits, M_unique);
      }));
  gpuErrchk(cudaGetLastError());

  return std::vector<torch::Tensor>({indexes, combined_indexes, weights});
}
