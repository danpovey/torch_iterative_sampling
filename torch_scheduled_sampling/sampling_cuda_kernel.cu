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

  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    if (threadIdx.x == 0) {
      // for now do everything in 1 thread, we'll gradually move to doing more
      // things in parallel to ease debugging.

      uint64_t rand_source = rand[b];


      { // LoadP()
        for (uint32_t n = 0; n < N; n++) {
          uint32_t multiple = 1 + M_unique * (rand_source >> (M_bits * n)) % (M / M_unique);

          uint32_t *this_P_cumsum = P_cumsum_ + n * (M+1);
          this_P_cumsum[0] = 0;
          uint32_t p_multiple = 1 << (p_bits);
          for (uint32_t m = 0; m < M; m++) {
            uint32_t src_m = (m * multiple) % M;
            Real src_p = probs[b][n][src_m];
            if (input_is_log)
              src_p = Exp(src_p);

            // add 1 because if we allow zero probabilities, we get nasty edge cases.
            uint32_t P = uint32_t(1) + uint32_t(p_multiple * src_p);

            // the index is m + 1 because we shift it right by 1, this will be convenient when
            // creating the exclusive-sum.
            // The shifting left by M_bits and adding m is a temporary thing so that we
            // can later sort these probabilities but retain the associated indexes.
            this_P_cumsum[m + 1] = P;
          }
        }
      }

      { // ComputeKLargest() in sampling_cpu.cpp.
        for (uint32_t n = 0; n < N; n++) {
          uint32_t *this_P_buf = P_cumsum_ + (n * (M+1)) + 1;
          // Each time we access this buffer we shift right by
          // K, because we need to remember the top-K items for each n.
          uint32_t *sort_buf = sort_buf32_ + (K * n);
          for (uint32_t m = 0; m < M; m++) {
            uint32_t p = this_P_buf[m];
            sort_buf[m] = m + (p << M_bits); // keep track of indexes.
          }
          sort_buf[M] = 0;
          // in CUDA we'll just sort the entire array.  Note, we don't need the
          // sorting algorithm to sort the indexes because we include them manually.
          std::nth_element(sort_buf, sort_buf + K, sort_buf + M, std::greater<>());
          std::sort(sort_buf, sort_buf + K, std::greater<>());
        }
        uint64_t *sort_combinations = sort_buf64_;
        uint32_t K_bits_mask = (K-1);
        for (uint32_t i = 0; i < KpowN; i++) {  // we'll parallelize over i on GPU.
          // product of probabilities.  This index i represents an n-tuple of e.g. k1,k2,k3, which are all indexes in [0..K-1] specifying a
          uint64_t P_prod = 1;
          for (uint32_t n = 0; n < N; n++) {
            uint32_t k = (i >> (n * K_bits)) & K_bits_mask;  // the n'th index 0 <= k < K into K-best.
            uint32_t this_p = (sort_buf32_[K*n + k] >> M_bits); // one of the k-best probs for the n'th softmax
            P_prod *= this_p;
          }
          sort_combinations[i] = (P_prod << (K_bits * N)) + i;
        }
        // we'll just sort the entire array, when we do this on GPU.
        std::nth_element(sort_combinations, sort_combinations + K, sort_combinations + KpowN,
                         std::greater<>());
        // work out the K-best combinations.
        std::sort(sort_combinations, sort_combinations + K, std::greater<>());

        uint32_t M_mask = (1 << M_bits) - 1; // M may not be a power of 2, can't use M-1.
        for (uint32_t k = 0; k < K; k++) {  // we'll parallelize over k on GPU.
          uint64_t combination = sort_combinations[k],
              P = combination >> (K_bits * N);
          topK_P_[k] = P;
          for (uint32_t n = 0; n < N ; n++) {
            // src_k is the k index among the top-K source items for this `n`.  We
            // need to look up the original 'm' index
            uint32_t src_k = (combination >> (n * K_bits)) & (K-1),
                src_m = sort_buf32_[K*n + src_k] & M_mask;
            topK_indexes_[k*N + n] = src_m;
          }
        }
        uint64_t topK_P_sum = 0;
        for (uint32_t k = 0; k < K; k++) {  // this would be done using a cub exclusive-sum.
          topK_P_exclusive_sum_[k] = topK_P_sum;
          topK_P_sum += topK_P_[k];
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
