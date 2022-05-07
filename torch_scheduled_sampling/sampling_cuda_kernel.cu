#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // for getCurrentCUDAStream()

/*
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
#include <cub/cub.cuh>*/
#include <cooperative_groups.h>
#include <cmath>  // for INFINITY
#include <stdio.h>


extern __shared__ char extern_buf[];
// `extern_buf` is general-purpose shared memory.

template <typename IntT>
__device__ void print_array(IntT *buf, int num_items, const char *name) {
  /*
   __syncthreads();
  if (threadIdx.x == 0) {
    printf("%s = [", name);
    for (int i = 0; i < num_items; i++) {
      printf("%ld ",  (long int)buf[i]);
    }
    printf("]\n");
    } */
}


/*
  c.f. https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
  A simple inclusive scan algorithm.  Require num_items <= blockDim.x,
  num_items does not need to be a power of 2, or the same as
  blockDim.x, but must be <= blockDim.x.

  Assumes blockDim is of the form (blockDim.x, 1, 1)
*/
template <typename IntT>
__device__ void simple_inclusive_scan(IntT *buf, int num_items) {
  print_array(buf, num_items, "simple-inclusive-scan-at-entry");

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
  __syncthreads(); // TODO: remove..
  print_array(buf, num_items, "simple-inclusive-scan-at-exit");
  if (threadIdx.x < num_items && threadIdx.x > 0) {
    assert(buf[threadIdx.x-1] <= buf[threadIdx.x]);  //TEMP
  }
}

/*
  This is a wrapper for simple_inclusive_scan() that is to be called only in cases
  where num_items > blockDim.x.  Assumes blockDim is of the form (blockDim.x, 1, 1)

  Args:
     data: an array of length `num_items`, of items to be inclusive-scanned
      buf: an array of length at least `blockDim.x` that we can use temporarily.

 */
template <typename IntT>
__device__ void inclusive_scan(IntT *data, IntT *buf, int num_items) {
  IntT sum = 0;
  int block_size = (num_items + blockDim.x - 1) / blockDim.x;
  for (int i = 0; i < block_size; i++) {
    int idx = threadIdx.x * block_size + i;
    if (idx < num_items)
      sum += data[idx];
  }
  buf[threadIdx.x] = sum;
  simple_inclusive_scan(buf, blockDim.x);
  IntT cur_value = buf[threadIdx.x] - sum;  // `-sum` makes it exclusive-sum for now
  for (int i = 0; i < block_size; i++) {
    int idx = threadIdx.x * block_size + i;
    if (idx < num_items) {
      cur_value += data[idx];
      data[idx] = cur_value;
    }
  }
}

/*
  This function does a partial sort of an array, in reverse order, so that it's
  as if the array `start..start+input_size-1` is sorted, but only
  the elements numbered `start..start+num_keep-1` will necessarily be
  correct at the output.

  see test_merge() in sorting_ref.py for the original Python code that this was
  based on.  This is not very elegant but is probably enough for now.

  Caution: this only works correctly if the elements x to be sorted satisfy
  x > (x-1) [not true, for example, for 0 in unsigned arithmetic];
  this is due to the subtraction of 1 in "x_val_mod" below.

  Args:
        buf: pointer to start of buffer
        num_keep: number of items we need to be correct at output,
           must be a power of 2 with num_keep >= 1, and must be < blockIdx.x.
        num_items: number of items in the array, must be a power of 2
           and >= num_keep.  Does not necessarily have to be <= blockIdx.x.
 */
template <typename IntT>
__device__ void merge_based_partial_sort_reverse(
    IntT *buf, uint32_t num_keep, uint32_t num_items) {
  // num_keep == max_elements_needed in python.
  print_array(buf, num_items, "merge-sort-at-entry");
  __syncthreads();
  uint32_t old_sublist_size = 1;
  while (old_sublist_size < num_items) {
    uint32_t new_sublist_size = old_sublist_size * 2;

    for (uint32_t offset = 0; offset < num_items; offset += blockDim.x) {
      __syncthreads();
      uint32_t new_pos, i = offset + threadIdx.x;
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
      if (i < num_items)
        buf[new_pos] = x_val;
    }
    if (old_sublist_size < num_keep || num_items <= blockDim.x) {
      // The "|| num_items <= blockDim.x" is just to save a little time.
      old_sublist_size *= 2;
    } else {
      // old_sublist_size == num_keep.  we'll be splitting each block of size
      // (new_sublist_size == 2*num_keep) items into two halves and keeping only
      // the 1st half, getting rid of the empty space and halving the number of
      // items while keeping old_sublist_size the same, i.e. a list like 12345678
      // is reduced to 1256, assuming num_keep == 2.  This is a mechanism
      // to support larger num_items than blockDim.x.
      num_items = num_items / 2;
      for (uint32_t offset = 0; offset < num_items; offset += blockDim.x) {
        __syncthreads();
        // i corresponds to the index into the reduced list of items.
        uint32_t i = threadIdx.x + offset;
        IntT my_item;
        if (i < num_items) {
          uint32_t reduced_sublist_start = i & ~(old_sublist_size - 1),
              old_idx = i + reduced_sublist_start;
          my_item = buf[old_idx];
        }
        __syncthreads();
        if (i < num_items) {
          buf[i] = my_item;
        }
      }
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
              end: one-past-the-last index in `cumsum` that we might return;
                 may also be accessed, and contain a larger value than `r` might
                 ever have!  [see: #if 0 below;]
                 Must satisfy end > begin.
               r:  value whose interval we are searching for in `cumsum`

   Return:
       Returns, in all threads in each thread-group, the value of i, with
       begin <= i < end, such that
       cumsum[i] <= r < cumsum[i + 1], treating cumsum[end] as +infinity.
       This is undefined if r < cumsum[begin].

*/
__forceinline__ __device__ int find_class(
    cooperative_groups::thread_group &g,
    int32_t *shared_int,
    uint32_t *cumsum, int begin, int end, uint32_t r) {
  assert(end > begin);
  int orig_begin = begin, orig_end = end;  // debug

  g.sync();
  int i = g.thread_rank(),
      tile_size = g.size();

  while (end > begin + 1) {
    // 'block_size' is the number of indexes that each thread is responsible for
    // at this stage of the computation.
    int block_size = (end - begin + tile_size - 1) / tile_size;

    // block_start and block_end are the (start,end) points of the
    // block of indexes that this thread is responsible for.
    int block_start = begin + i * block_size,
        block_end = block_start + block_size;
    if (block_start < end &&
        r >= cumsum[block_start] && (
            (block_end >= end ||
             r < cumsum[block_end]))) {
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
      printf("[failure!]blockIdx.x=%d, threadIdx.{x,y}=%d,%d, orig begin,end=%d->%d,%d->%d, begin=%d, x,r,y=%d,%d,%d\n", blockIdx.x, threadIdx.x, threadIdx.y,
             orig_begin, cumsum[orig_begin], orig_end, cumsum[orig_end], begin,
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
  if (!(r >= cumsum[begin] && r < cumsum[begin + 1])) {
    uint32_t y = cumsum[begin + 1]; // (begin + 1 < orig_end ? cumsum[begin + 1] : 300000);
    printf("blockIdx.x=%d, threadIdx.{x,y}=%d,%d, search error:  orig_begin,orig_end=%u,%u, returning begin=%u, x,r,y=%u,%u,%u\n",
           blockIdx.x, threadIdx.x, threadIdx.y,
           orig_begin, orig_end, begin, cumsum[begin], r, y);
  }

  return begin;
}

inline __device__ double exp_wrapper(double f) { return exp(f); }
inline __device__ float exp_wrapper(float f) { return expf(f); }

template<typename T>
class PromoteHalfToFloat {
 public:
  using Type = T;
};
template<>
class PromoteHalfToFloat<torch::Half> {
 public:
  using Type = float;
};

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
    torch::PackedTensorAccessor32<Real, 3> probs,   // [B][N][M]
    torch::PackedTensorAccessor32<int64_t, 1> rand,     // [B]
    torch::PackedTensorAccessor32<int64_t, 3> indexes,     // [B][K][N]
    torch::PackedTensorAccessor32<int64_t, 2> combined_indexes,     // [B][K]
    torch::PackedTensorAccessor32<Real, 2> weights,     // [B][K]
    uint32_t K, bool input_is_log, uint32_t p_bits,
    uint32_t M_bits, uint32_t K_bits, uint32_t M_unique,
    uint32_t K_nthroot) {
  //__shared__ typename BlockScan::TempStorage temp_storage;
  int B = probs.size(0);  // batch size
  uint32_t N = probs.size(1),  // num distributions
      M = probs.size(2),  // num classes
      M_round = (1 << M_bits);  // M rounded up to power of 2.

  // For now just use 1 thread, we'll gradually parallelize operations.  The
  // following arrays are all located in shared memory.  Trailing underscore
  // is for variables that, in class CombinedSampler in sampling_cpu.cpp, were
  // class member variables.
  uint64_t *topK_P_ = reinterpret_cast<uint64_t*>(extern_buf),  // [K]
      *sorted_topK_P_ = topK_P_,   // [K], share with topK_P,
                                   // but dont change name for easier
                                   // comparison with CPu code.
      *topK_P_exclusive_sum_ = sorted_topK_P_ + K, // [K]
      *topK_delta_P_ = topK_P_exclusive_sum_ + K, // [K]
      *sorted_topK_delta_P_ = topK_delta_P_, // [K], share with topK_delta_P_,
                                             // but dont change name for easier
                                             // comparison with CPu code.
      *topK_cumsums_ = sorted_topK_delta_P_ + K, // [K]
      *sorted_topK_cumsums_ = topK_cumsums_, // [K], share with topK_cumsums,
                                             // but dont change name for easier
                                             // comparison with CPu code.
      *sorted_topK_delta_P_cumsum_ = sorted_topK_cumsums_ + K, // [K]
      *sorted_topK_cumsums_reduced_ = sorted_topK_delta_P_cumsum_, // [K], share with sorted_topK_delta_P_cumsum_
      *unreduced_samples_ = sorted_topK_cumsums_reduced_ + K, // [K]
      *P_sum_cumprod_ = unreduced_samples_ + K, // [N+1]
      *B_ = P_sum_cumprod_ + (N+1);  // [1]
  uint32_t *topK_indexes_ = reinterpret_cast<uint32_t*>(B_ + 1), // [K*N]
      *P_cumsum_ = topK_indexes_ + (K*N); // [(M+1) * N]
  uint32_t *indexes_for_samples_ = P_cumsum_ + ((M+1) * N);  // [K*N]
  // sort_buf32_ shares memory with indexes_for_samples_.  It is of length [M_round+K*(N-1)].
  uint32_t *sort_buf32_ = indexes_for_samples_;
  // sort_buf64_ is of size (K*K).  It
  // does not overlap with the first (K*N) elements of sort_buf32_, or with indexes_for_samples_,
  // which is the same size (K*N); but it overlaps with later elements of sort_buf32_.

  // Below, we have to do some bit-magic with the address of sort_buf64_ to make
  // sure it is properly aligned.  That is also what the +1 is for, to avoid the
  // "&~7" from possibly causing overlap with the int32_t buffer that precedes
  // it.
  uint64_t *sort_buf64_ = reinterpret_cast<uint64_t*>(
      reinterpret_cast<size_t>(sort_buf32_ + (K*N) + 1) & ~(size_t(7))); // [K*K]

  if (threadIdx.x < N) {
    // Zero the leading 0's in the P_cumsum_ array.
    P_cumsum_[threadIdx.x * (M+1)] = 0;
  }

  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    __syncthreads();  // So we don't overwrite things the previous block 'b' is doing.
    // for now do everything in 1 thread, we'll gradually move to doing more
    // things in parallel to ease debugging.

    uint64_t rand_source = rand[b];


    { // load_p() in sampling_cpu.cpp
      for (uint32_t n = 0; n < N; n++) {
        uint32_t multiple = 1 + M_unique * ((rand_source >> (M_bits * n)) % (M / M_unique));
        // First load P linearly from global memory to shared memory.
        uint32_t *P_buf = sort_buf32_;
        __syncthreads();  // re-use P_buf from last iter over n.
        for (uint32_t m = threadIdx.x; m < M; m += blockDim.x) {
          typename PromoteHalfToFloat<Real>::Type p = probs[b][n][m];
          if (input_is_log)
            p = exp_wrapper(p);
          // add K_nthroot for 2 reasons: (a) to prevent zero probs, which causes
          // all kinds of nasty edge cases, (b) it must be large enough that
          // "remainder_k", which could be as large as K-2 in the cases that
          // we are worried about, does not exceed the k'th largest product
          // of probs.  Ensuring products of probs are at least K satisfies this.

          uint32_t P = K_nthroot + uint32_t((1 << p_bits) * p);
          P_buf[m] = P;
        }
        // .. then pseudo-randomly reorder/shuffle P based on "multiple".
        __syncthreads();
        for (uint32_t m = threadIdx.x; m < M; m += blockDim.x) {
          uint32_t src_m = (m * multiple) % M;
          P_cumsum_[n * (M+1) + 1 + m] = P_buf[src_m];
        }
      }
    }

    /*
    for (uint32_t n = 0; n < N; n++) {
      __syncthreads();
      print_array(P_cumsum_ + n*(M+1), M+1,
                  "P_cumsum, prior to cumsum");
                  }*/

    { // compute_k_largest() in sampling_cpu.cpp.
      // This loop populates sort_buf32_ with the top-K probabilities
      // for each source distribution.
      for (uint32_t n = 0; n < N; n++) {
        __syncthreads();
        uint32_t *sort_buf = sort_buf32_ + (K*n);

        for (uint32_t m = threadIdx.x; m < M_round; m += blockDim.x) {
          uint32_t P = (m < M ? P_cumsum_[n*(M+1) + m + 1] : 0);
          sort_buf[m] = (P << M_bits) + m;
        }
        merge_based_partial_sort_reverse(sort_buf, K, M_round);

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
        for (uint32_t offset = 0; offset < K * K; offset += blockDim.x) {
          uint64_t new_S;
          uint32_t i = threadIdx.x + offset;
          if (i < K * K) {
            uint32_t best_k = i % K,
                new_k = i / K;
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
          if (i < K * K) {
            sort_combinations[i] = new_S;
          }
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

    {  // This block corresponds to compute_p_cumsum() in sampling_cpu.cpp.
      // P_cumsum_ currently stores integerized probs P, preceded by a zero; after
      // this function it will store the [exclusive] cumulative sum, of size M+1.
      for (uint32_t n = 0; n < N; n++) {
        // Compute inclusive cumulative sum of size M; we already padded on the
        // left with 0, so the effect is the same as exclusive sum.
        uint32_t *this_P_cumsum = P_cumsum_ + (M+1) * n + 1; // + 1: skip the 0.

        if (M <= blockDim.x) {
          simple_inclusive_scan(this_P_cumsum, M);
        } else {
          inclusive_scan(this_P_cumsum, sort_buf32_, M);
        }
        print_array(this_P_cumsum-1, M+1, "this_P_cumsum");
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint64_t P_sum_cumprod = 1;
        for (int n = 0; n < N; n++) {
          P_sum_cumprod_[n] = P_sum_cumprod;
          P_sum_cumprod *= P_cumsum_[(M+1)*n + M];
        }
        P_sum_cumprod_[N] = P_sum_cumprod;
      }
      print_array(P_sum_cumprod_, (N+1), "P_sum_cumprod_");
    }

    { // This block corresponds to compute_beta() in sampling_cpu.cpp, and
      // compute_bets_prods() in sampling_ref.py.  It computes B which
      // is integerized beta.
      // outputs: B_, and the array topK_delta_P_.
      __syncthreads();
      uint64_t Psum = P_sum_cumprod_[N];   // This is the total probability mass.
      // is_ok will be 1 if this i the chosen k, 0 otherwise, undefined if threadIdx.x >= K.
      uint32_t is_ok;
      uint32_t remainder_k;
      uint64_t this_P;
      if (threadIdx.x < K) {
        uint32_t k = threadIdx.x;
        uint64_t prev_P = (k == 0 ? (~((uint64_t)0)) : // infinity
                           topK_P_[k-1]),
            S1 = Psum - topK_P_exclusive_sum_[k],  // corresponds to 1-s_k in the math.
            B_k = S1 / (K-k);
        this_P = topK_P_[k];
        remainder_k = S1 % (K-k);
        is_ok = prev_P > B_k && this_P <= B_k;
        if (is_ok) { // should happen for exactly one k!!
          *B_ = B_k;
        }
      }
      __syncthreads();
      if (threadIdx.x < K) {
        uint32_t k = threadIdx.x;
        uint64_t B = *B_;  // read from the one
        // The following is equivalent to the following in sampling_cpu.cpp:
        // delta_P = (remainder * (k == chosen_k)) + P - std::min<uint64_t>(P, B);
        topK_delta_P_[k] = (this_P > B) * (this_P - B)  + (is_ok * remainder_k);
      }
      __syncthreads(); print_array(topK_delta_P_, K, "topK_delta_P_");
    }

    { // this block corresponds to compute_topk_cumsums() in sampling_cpu.cpp.
      __syncthreads();
      if (threadIdx.x < K) {
        uint32_t k = threadIdx.x;
        uint64_t P_selected_laterprod = 1,
            cumsum = 0;

        // This involves summations over the N dimension but we do this sequentially
        // as N will only be 2 or at most 3 in practice.
        for (int32_t n = int32_t(N) - 1; n >= 0; --n) {
          // 0 <= this_m < M
          uint32_t this_m = topK_indexes_[k*N + n],
              this_P_cumsum_idx = (n*(M+1)) + this_m;

          uint32_t this_P_cumsum = P_cumsum_[this_P_cumsum_idx],
              next_P_cumsum = P_cumsum_[this_P_cumsum_idx + 1],
              this_P = next_P_cumsum - this_P_cumsum;

          uint64_t prev_Psum_cumprod = P_sum_cumprod_[n];

          cumsum += prev_Psum_cumprod * P_selected_laterprod * uint64_t(this_P_cumsum);
          P_selected_laterprod *= this_P;
        }
        // we sort in a different way than the C code, so we don't need to encode
        // the index k in the index.  note: all these cumsum values are distinct
        // because the top-K indexes are distinct and integerized probabilities
        // are nonzero.
        topK_cumsums_[k] = cumsum;
      }
      __syncthreads(); print_array(topK_cumsums_, K, "topK_cumsums_[unsorted,no-k]");

      { // This block sorts topK_cumsums_ (an array of size K) using K*K threads
        // to count how many items are less than the current item.
        // It then reorders various things, using the reordering indexes from the
        // sorting.

        // TODO: could tune the type of buf, uint32_t would also work, or
        // uint8_t if we can assume K <= 256.
        uint16_t *buf = reinterpret_cast<uint16_t*>(sort_buf64_);
        __syncthreads();

        // K_reduced will normally equal K; it's less than K of K*K > blockDim.x.
        uint32_t K_reduced = std::min<uint32_t>(K, blockDim.x / K);  // power of 2
        uint64_t this_topK_cumsum;
        uint32_t this_k = threadIdx.x / K_reduced, other_k = threadIdx.x % K_reduced;

        if (this_k < K) {
          this_topK_cumsum = topK_cumsums_[this_k];
          uint32_t count = 0;
          for (uint32_t other_kk = other_k; other_kk < K; other_kk += K_reduced)
            count += (topK_cumsums_[other_kk] < this_topK_cumsum); // add 0 or 1
          buf[threadIdx.x] = count;
        }
        // sum up each block of K_reduced elements of `buf`, to work out how many
        // other elements of topK_cumsums_ are less than this element.
        __syncthreads();
        for (uint32_t s = 1; s < K_reduced; s *= 2) {
          if (this_k < K && threadIdx.x % (2*s) == 0)
            buf[threadIdx.x] += buf[threadIdx.x + s];
          __syncthreads();
        }
        uint32_t new_k;
        uint64_t topK_P, topK_delta_P;
        if (other_k == 0 && this_k < K) {
          new_k = buf[threadIdx.x];
          this_topK_cumsum = topK_cumsums_[this_k];
          topK_P = topK_P_[this_k];
          topK_delta_P = topK_delta_P_[this_k];
        }
        __syncthreads();
        if (other_k == 0 && this_k < K) {
          sorted_topK_cumsums_[new_k] = this_topK_cumsum;
          sorted_topK_P_[new_k] = topK_P;
          sorted_topK_delta_P_[new_k] = topK_delta_P;
          // next we'll want the exclusive-sum of sorted_topK_delta_P_
          // so shift right and copy to sorted_topK_delta_P_cumsum_.
          if (new_k+1 == K) {
            sorted_topK_delta_P_cumsum_[0] = 0;
          } else {
            sorted_topK_delta_P_cumsum_[new_k+1] = topK_delta_P;
          }
        }
      }
      __syncthreads();
      simple_inclusive_scan(sorted_topK_delta_P_cumsum_, K);

      if (threadIdx.x == 0) {
        uint64_t Psum = P_sum_cumprod_[N],
            B = *B_,
            delta_P_sum = sorted_topK_delta_P_cumsum_[K-1] + sorted_topK_delta_P_[K-1],
            err = Psum - delta_P_sum - (B * K);
        assert(err == 0); // this is a canary in
      }

      {
        __syncthreads(); // TEMP
        print_array(topK_cumsums_, K, "topK_cumsums_[sorted,no-k]");
        print_array(sorted_topK_delta_P_cumsum_, K, "sorted_topK_delta_P_cumsum_");
        print_array(sorted_topK_delta_P_, K, "sorted_topK_delta_P_");
      }

      __syncthreads();
      if (threadIdx.x < K) {
        uint32_t k = threadIdx.x;
        sorted_topK_cumsums_reduced_[k] = (sorted_topK_cumsums_[k] -
                                           sorted_topK_delta_P_cumsum_[k]);
      }
      __syncthreads(); print_array(sorted_topK_cumsums_reduced_, K, "sorted_topK_cumsums_reduced_");
    }

    { // this block corresponds to compute_unreduced_samples() in sampling_cpu.cpp.
      uint64_t *buf = sort_buf64_;
      // k2 corresponds to the output samples, k to the top-K probs
      // note, we might have k2 > K at this point.
      uint32_t k2 = threadIdx.x / K,
          k = threadIdx.x % K;
      // each k2 has a different `rand`.  Treat rand as "reduced" at this
      // point, meaning it's in a space where disallowed regions (of size,
      // delta_P_[k]), have been removed.
      uint64_t B = *B_,
          rand = (rand_source % B) + B * k2;
      if (k2 < K) {
        uint64_t reduced_cumsum_k = sorted_topK_cumsums_reduced_[k],
            delta_P =  sorted_topK_delta_P_[k];
        buf[threadIdx.x] = (rand >= reduced_cumsum_k) * delta_P;
      }
      __syncthreads();
      for (uint32_t s = 1; s < K; s *= 2) {
        if (k2 < K && threadIdx.x % (2*s) == 0)
          buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
      }
      if (k2 < K) {
        uint64_t delta_P_sum = buf[k2 * K],
            rand_shifted = rand + delta_P_sum;
        if (k == 0)
          unreduced_samples_[k2] = rand_shifted;
        uint64_t topK_disallowed_start = sorted_topK_cumsums_[k],
            topK_disallowed_end = topK_disallowed_start + sorted_topK_delta_P_[k];
        if (rand_shifted >= topK_disallowed_start &&
            rand_shifted < topK_disallowed_end) {
          printf("In disallowed region: topK_disallowed_start=%ul, rand_shifted=%ul, topK_disallowed_end=%ul, k=%d k2=%d\n",
                 topK_disallowed_start, rand_shifted, topK_disallowed_end, k, k2);
        }
        assert(!(rand_shifted >= topK_disallowed_start &&
                      rand_shifted < topK_disallowed_end));
        assert(rand_shifted < P_sum_cumprod_[N]);
      }
      __syncthreads();
      print_array(unreduced_samples_, K, "unreduced_samples_");
    }
    { // this block corresponds to compute_indexes_for_samples() in sampling_cpu.cpp.
      // We make the thread_group_size blockDim.x / K so that all threads can
      // participate; this removes any problems with needing a guard if any
      // threads do not participate.
      // it seems thread group size >32 may not be supported, at least there
      // is some
      uint32_t thread_group_size = std::min<uint32_t>(blockDim.x / K, uint32_t(32));
      cooperative_groups::thread_group tile =
          cooperative_groups::tiled_partition(cooperative_groups::this_thread_block(),
                                              thread_group_size);


      // buf is of size [K], overlaps with topK_P_ == topK_P_sorted_.
      int32_t *buf = reinterpret_cast<int32_t*>(topK_P_);

      __syncthreads();
      uint32_t k2 = threadIdx.x / thread_group_size;
      if (k2 < K) {
        uint64_t cur_sample = unreduced_samples_[k2];

        for (uint32_t n = N-1; ; --n) {  // we break in the loop if n == 0.
          uint64_t P_sum_cumprod = P_sum_cumprod_[n];  // product of previous Psum's.
          uint32_t this_sample = uint32_t(cur_sample / P_sum_cumprod);

          uint32_t *P_cumsum_start = P_cumsum_ + n * (M+1);
          uint32_t this_m_idx = find_class(tile,
                                           buf + k2,
                                           P_cumsum_start,
                                           (int)0, (int)M, this_sample);
          if (threadIdx.x % thread_group_size == 0)
            indexes_for_samples_[k2 * N + n] = this_m_idx;
          if (n == 0)
            break;
          uint32_t this_cumsum = P_cumsum_start[this_m_idx],
              next_cumsum = P_cumsum_start[this_m_idx + 1],
              this_P = next_cumsum - this_cumsum;

          uint64_t remainder = cur_sample - this_cumsum * P_sum_cumprod;
          cur_sample = remainder / this_P;
        }
      }
      __syncthreads(); // keep this __syncthreads(), it guards next block.
      print_array(indexes_for_samples_, K*N, "indexes_for_samples_");
    }
    { // this block corresponds to get_weights_for_samples() in sampling_cpu.cpp.
      // can't use half precision here or we'd get overflow.
      typename PromoteHalfToFloat<Real>::Type denom = P_sum_cumprod_[N],
          beta = *B_ / denom;
      uint64_t prod_P = 1;
      if (threadIdx.x < K) {
        uint32_t k2 = threadIdx.x;
        for (uint32_t n = 0; n < N; n++) {
          uint32_t this_m_idx = indexes_for_samples_[k2 * N + n];
          uint32_t this_P = P_cumsum_[n * (M + 1) + this_m_idx + 1] -
              P_cumsum_[n * (M + 1) + this_m_idx];
          prod_P *= this_P;
        }
        typename PromoteHalfToFloat<Real>::Type p = prod_P / denom;
        weights[b][k2] = (Real)std::max(p, beta);
        if (p < 0 || p > 1) {
          printf("ERROR: k2=%d, denom=%f, beta=%f, prod_P=%ul, p=%f", k2, denom, beta, prod_P, p);
        }
      }
    }
    { // this block corresponds to get_indexes_for_samples() in sampling_cpu.cpp.
      uint64_t combined_index = 0;
      uint32_t M_prod = 1;
      if (threadIdx.x < K) {
        uint32_t k2 = threadIdx.x;
        for (uint32_t n = 0; n < N; n++) {
          // same multiple we used in block corresponding to load_p(), above
          uint32_t multiple = 1 + M_unique * ((rand_source >> (M_bits * n)) % (M / M_unique)),
              this_m = indexes_for_samples_[k2 * N + n],
              orig_m = (this_m * multiple) % M;
          indexes[b][k2][n] = orig_m;
          combined_index += orig_m * M_prod;
          M_prod *= M;
        }
        combined_indexes[b][k2] = combined_index;
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
inline int find_num_bits_for(int n) {
  int num_bits = 1;
  while ((int(1) << num_bits) < n)
    num_bits++;
  return num_bits;
}

uint32_t find_prod_unique_prime_factors(uint32_t i) { // returns smallest number coprime to
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
  int M_bits = find_num_bits_for(M),
      K_bits = find_num_bits_for(K),
      p_bits = std::min(54 / N, // so product of N of these is comfortably less
                                // than 64, search for "headroom".
                        std::min((63/N) - K_bits,  // for when we sort `sort_combinations`.
                                 31 - M_bits)), // for when we sort `sort_buf` in ComputeKLargest()
      M_unique = find_prod_unique_prime_factors(M),
      M_round = 1 << M_bits,
      K_nthroot = 1 << ((K_bits + N - 1) / N);  // such that (K_nthroot**N) >= K,

  float epsilon = float(K_nthroot) / float(1 << p_bits);

  int size64 = sizeof(uint64_t),
      size32 = sizeof(uint32_t);
  int grid_dim_x = std::min<int>(B, 256),
      block_dim_x = std::min(std::max(M_round, K*K), 256);


  // the order in which we list these buffers differs from their declaration ordere,
  // because we are trying to keep the expressions for working out addresses, generally
  // as simple as possible (since the CUDA code would have to do this).
  int extern_memory_bytes = size64 * K + // topK_P_, sorted_topK_P
      size64 * K + // topK_P_exclusive_sum_
      size64 * K + // topK_delta_P_,sorted_topK_delta_P_
      size64 * K + // topK_cumsums_,sorted_topK_cumsums_
      size64 * K + // sorted_topK_delta_P_cumsum_,sorted_topK_cumsums_reduced_
      size64 * K + // unreduced_samples_
      size64 * (N+1) +  // P_sum_cumprod_
      size64 * 1 + // B_.
      size32 * (K*N) + // topK_indexes_
      size32 * (M+1) * N + // P_cumsum_
      std::max<int>(size32 * (K*(N-1) + M_round),  // sort_buf32_
                    size32 * (K*N + 1) + size64 * K*K); // [(indexes_for_samples_ or alternatively the 1st
                                                        // K*N elements of sort_buf32_), and then+sort_buf64_. [
                                                        // the +1 is for int64
                                                        // alignment issues.]

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(probs.scalar_type(), "sample_combined_cpu_forward_dispatch", ([&] {
        // scalar_t is defined by the macro AT_DISPATCH_FLOATING_TYPES
        sample_combined_forward_kernel<scalar_t><<<grid_dim_x, block_dim_x, extern_memory_bytes>>>(
            probs.packed_accessor32<scalar_t, 3>(),
            rand.packed_accessor32<int64_t, 1>(),
            indexes.packed_accessor32<int64_t, 3>(),
            combined_indexes.packed_accessor32<int64_t, 2>(),
            weights.packed_accessor32<scalar_t, 2>(),
            K, input_is_log, p_bits, M_bits, K_bits, M_unique,
            K_nthroot);
      }));
  gpuErrchk(cudaGetLastError());

  // if Real == torch::Half this will round to zero but we'll never use it in that
  // case as we require input_is_log == True for half precision.
  torch::Tensor epsilon_tensor = torch::full({}, epsilon, real_opts);
  return std::vector<torch::Tensor>({indexes, combined_indexes, weights, epsilon_tensor});
}
