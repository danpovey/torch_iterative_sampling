#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>



inline torch::Half exp_wrapper(torch::Half f) { return (torch::Half)expf((float)f); }
inline float exp_wrapper(float f) { return expf(f); }
inline double exp_wrapper(double f) { return exp(f); }

/*
  Return the index i into cumsum, with begin <= i < end,
  such that cumsum[i] <= r < cumsum[i + 1].
  We assume that cumsum[begin] <= r < cumsum[end], and we do not
  access cumsum[begin] or cumsum[end].
*/
template <typename IterType> int find_class(
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



template <typename IntT>
void print_array(IntT *buf, int num_items, const char *name) {
  /*
  printf("%s = [", name);
  for (int i = 0; i < num_items; i++) {
    printf("%ld ",  (long)buf[i]);
  }
  printf("]\n");
  */
}


/*
  This is a prototype for some CUDA code.  It does a partial sort of an array,
  in reverse order, so that it's as if the array `start..start+num_items-1` is
  sorted, but we only care about the `start..start+num_keep-1` elements.

  see test_merge() in sorting_ref.py for the original Python code that this was
  based on.  This is not very elegant but is probably enough for now.

  Caution: this only works correctly if the elements x to be sorted satisfy
  x > (x-1) [not true, for example, for 0 in unsigned arithmetic];
  this is due to the subtraction of 1 in "x_val_mod" below.

template <typename IntT> void merge_based_partial_sort_reverse(
    IntT* start, uint32_t num_keep, uint32_t num_items) {
  // num_keep == max_elements_needed in python.
  print_array(start, num_items, "merge-sort-at-entry");
  std::vector<IntT> sorted(start, start + num_items);
  std::sort(sorted.begin(), sorted.end(), std::greater<IntT>());
  std::vector<IntT> temp(num_items);
  for (uint32_t new_sublist_size = 2;
       new_sublist_size <= num_items;
       new_sublist_size *= 2) {
    uint32_t old_sublist_size = new_sublist_size / 2;
    for (uint32_t i = 0; i < num_items; i++) {
      IntT x_val = start[i];
      uint32_t offset_in_old = i & (old_sublist_size - 1);
      if (offset_in_old >= num_keep)
        continue;
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
        if (x_val_mod < start[search_offset + mid - 1]) search_begin = mid;
        else search_end= mid;
      }
      uint32_t new_pos = new_sublist_start + offset_in_old + search_begin;
      temp[new_pos] = x_val;
    }
    for (uint32_t i = 0; i < num_items; i++) {
      start[i] = temp[i];
    }
  }
  for (uint32_t i = 0; i < uint32_t(std::min<int>(num_items, num_keep)); i++) {
    TORCH_CHECK(start[i] == sorted[i]);
  }
  print_array(start, num_items, "merge-sort-at-exit");
}
*/



class CombinedSampler {
 public:
  CombinedSampler(uint32_t N, uint32_t M, uint32_t K):
      N_(N), M_(M), K_(K),
      M_unique_(find_prod_unique_prime_factors(M)) {
    TORCH_CHECK(N < 5);
    TORCH_CHECK((K&(K-1)) == 0);  // require K is a power of 2.
    M_bits_ = FindNumBitsFor(M);
    K_bits_ = FindNumBitsFor(K);

    p_bits_ = std::min(uint32_t(54) / N, // so product of N of these is
                                         // comfortably less than 64, search for
                                         // "headroom"
                       std::min((uint32_t(63)/N) - K_bits_,  // for when we sort `sort_combinations`.
                                uint32_t(31) - M_bits_)); // for when we sort `sort_buf` in ComputeKLargest()

    // TEMP.
    // p_bits_ = 15;

    // TODO: allocate buffers..
    uint32_t Kpow1or2 = (N > 1 ? (K*K) : K),
        size64 = sizeof(uint64_t) / sizeof(uint32_t);
    // the order in which we list these buffers differs from their declaration ordere,
    // because we are trying to keep the expressions for working out addresses, generally
    // as simple as possible (since the CUDA code would have to do this).
    int tot_size = size64 * K + // topK_P_
        size64 * K + // topK_P_exclusive_sum_
        size64 * K + // topK_delta_P_
        size64 * K + // topK_cumsums_
        size64 * K + // sorted_topK_P_
        size64 * K + // sorted_topK_delta_P_
        size64 * K + // sorted_topK_delta_P_cumsum_
        size64 * K + // sorted_topK_cumsums_
        size64 * K + // sorted_topK_cumsums_reduced_
        size64 * K + // unreduced_samples_
        size64 * Kpow1or2 + // sort_buf64_, because double the element size
        size64 * (N+1) +  // P_sum_cumprod_
        (K*N) + // topK_indexes_
        (M+1) * N + // P_cumsum_
        std::max<uint32_t>((M+(N-1)*K), // sort_buf32_,
                           K*N); // indexes_for_samples_

    // indexes_for_samples_, of size K*N, shares memory with sort_buf32_.
    // allocate as uint64_t for alignment reasons.
    buffers_.resize((tot_size+1)/2);
    uint64_t *p = buffers_.data();
    set_buffer(topK_P_, p, K);
    set_buffer(topK_P_exclusive_sum_, p, K);
    set_buffer(topK_delta_P_, p, K);
    set_buffer(topK_cumsums_, p, K);
    set_buffer(sorted_topK_P_, p, K);
    set_buffer(sorted_topK_delta_P_, p, K);
    set_buffer(sorted_topK_delta_P_cumsum_, p, K);
    set_buffer(sorted_topK_cumsums_, p, K);
    set_buffer(sorted_topK_cumsums_reduced_, p, K);
    set_buffer(unreduced_samples_, p, K);
    set_buffer(sort_buf64_, p, Kpow1or2);
    set_buffer(P_sum_cumprod_, p, N+1);
    // now, 32-but
    uint32_t *p32 = reinterpret_cast<uint32_t*>(p);
    set_buffer(topK_indexes_, p32, K*N);
    set_buffer(P_cumsum_, p32, (M+1) * N);
    sort_buf32_ = p32;
    indexes_for_samples_ = p32;
    p32 += std::max<uint32_t>((M+(N-1)*K), K*N);  // indexes_for_samples_
    int size = p32 - reinterpret_cast<uint32_t*>(buffers_.data());
    TORCH_CHECK(size == tot_size);
  }

  void set_buffer(uint32_t* &buffer, uint32_t* &p, uint32_t size) {
    buffer = p;
    p += size;
  }
  void set_buffer(uint64_t* &buffer, uint64_t* &p, uint32_t size) {
    buffer = p;
    p += size;
  }


  // returns a pseudo random number coprime to M_, as a function of n.
  // this is deterministic within a batch.
  inline uint32_t get_random_coprime(uint32_t n) {
    return 1 + M_unique_ * ((rand_source_ >> (M_bits_ * n)) % (M_ / M_unique_));
  }

  template <typename Real, typename AccessorT>
  void load_p(uint64_t rand_source,
             bool input_is_log,
             AccessorT p) { // p: [N][M]
    rand_source_ = rand_source;

    uint32_t N = N_, M = M_;
    for (uint32_t n = 0; n < N; n++) {
      auto p_n = p[n];

      uint32_t multiple = get_random_coprime(n);
      printf("M_unique_ = %ld\n", (long int) M_unique_);
      printf("multiple = %ld\n", (long int) multiple);

      // Here we do the reordering as 1 operation.  For the CUDA code we'll
      // probably load from main memory to shared memory without reordering, and
      // then reorder as a separate operation, because we want to keep main
      // memory access linear.

      uint32_t *this_P_cumsum = P_cumsum_ + n * (M+1);
      this_P_cumsum[0] = 0;
      uint32_t p_multiple = 1 << (p_bits_);
      for (uint32_t m = 0; m < M; m++) {
        uint32_t src_m = (m * multiple) % M;
        Real src_p = p_n[src_m];
        if (input_is_log)
          src_p = exp_wrapper(src_p);
        printf("%f ", src_p);
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


  void  __attribute__ ((noinline))  compute() {
    for (uint32_t n = 0; n < N_; n++) {
      print_array(P_cumsum_ + n*(M_+1), M_+1, "P_cumsum, prior to cumsum");
    }
    compute_k_largest();
    compute_p_cumsum();
    compute_beta();
    compute_topk_cumsums();  // also re-sorts top-K
    compute_unreduced_samples();
    compute_indexes_for_samples();
    // next is get_weights_for_samples(); this is called by the user,
    // then get_indexes_for_samples(); this is called by the user.
  }

  /*
    `weights` is of shape [K].  We write the samples' weights to here.
   */
  template <typename Real, typename AccessorT>
  void get_weights_for_samples(
      AccessorT weights) {
    uint32_t N = N_, M = M_, K = K_;
    float denom = (float)P_sum_cumprod_[N];
    float beta = (float)B_ / denom;

    for (uint32_t k2 = 0; k2 < K; k2++) { // parallelize over k2
      uint64_t prod_P = 1;
      for (uint32_t n = 0; n < N; n++) {
        uint32_t this_m_idx = indexes_for_samples_[k2 * N + n];
        uint32_t this_P = P_cumsum_[n * (M + 1) + this_m_idx + 1] - P_cumsum_[n * (M + 1) + this_m_idx];
        prod_P *= this_P;
      }
      float p = (float)prod_P / denom;
      weights[k2] = (Real)std::max(p, beta);
    }
  }


  template <typename Accessor1, typename Accessor2>
  void get_indexes_for_samples(
      Accessor1 indexes,  // torch::TensorAccessor32<int64_t, 2>
      Accessor2 combined_indexes) {  // torch::TensorAccessor32<int64_t, 1>
    uint32_t N = N_, M = M_, K = K_;
    for (uint32_t k2 = 0; k2 < K; k2++) { // parallelize over k2, maybe also over n.
      uint64_t combined_index = 0;
      uint32_t M_prod = 1;
      for (uint32_t n = 0; n < N; n++) {
        uint32_t multiple = get_random_coprime(n);  // same multiple we used in LoadP().
        uint32_t this_m = indexes_for_samples_[k2 * N + n];
        uint32_t orig_m = (this_m * multiple) % M;
        indexes[k2][n] = orig_m;
        combined_index += orig_m * M_prod;
        M_prod *= M;
      }
      combined_indexes[k2] = combined_index;
    }
  }


 private:

  uint32_t N_; // number of softmaxes
  uint32_t M_; // size of each softmax
  uint32_t K_; // number of samples we require per softmax.
  uint32_t M_unique_;  // product of unique prime factors of M_

  // rand_source_ is the random source for reordering modulo M (lowest bits) and
  // for generating samples (all bits).
  uint64_t rand_source_;

  // number of bits used for storing index 0 <= m < M_ when sorting R_; require (1<<M_bits_) >= M_
  uint32_t M_bits_;

  // number of bits used for probabilities; require:
  //  (i) N*p_bits_ <= 54 [an arbitrary choice with 54 << 64,
  //       search for "headroom" in sampling_ref.py to understand]
  //  (ii) (p_bits_+K_bits)*N < 64 [strictly less than 64 to allow for rounding error].
  //  (iii) also p_bits_ + M_bits_ + 1 < 32 because of how we
  //      sort to find K-best probs [the +1 is in case some probs or sums of probs
  //      are slightly more than 1 due to rounding.
  uint32_t p_bits_;
  // number of bits used for indexes 0 <= k < K;
  uint32_t K_bits_;


  std::vector<uint64_t> buffers_;

  /* of shape [N][M+1], indexed P_cumsum_[n*(M+1) + m], P_cumsum_ is used
     to store the exclusive cumulative sums of the integerized input probabilities P;
     elements with m==0 are zero, and the last column contains the sums of P. */
  uint32_t *P_cumsum_;
  /*
    Of shape [M + (N-1)*K], this buffer is used to sort the K-best probabilities per
    input distribution. [The +(N-1)*K is because we shift right by K each time we
    use it.
   */
  uint32_t *sort_buf32_;
  /*
    Of shape [K**N], this buffer is used to sort the K-best tuples of probabilities
    per input distribution.  It does not overlap with the first (N*K) items of
    sort_buf32_.
   */
  uint64_t *sort_buf64_;
  /*
    Of shape [N+1], P_sum_cumprod_ is the exclusive cumulative product of P_cumsum_[n][M],
    i.e. the cumulative product of the total sum of P which is the last column of P_cumsum_.
   */
  uint64_t *P_sum_cumprod_;
  /*
    of shape [K][N], indexed [k*N + n], topK_indexes_ contains the N-tuples of indexes
    (in 0,1,...,M-1) of the top K combinations of indexes, from highest to lowest
    probability.
   */
  uint32_t *topK_indexes_;
  /*
    Of shape [K], this contains the same info as topK_indexes_, but as a single index, also
    combined with the k value; this is used in sorting the topK indexes numerically
    by index.
    Specifically, before we sort it, it contains:
      sum_{n=0}^{N-1}:
         topK_indexes_[k][n] << (M_bits_*(n+1))
       + k  [this only works because we know that K < M.]
   */
  uint32_t *topK_indexes_combined_;
  /*,
    of shape [K], contains, from highest to lowest, the top-K product probabilities;
    these are products of "P" values.
   */
  uint64_t *topK_P_;
  /*
    of shape [K], contains the exclusive cumulative sum of topK_P_.
   */
  uint64_t *topK_P_exclusive_sum_;
  /*
    of shape [K], contains the delta_P values as returned
    by compute_beta_prods() in the python version.  These are the amounts of probability mass we remove
    for each of the top-K index-N-tuples to when we assign sampling-probs.

    Originally they are ordered from largest to smallest probs; after ReorderTopk() they are ordered
    from smallest to largest index-tuple [i.e. by topK_indexes_].
   */
  uint64_t *topK_delta_P_;
  /*
    topK_cumsums_, of shape [K], contains the exclusive cumulative-sums of products of
    all probs.   Note: this is the cumulative-sum over the N-tuples of indexes in [0..M-1],
    NOT the cumulative sum over the K index.  We compute this using a formula
    because it would be slow to manually compute that exclusive sum.

    Caution: also contains the k index.  TODO, document this.
  */
  uint64_t *topK_cumsums_;
  /*
    of shape [K], the same as topK_P_, but sorted by the N-tuple of indexes.
    in CUDA version can just make this the same address as topK_P_.
  */
  uint64_t *sorted_topK_P_;
  /*
    of shape [K], the same as topK_delta_P_, but sorted by the N-tuple of indexes.
    in CUDA version can just make this the same address as topK_delta_P_.
  */
  uint64_t *sorted_topK_delta_P_;
  /*
    of shape [K+1], sorted_topK_delta_P_cumsum_ contains the exclusive cumulative
    sum of sorted_topK_delta_P_.
   */
  uint64_t *sorted_topK_delta_P_cumsum_;
  /*
    sorted_topK_cumsums_, of shape [K], contains the exclusive cumulative-sums
    of products of all probs.  Note: this is the cumulative-sum over the
    N-tuples of indexes in [0..M-1], NOT the cumulative sum over the K index.
    We compute this using a formula because it would be slow to manually compute
    that exclusive sum.
  */
  uint64_t *sorted_topK_cumsums_;
  /*
    Of shape [K], equals sorted_topK_cumsums_ - sorted_topK_delta_P_, i.e. it
    contains the current cumsum minus the sum of preceding delta_P's.
   */
  uint64_t *sorted_topK_cumsums_reduced_;
  /*
    Contains which is the integerized beta value.
   */
  uint64_t B_;
  /*
    Contains the K random samples [these are not independent, they were separated by
    B when generated]that have been un-reduced by excluding disallowed regions.
    Shape: [K2].   K2 is numerically the same as K but refers to the indexes of
        the samples, which are separate from the indexes of the top-K
        index-tuples' probabilities.
   */
  uint64_t *unreduced_samples_;
  /*
    Of size [K][N], indexed as indexes_for_samples[k*N + n], contains the indexes
                    in {0..M-1} for the samples in unreduced_samples_.  These are
                    pseudo-random-reordered indexes, as are m indexes in P_cumsum_,
                    search for get_random_coprime().
   */
  uint32_t *indexes_for_samples_;

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


  void compute_k_largest() {
    // [1] compute K largest probabilities P and their corresponding indexes.
    // [2] sort the [K**N] products of the K largest probabilities, remembering their
    //     original indexes in 0..K-1.
    // [3] note the original [K][N] indexes of these products (in [0..M-1],
    //     and the corresponding [K] products of probabilities, ordered from
    //     largest to smallest.

    // in CUDA we'll do this sequentially over n (to save memory) and
    // via a parallel sorting algorithm from cub over k.

    uint32_t M_bits = M_bits_, M = M_, N = N_, K = K_;

    for (uint32_t n = 0; n < N; n++) {
      uint32_t *this_P_buf = P_cumsum_ + (n * (M+1)) + 1;
      // Each time we access this buffer we shift right by
      // K, because we need to remember the top-K items for each n.
      uint32_t *sort_buf = sort_buf32_ + (K * n);
      for (uint32_t m = 0; m < M; m++) {
        uint32_t p = this_P_buf[m];
        sort_buf[m] = m + (p << M_bits); // keep track of indexes.
      }
      // in CUDA we'll just sort the entire array.  Note, we don't need the
      // sorting algorithm to sort the indexes because we include them manually.

      //merge_based_partial_sort_reverse(sort_buf, K, M);
      std::nth_element(sort_buf, sort_buf + K, sort_buf + M, std::greater<uint32_t>());
      std::sort(sort_buf, sort_buf + K, std::greater<uint32_t>());
    }
    uint64_t *sort_combinations = sort_buf64_;
    uint32_t K_bits = K_bits_;

    uint32_t n = 0;
    for (uint32_t k = 0; k < K; k++) {
      uint32_t this_p = (sort_buf32_[K*n + k] >> M_bits); // one of the k-best probs for the n'th softmax
      sort_combinations[k] = (((uint64_t)this_p) << (N * K_bits)) | k;
    }
    print_array(sort_combinations, K, "sort-combinations-n=0");
    for (n = 1; n < N; n++) {
      uint64_t K_mask = (uint64_t(1) << uint64_t(n * K_bits)) - 1;
      // best_k is an index into the 1st K elements of array `sort_combinations`
      for (uint32_t best_k = 0; best_k < K; best_k++) {
        uint64_t S = sort_combinations[best_k],
            P = S & ~K_mask,
            prev_ks = S & K_mask;

        for (uint64_t new_k = 0; new_k < K; new_k++) {
          uint64_t combined_k = prev_ks | (new_k << (n * K_bits));
          // one of the k-best probs for the n'th softmax
          uint32_t this_p = (sort_buf32_[K*n + new_k] >> M_bits);
          uint64_t new_S = (P * this_p) | combined_k;
          sort_combinations[best_k + (K * new_k)] = new_S;
        }
      }
      // work out the K-best combinations so far.
      std::nth_element(sort_combinations, sort_combinations + K, sort_combinations + (K*K),
                       std::greater<uint64_t>());
      // merge_based_partial_sort_reverse(sort_combinations, K, K*K);
    }
    //if (N == 1) {
    //  merge_based_partial_sort_reverse(sort_combinations, K, K);
    //}
    //print_array(sort_combinations, K, "sort-combinations-n=N");
    std::sort(sort_combinations, sort_combinations + K, std::greater<uint64_t>());

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

    print_array(topK_indexes_, K, "topK_indexes");
    print_array(topK_P_exclusive_sum_, K, "topK_P_exclusive_sum");
    print_array(topK_P_, K, "topK_P_");
  }

  void compute_p_cumsum() {
    // Compute P_cumsum_, P_sum_cumprod_.

    // P_cumsum_ currently stores integerized probs P, preceded by a zero; after
    // this function it will store the [exclusive] cumulative sum, of size M+1.
    uint32_t M = M_, N = N_;
    uint64_t P_sum_cumprod = 1;
    for (uint32_t n = 0; n < N; n++) {
      // Compute inclusive cumulative sum of size M; we already padded on the
      // left with 0, so the effect is the same as exclusive sum.
      uint32_t *this_P_cumsum = P_cumsum_ + (M+1) * n + 1; // + 1: skip the 0.
      uint32_t sum = 0;
      for (uint32_t m = 0; m < M; m++) {  // note, we wrote 0 at one past the end.
        sum += this_P_cumsum[m];
        this_P_cumsum[m] = sum;
      }
      print_array(this_P_cumsum-1, M+1, "this_P_cumsum");
      P_sum_cumprod_[n] = P_sum_cumprod;
      P_sum_cumprod *= sum;
    }
    P_sum_cumprod_[N] = P_sum_cumprod;
    print_array(P_sum_cumprod_, (N+1), "P_sum_cumprod_");
  }

  void compute_beta() {
    // see compute_beta_prods() in sampling_ref.py.  Computes B which
    // is integerized beta.
    uint32_t N = N_, K = K_;
    uint64_t Psum = P_sum_cumprod_[N];   // This is the total probability mass.
    // Ptop corresponds to topK_P_[0..K-1]

    uint64_t B = 0,  // suppress warning
        remainder;
    uint32_t k, chosen_k;
    for (k = 0; k < K; k++) {  // We can parallelize over k.
      // We are trying out to see whether we get an admissible B (i.e.,
      // integerized beta) value with exactly k top probabilities exceeding B.  Exactly
      // one such k index will "work".
      uint64_t prev_P = (k == 0 ? (~((uint64_t)0)) : // infinity
                         topK_P_[k-1]),
          this_P = topK_P_[k];
      uint64_t S1 = Psum - topK_P_exclusive_sum_[k];  // corresponds to 1-s_k in the math.
      uint64_t B_k = S1 / (K-k);
      uint32_t remainder_k = S1 % (K-k);
      bool is_ok = prev_P > B_k && this_P <= B_k;
      if (is_ok) { // should happen for exactly onc k!!
        B = B_k;
        remainder = remainder_k;
        chosen_k = k;
        break;
      }
    }
    TORCH_CHECK(B != 0);
    B_ = B;
    TORCH_CHECK(k < K);  // check that we broke from the loop.

    uint64_t delta_P_sum = 0;  // only needed for checking.
    for (uint32_t k = 0; k < K; k++) { // parallelize over k.
      uint64_t P = topK_P_[k];
      topK_delta_P_[k] = (remainder * (k == chosen_k)) + P - std::min<uint64_t>(P, B);
      delta_P_sum += topK_delta_P_[k];
    }
    uint64_t err = Psum - delta_P_sum - (B * K);
    TORCH_CHECK(err == 0);
    // outputs: B_, and the array topK_delta_P_.
    print_array(topK_delta_P_, K, "topK_delta_P_");
  }

  void compute_topk_cumsums() {
    // Computes top-K cumulative sums; these are the cumulative sums of probabilities of
    // all N-tuples of indexes that precede each of the top-K cumulative sums.
    uint32_t M = M_, N = N_, K = K_;

    for (uint32_t k = 0; k < K_; k++) {  // this will be separate kernel threads.
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
      // we're about to sort on k, and we want to remember the original k index,
      // so include the k-index as the lowest order bits.  there is some "headroom"
      // on top of the 54 bits, so there's room for this.
      topK_cumsums_[k] = (cumsum << K_bits_) + k;
    }
    print_array(topK_cumsums_, K, "topK_cumsums_[unsorted]");
    std::sort(topK_cumsums_, topK_cumsums_ + K);
    print_array(topK_cumsums_, K, "topK_cumsums_[sorted]");

    uint32_t K_bits = K_bits_;

    for (uint32_t k = 0; k < K; k++) {
      uint64_t cumsum_with_k = topK_cumsums_[k];
      uint32_t k_orig = cumsum_with_k & (K-1);
      uint64_t cumsum = cumsum_with_k >> K_bits;
      sorted_topK_cumsums_[k] = cumsum;  // remove the index.

      sorted_topK_P_[k] = topK_P_[k_orig];
      uint64_t delta_P = topK_delta_P_[k_orig];
      sorted_topK_delta_P_[k] = delta_P;
    }
    uint64_t topK_delta_P_cumsum = 0;
    for (uint32_t k = 0; k < K; k++) {
      uint64_t delta_P = sorted_topK_delta_P_[k];
      sorted_topK_delta_P_cumsum_[k] = topK_delta_P_cumsum;
      topK_delta_P_cumsum += delta_P;
    }

    print_array(sorted_topK_delta_P_cumsum_, K, "sorted_topK_delta_P_cumsum_");
    print_array(sorted_topK_delta_P_, K, "sorted_topK_delta_P_");

    for (uint32_t k = 0; k < K; k++) {
      sorted_topK_cumsums_reduced_[k] = (sorted_topK_cumsums_[k] -
                                         sorted_topK_delta_P_cumsum_[k]);
    }
    print_array(sorted_topK_cumsums_reduced_, K, "sorted_topK_cumsums_reduced_");
  }

  void compute_unreduced_samples() {
    // will parallelize over (k2, k) pairs.  k2 corresponds to the output samples, k to the top-K probs
    uint32_t K = K_, N = N_;
    for (uint32_t k2 = 0; k2 < K; k2++) {
      uint64_t B = B_;
      // each k2 has a different rand.  Treat rand as "reduced" at this
      // point, meaning it's in a space where disallowed regions have been
      // removed.
      uint64_t rand = (rand_source_ % B) + B * k2;

      // uint64_t *delta_P_buf_ = M_buf64_;
      uint64_t delta_P_sum = 0;  // we'll compute this via a reduction in CUDA.
                                 // assume K is power of 2.

      // the following is compute_unreduced_samples() in python.
      // Note: we should be able to parallelize over (k2, k) pairs, summing
      // using some kind of logarithmic reduction.
      for (uint32_t k = 0; k < K; k++) {
        uint64_t reduced_cumsum_k = sorted_topK_cumsums_reduced_[k],
            delta_P =  sorted_topK_delta_P_[k];
        delta_P_sum += (rand >= reduced_cumsum_k) * delta_P;
      }
      uint64_t rand_shifted = rand + delta_P_sum;
      unreduced_samples_[k2] = rand_shifted;

      for (uint32_t k = 0; k < K; k++) { // check_unreduced_samples()
        uint64_t topK_disallowed_start = sorted_topK_cumsums_[k],
            topK_disallowed_end = topK_disallowed_start + sorted_topK_delta_P_[k];
        TORCH_CHECK(!(rand_shifted >= topK_disallowed_start &&
                      rand_shifted < topK_disallowed_end));
        TORCH_CHECK(rand_shifted < P_sum_cumprod_[N]);
      }
    }
    print_array(unreduced_samples_, K, "unreduced_samples_");
  }

  void compute_indexes_for_samples() {
    uint32_t K = K_, N = N_, M = M_;
    for (uint32_t k2 = 0; k2 < K; k2++) {
      uint64_t cur_sample = unreduced_samples_[k2];

      // we'll parallelize over k2.  We have to do the n index sequentially.
      for (uint32_t n = N-1; ; --n) {  // we break in the loop if n == 0.

        uint64_t P_sum_cumprod = P_sum_cumprod_[n];  // product of previous Psum's.
        uint32_t this_sample = uint32_t(cur_sample / P_sum_cumprod);

        uint32_t *P_cumsum_start = P_cumsum_ + n * (M+1);
        // This uses find_class() which implements a binary search.
        uint32_t this_m_idx = find_class(P_cumsum_start,
                                         (int)0, (int)M, this_sample);

        //std::cout << "Indexes-for-samples: k2=" << k2 << ", n=" << n << "=" << this_m_idx << "\n";
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
    print_array(indexes_for_samples_, K*N, "indexes_for_samples_");
  }

  // returns num_bits >= 1 such that (1 << num_bits) >= n.
  inline uint32_t FindNumBitsFor(uint32_t n) {
    uint32_t num_bits = 1;
    while ((uint32_t(1) << num_bits) < n)
      num_bits++;
    return num_bits;
  }

};


/*
  sampling forward function (the backward is implemented in python).  Please see
  `sample_combined_forward` in sampling_ref.py for a comparable PyTorch implementation
  in Python, that is easier to follow.

  Sample from a distribution that is the product of softmaxes.  We will sample
  K *distinct* samples.  This entails using sampling weights of the form min(1, p/beta)
  for a computed beta.

    Args:
         p: A Tensor of shape (B, N, M): either normalized log-probs (if input_is_log==False),
             or normalized probabilities; normalized along the M axis.  B is the batch
             size; M will typically be a power of 2 but can be any number, ideally with only
             small prime factors; and N must be in [1,2,3,4].  The type of p can be half,
             float or double.

        rand: A Tensor of int64_t of shape (B * (N+1),) containing random numbers in [0..2**63-1]
           which is the largest int64_t.  Actually this could just as easily be randum
           numbers in 0..2**64-1, as we'll be interpreting this as uint64_t.  The reason
           we want a flat array rather than array of shape (B, N+1) is so that we
           can access it in a more memory-efficient way, avoiding scattered reads.

         K: An integer, the number of samples required, with 0 < K < M
   input_is_log:  True if p represents normalized log-probs, False if it represents
             probabilities.

    Returns: (indexes, combined_indexes, weights)
       indexes: of shape (B, K, N), for each of K samples from a distribution it contains
            an N-tuple of indexes saying which combination of indexes from the
            component distributions were sampled.
       combined_indexes: of shape (B, K), contains the N-tuples in `indexes` combined
           into a single integer in [0..(K**N)-1]
       weights: of shape (B, K), gives the weight associated with each sample,
            which will equal max(p, beta) for a beta specific to the batch element,
            i.e. to the product of the distributions (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
*/
std::vector<torch::Tensor>
sample_combined_forward_cpu(torch::Tensor probs, // [B][N][M]
                            torch::Tensor rand,   // [B]
                            int K, bool input_is_log) {
  TORCH_CHECK(probs.dim() == 3, "probs must be 2-dimensional");
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

  TORCH_CHECK(probs.device().is_cpu() && rand.device().is_cpu(),
              "inputs must be CPU tensors");

  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(probs.device());
  auto real_opts = torch::TensorOptions().dtype(probs.dtype()).device(probs.device());

  // TODO: make empty
  torch::Tensor indexes = torch::empty({B, K, N}, long_opts),
      combined_indexes = torch::empty({B, K}, long_opts);

  torch::Tensor weights = torch::empty({B, K}, real_opts);

  AT_DISPATCH_FLOATING_TYPES(probs.scalar_type(), "sample_combined_cpu_forward_dispatch", ([&] {
        auto probs_a = probs.packed_accessor32<scalar_t, 3>();  // scalar_t comes from the macro.
        auto weights_a = weights.packed_accessor32<scalar_t, 2>();
        auto rand_a = rand.packed_accessor32<int64_t, 1>();
        auto indexes_a = indexes.packed_accessor32<int64_t, 3>();
        auto combined_indexes_a = combined_indexes.packed_accessor32<int64_t, 2>();

        CombinedSampler sampler(N, M, K);

        for (int b = 0; b < B; b++) {
          uint64_t rand = uint64_t(rand_a[b]);
          //          torch_scheduled_sampling/sampling_cpu.cpp:563:55: error: no matching function for call to 'CombinedSampler::LoadP(uint64_t&, bool&, at::TensorAccessor<double, 2, at::DefaultPtrTraits, int>)'
          sampler.load_p<scalar_t>(rand, input_is_log, probs_a[b]);
          sampler.compute();
          sampler.get_weights_for_samples<scalar_t>(weights_a[b]);
          // torch_scheduled_sampling/sampling_cpu.cpp:565:52: note:   'at::TensorAccessor<double, 1, at::DefaultPtrTraits, int>' is not derived from 'at::PackedTensorAccessor32<Real, 1>'
          sampler.get_indexes_for_samples(indexes_a[b], combined_indexes_a[b]);
        }
      }));
  //  std::cout << "combined_indexes = " << combined_indexes;
  return std::vector<torch::Tensor>({indexes, combined_indexes, weights});
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_combined_forward_cpu", &sample_combined_forward_cpu,
        "Multi-softmax sampling function forward (CPU)");
}
