#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>


template <typename Real> Real Exp(Real f);

inline torch::Half Exp(torch::Half f) { return (torch::Half)expf((float)f); }
inline float Exp(float f) { return expf(f); }
inline double Exp(double f) { return exp(f); }

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


class CombinedSampler {
 public:
  CombinedSampler(uint32_t N, uint32_t M, uint32_t K):
      N_(N), M_(M), K_(K),
      M_unique_(find_prod_unique_prime_factors(M)) {
    std::cerr << "N="<<N <<",M="<<M<<",K="<<K<<std::endl;
    TORCH_CHECK(N < 5);
    TORCH_CHECK((K&(K-1)) == 0);  // require K is a power of 2.
    M_bits_ = FindNumBitsFor(M);
    K_bits_ = FindNumBitsFor(K);


    p_bits_ = std::min(uint32_t(54) / K, // so product of N of these is
                                         // comfortably less than 64, search for
                                         // "headroom"
                       std::min(uint32_t(63) - (K_bits_ * N) / K, // for when we sort `this_sort_P`
                                uint32_t(31) - M_bits_)); // for when we sort `sort_combinations`.

    // TEMP.
    p_bits_ = 8;

    // TODO: allocate buffers..
    uint32_t KpowN = uint32_t(1) << (K_bits_ * N),
        size64 = sizeof(uint64_t) / sizeof(uint32_t);
    int tot_size = (M+1) * N + // P_cumsum_
        std::max<uint32_t>((M+(N-1)*K), // sort_buf32_,
                           K*N) + // indexes_for_samples_
        size64 * KpowN + // sort_buf64_, because double the element size
        size64 * (N+1) +  // P_sum_cumprod_
        (K*N) + // topK_indexes_
        K + // topK_indexes_combined_
        size64 * K + // topK_P_
        size64 * K + // topK_P_exclusive_sum_
        size64 * K + // topK_delta_P_
        size64 * K + // topK_cumsums_
        size64 * K + // sorted_topK_P_
        size64 * K + // sorted_topK_delta_P_
        size64 * (K+1) + // sorted_topK_delta_P_cumsum_, TODO: check size!
        size64 * K + // sorted_topK_cumsums_
        size64 * K + // sorted_topK_cumsums_reduced_
        size64 * K; // unreduced_samples_
    // indexes_for_samples_, of size K*N, shares memory with sort_buf32_.

    buffers_ = std::unique_ptr<uint32_t>(new uint32_t[tot_size]);
    uint32_t *p = buffers_.get();
    SetBuffer(P_cumsum_, p, (M+1) * N);
    sort_buf32_ = p;
    indexes_for_samples_ = p;
    p += std::max<uint32_t>((M+(N-1)*K), K*N);  // indexes_for_samples_
    SetBuffer(sort_buf64_, p, KpowN);
    SetBuffer(P_sum_cumprod_, p, N+1);
    SetBuffer(topK_indexes_, p, K*N);
    SetBuffer(topK_indexes_combined_, p, K);
    SetBuffer(topK_P_, p, K);
    SetBuffer(topK_P_exclusive_sum_, p, K);
    SetBuffer(topK_delta_P_, p, K);
    SetBuffer(topK_cumsums_, p, K);
    SetBuffer(sorted_topK_P_, p, K);
    SetBuffer(sorted_topK_delta_P_, p, K);
    SetBuffer(sorted_topK_delta_P_cumsum_, p, K+1);     // ??
    SetBuffer(sorted_topK_cumsums_, p, K);     // ??
    SetBuffer(sorted_topK_cumsums_reduced_, p, K);     // ??
    SetBuffer(unreduced_samples_, p, K);     // ??

    uint32_t size = p - buffers_.get();
    std::cout << "size = " << size << ", tot_size=" << tot_size;
  }

  void SetBuffer(uint32_t* &buffer, uint32_t* &p, uint32_t size) {
    buffer = p;
    p += size;
  }
  void SetBuffer(uint64_t* &buffer, uint32_t* &p, uint32_t size) {
    buffer = reinterpret_cast<uint64_t*>(p);
    p += 2 * size;
  }


  // returns a pseudo random number coprime to M_, as a function of n.
  // this is deterministic within a batch.
  inline uint32_t GetRandomCoprime(uint32_t n) {
    return 1 + M_unique_ * (rand_source_ >> (M_bits_ * n)) % (M_ / M_unique_);
  }

  template <typename Real, typename AccessorT>
  void LoadP(uint64_t rand_source,
             bool input_is_log,
             AccessorT p) { // p: [N][M]
    rand_source_ = rand_source;

    uint32_t N = N_, M = M_;
    for (uint32_t n = 0; n < N; n++) {
      auto p_n = p[n];

      uint32_t multiple = GetRandomCoprime(n);
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


  void  __attribute__ ((noinline))  Compute() {
    ComputeKLargest();
    ComputePCumsum();
    ComputeBeta();
    ComputeTopkCumsums();  // also re-sorts top-K
    ComputeUnreducedSamples();
    ComputeIndexesForSamples();
    // next is GetWeightsForSamples; this is called by the user.
    // then GetIndexesForSamples; this is called by the user.
  }

  /*
    `weights` is of shape [K].  We write the samples' weights to here.
   */
  template <typename Real, typename AccessorT>
  void GetWeightsForSamples(
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
  void GetIndexesForSamples(
      Accessor1 indexes,  // torch::TensorAccessor32<int64_t, 2>
      Accessor2 combined_indexes) {  // torch::TensorAccessor32<int64_t, 1>
    uint32_t N = N_, M = M_, K = K_;
    for (uint32_t k2 = 0; k2 < K; k2++) { // parallelize over k2, maybe also over n.
      uint64_t combined_index = 0;
      uint32_t M_prod = 1;
      for (uint32_t n = 0; n < N; n++) {
        uint32_t multiple = GetRandomCoprime(n);  // same multiple we used in LoadP().
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
  //  (i) K*p_bits_ <= 54 [an arbitrary choice with 54 << 64,
  //       search for "headroom" to understand]
  //  (ii) (p_bits_*K_bits) <= 63 [1 less than 64 to allow for rounding error].
  //  (iii) also p_bits_ + M_bits_ + 1 <= 32 because of how we
  //      sort to find K-best probs [the +1 is in case some probs or sums of probs
  //      are slightly more than 1 due to rounding.
  uint32_t p_bits_;
  // number of bits used for indexes 0 <= k < K;
  uint32_t K_bits_;


  std::unique_ptr<uint32_t> buffers_;

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
                    search for GetRandomCoprime().
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


  void ComputeKLargest() {
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
      sort_buf[M] = 0;
      // in CUDA we'll just sort the entire array.  Note, we don't need the
      // sorting algorithm to sort the indexes because we include them manually.
      std::nth_element(sort_buf, sort_buf + K, sort_buf + M, std::greater<>());
      std::sort(sort_buf, sort_buf + K, std::greater<>());
    }
    uint64_t *sort_combinations = sort_buf64_;
    uint32_t K_bits = K_bits_, K_bits_mask = (K-1),
        KpowN_bits = K_bits_ * N,
        KpowN = 1 << KpowN_bits;
    for (uint32_t i = 0; i < KpowN; i++) {  // we'll parallelize over i on GPU.
      // product of probabilities.  This index i represents an n-tuple of e.g. k1,k2,k3, which are all indexes in [0..K-1] specifying a
      uint64_t P_prod = 1;
      for (uint32_t n = 0; n < N; n++) {
        uint32_t k = (i >> (n * K_bits)) & K_bits_mask;  // the n'th index 0 <= k < K into K-best.
        uint32_t this_p = (sort_buf32_[K*n + k] >> M_bits); // one of the k-best probs for the n'th softmax
        P_prod *= this_p;
      }
      sort_combinations[i] = (P_prod << KpowN_bits) + i;
    }
    // we'll just sort the entire array, when we do this on GPU.
    std::nth_element(sort_combinations, sort_combinations + K, sort_combinations + KpowN,
                     std::greater<>());
    // work out the K-best combinations.
    std::sort(sort_combinations, sort_combinations + K, std::greater<>());

    uint32_t M_mask = (1 << M_bits) - 1; // M may not be a power of 2, can't use M-1.
    for (uint32_t k = 0; k < K; k++) {  // we'll parallelize over k on GPU.
      uint64_t combination = sort_combinations[k],
          P = combination >> KpowN_bits;
      topK_P_[k] = P;
      uint32_t index_combined = 0;  // will be in [0..M**K-1]
      for (uint32_t n = 0; n < N ; n++) {
        // src_k is the k index among the top-K source items for this `n`.  We
        // need to look up the original 'm' index
        uint32_t src_k = (combination >> (n * K_bits)) & (K-1),
            src_m = sort_buf32_[K*n + src_k] & M_mask;
        topK_indexes_[k*N + n] = src_m;
        index_combined += src_m << (n * M_bits_);
      }
      topK_indexes_combined_[k] = index_combined;
    }
    uint64_t topK_P_sum = 0;
    for (uint32_t k = 0; k < K; k++) {  // this would be done using a cub exclusive-sum.
      topK_P_exclusive_sum_[k] = topK_P_sum;
      topK_P_sum += topK_P_[k];
    }
  }

  void ComputePCumsum() {
    // Compute P_cumsum_, P_sum_cumprod_.

    // P_cumsum_ currently stores integerized probs P, preceded by a zero; after
    // this function it will store the [exclusive] cumulative sum, of size M+1.
    uint32_t *P = P_cumsum_;
    uint32_t M = M_, N = N_;
    uint64_t P_sum_cumprod = 1;
    for (uint32_t n = 0; n < N; n++) {
      // Compute inclusive cumulative sum of size M; we already padded on the
      // left with 0, so the effect is the same as exclusive sum.
      uint32_t *this_P = P + (M+1) * n + 1; // + 1: skip the 0.
      uint32_t sum = 0;
      for (uint32_t m = 0; m < M; m++) {  // note, we wrote 0 at one past the end.
        sum += this_P[m];
        this_P[m] = sum;
      }
      P_sum_cumprod_[n] = P_sum_cumprod;
      P_sum_cumprod *= sum;
    }
    P_sum_cumprod_[N] = P_sum_cumprod;
  }

  void ComputeBeta() {
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
      // one such k index will "work".  We can parallelize over this "k".
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
    // outputs: B_, and the arrays delta_P_ and topK_delta_P.
  }


  void ComputeTopkCumsums() {
    // Computes top-K cumulative sums; these are the cumulative sums of probabilities of
    // all N-tuples of indexes that precede each of the top-K cumulative sums.
    uint32_t M = M_, N = N_, M_bits = M_bits_, M_mask = (1 << M_bits)  - 1, K = K_;

    for (uint32_t k = 0; k < K_; k++) {  // this will be separate kernels
      uint64_t P_selected_laterprod = 1,
          cumsum = 0;

      // This involves summations over the N dimension but we do this sequentially
      // as N will only be 2 or at most 3 in practice.
      uint64_t index_combined = topK_indexes_combined_[k];
      for (int32_t n = int32_t(N) - 1; n >= 0; --n) {
        // 0 <= this_m < M
        uint32_t this_m = (index_combined >> (M_bits * n)) & M_mask,
            this_P_cumsum_idx = (n*(M+1)) + this_m;
        TORCH_CHECK(this_m == topK_indexes_[k*N + n]);

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
    std::sort(topK_cumsums_, topK_cumsums_ + K);

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
    sorted_topK_delta_P_cumsum_[0] = 0;
    for (uint32_t k = 0; k < K; k++) {
      uint32_t delta_P = sorted_topK_delta_P_[k];
      topK_delta_P_cumsum += delta_P;
      sorted_topK_delta_P_cumsum_[k+1] = topK_delta_P_cumsum;
      // TODO: may not need the last element.
    }

    for (uint32_t k = 0; k < K; k++) {
      sorted_topK_cumsums_reduced_[k] = (sorted_topK_cumsums_[k] -
                                         sorted_topK_delta_P_cumsum_[k]);
    }
  }

  void ComputeUnreducedSamples() {
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
      // within
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
  }

  void ComputeIndexesForSamples() {
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
  auto int64_type = torch::kInt64,
      float_type = torch::kFloat32;
  TORCH_CHECK(probs.scalar_type() == float_type);
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
          sampler.LoadP<scalar_t>(rand, input_is_log, probs_a[b]);
          sampler.Compute();
          // torch_scheduled_sampling/sampling_cpu.cpp:565:52: error: no matching function for call to 'CombinedSampler::GetWeightsForSamples(at::TensorAccessor<double, 1, at::DefaultPtrTraits, int>)'
          sampler.GetWeightsForSamples<scalar_t>(weights_a[b]);
          // torch_scheduled_sampling/sampling_cpu.cpp:565:52: note:   'at::TensorAccessor<double, 1, at::DefaultPtrTraits, int>' is not derived from 'at::PackedTensorAccessor32<Real, 1>'
          sampler.GetIndexesForSamples(indexes_a[b], combined_indexes_a[b]);
        }
      }));
  //  std::cout << "combined_indexes = " << combined_indexes;
  return std::vector<torch::Tensor>({indexes, combined_indexes, weights});
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_combined_forward_cpu", &sample_combined_forward_cpu,
        "Multi-softmax sampling function (CPU)");
}
