#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>



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
    TORCH_CHECK(N < 5);
    TORCH_CHECK((K&(K-1)) == 0);  // require K is a power of 2.
    M_bits_ = FindNumBitsFor(N);
    K_bits_ = FindNumBitsFor(K);


    p_bits_ = std::min(uint32_t(54) / K, // so product of N of these is comfortably less than 64, search for "headroom"
                       std::min(uint32_t(63) - (K_bits_ * N) / K, // for when we sort `this_sort_P`
                                uint32_t(31) - M_bits_)); // for when we sort `sort_combinations`.

    // TODO: allocate buffers..
  }



  template <typename Real, typename AccessorT>
  void LoadP(uint64_t rand_source,
             bool input_is_log,
             AccessorT p) { // p: [N][M]
    // torch::PackedTensorAccessor32<Real, 2, torch::DefaultPtrTraits> p) {

    // TODO: maybe in future, use torch::RestrictPtrTraits
    rand_source_ = rand_source;


    // loads P into internal buffer as integers, with reordering.
    // Sets M_reorder_, rand_source_.

    // TODO.

    // TODO: write zero, one past the end.
  }


  void Compute() {
    ComputeKLargest();


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
        uint32_t this_m = indexes_for_samples_[k2*N + n];
        indexes[k2][n] = this_m;
        combined_index += this_m * M_prod;
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


  /* of shape [N][M+1], indexed P_cumsum_[n*(M+1) + m], P_cumsum_ is used
     to store the exclusive cumulative sums of the integerized input probabilities P;
     elements with m==0 are zero, and the last column contains the sums of P. */
  uint32_t *P_cumsum_;

  /*
    Of shape [N+1], P_sum_cumprod_ is the exclusive cumulative product of P_cumsum_[n][M],
    i.e. the cumulative product of the total sum of P which is the last column of P_cumsum_.
   */
  uint64_t *P_sum_cumprod_;

  /*
    of shape [N][M], this is a temporary buffer used to store things we're sorting, in particular
    it is used to store the sorted probability values which we call R, shifted up by  M_bits_,
    plus the corresponding indexes.
    [Note: we can eventually share the space for this with P_cumsum_, since we need them at different
    times.  Although the sorting process would overwrite P_cumsum_, because we store the indexes with
    the probs we have enough information to recover the order; a load, __syncthreads__, recover-indexes,
    and write would do it.
   */
  uint32_t *M_buf1_;  // R_and_indexes_buf_;   k_largest_values_and_indexes_

  /*
    second misc. temporary buffer of shape [max(M, K**N)].
   */
  uint32_t *M_buf2_;
  /*
    buffer of uint64_t of size [K**M], may overlap with some other buffers.
   */
  uint64_t *M_buf64_;

  /*
    of shape [K][N], indexed [k*N + n], topK_indexes_ contains the N-tuples of indexes
    (in 0,1,...,M-1) of the top K combinations of indexes, from highest to lowest
    probability.
   */
  uint32_t *topK_indexes_;

  /*
    This contains the same info as topK_indexes_, but as a single index.
   */
  uint64_t *topK_indexes_combined_;
  /*
    of shape [K], contains from highest to lowest the top-K product probabilities;
    these are products of "P" values.
   */
  uint64_t *topK_P_;

  /*
    Contains B_ which is the integerized beta value.
   */
  uint64_t B_;
  /*
    of shape [K], accessed delta_P_[n*K + k], contains original un-reordered delta_P values as returned
    by compute_beta_prods() in the python version.  These are the amounts of probability mass we remove
    for each of the top-K index-N-tuples to when we assign sampling-probs.

    Originally they are ordered from largest to smallest probs; after ReorderTopk() they are ordered
    from smallest to largest index-tuple [i.e. by topK_indexes_].
   */
  uint64_t *topK_delta_P_;
  /*
    of shape [K+1], delta_p_cumsum_ contains the exclusive cumulative
    sum of delta_P_.
   */
  uint64_t *topK_delta_P_cumsum_;

  /*
    topK_cumsums_, of shape [K], contains the cumulative-sums of products of
    probs [accumulated over all possible products.. we use a formula for this!],
    evaluated at the indexes topK_indexes_.  these are the exclusive cum-sums,
    so they correspond to the beginning of the probability interval, not the
    end.
   */
  uint64_t *topK_cumsums_;

  /*
    Contains the samples that have been shifted to the right by excluding
    disallowed regions.
    Shape: [K2].   K2 is numerically the same as K but refers to the indexes of
        the samples, which are separate from the indexes of the top-K
        index-tuples' probabilities.

   */
  uint64_t *shifted_samples_;

  /*
    Of size [K][N], indexed as indexes_for_samples[k*N + n], contains the indexes
                    in {0..M-1} for the samples in shifted_samples_.
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
    //     and the corresponding [K] probabilities, ordered from largest to smallest.

    uint32_t *sort_P = M_buf1_,
        *P = P_cumsum_;  // currently stores integerized probs P.
    uint32_t M_bits = M_bits_, M = M_, N = N_, K = K_;
    for (uint32_t n = 0; n < N; n++) {
      uint32_t *this_sort_P = sort_P + (n * (M+1)),
          *this_P = P + (n * (M+1));
      for (uint32_t m = 0; m < M; m++) {
        uint32_t p = this_P[m];
        this_sort_P[m] = m + (p << M_bits);
      }
      this_sort_P[M] = 0;
      // note: we are only actually interested in the K-best here.  In principle we could
      // use std::nth_element and then std::sort.  We'll probably just sort when we do it in
      // CUDA though.
      std::nth_element(this_sort_P, this_sort_P + K, this_sort_P + M, std::greater<>());
      std::sort(this_sort_P, this_sort_P + K, std::greater<>());
    }
    uint32_t KpowN = K;  // K ** N
    for (uint32_t i = 1; i < N; i++)
      KpowN *= N;
    uint64_t *sort_combinations = M_buf64_;
    uint32_t K_bits = K_bits_, K_bits_mask = (K-1), KpowN_bits = K_bits_ * N;
    for (uint32_t i = 0; i < KpowN; i++) {
      // product of probabilities.  This index i represents an n-tuple of e.g. k1,k2,k3, which are all indexes in [0..K-1] specifying a
      uint64_t P_prod = 1;
      for (uint32_t n = 0; n < N; n++) {
        uint32_t k = (i >> (n * K_bits)) & K_bits_mask;  // the n'th index 0 <= k < K
        uint32_t this_p = sort_P[(n * M) + k]; // one of the k-best for the n'th softmax
        P_prod *= this_p;
      }
      sort_combinations[i] = (P_prod << KpowN_bits) + i;
    }
    // we'll probably just sort, when we do this on GPU.   But only need top-K.
    std::nth_element(sort_combinations, sort_combinations + K, sort_combinations + KpowN,
                     std::greater<>());
    std::sort(sort_combinations, sort_combinations + K, std::greater<>());
    // OK, now top-K combinations are in `sort_combinations`.

    // TODO: set topK_indexes_, topK_P_.
  }

  void ComputePsum() {
    // P_cumsum_ currently stores integerized probs P, we'll make it store the cumulative sum.
    uint32_t *P = P_cumsum_;
    uint32_t M = M_, N = N_;
    uint64_t P_sum_cumprod = 1;
    for (uint32_t n = 0; n < N; n++) {
      // compute exclusive cumulative sum
      uint32_t *this_P = P + (M+1) * N;
      uint32_t sum = 0;
      for (uint32_t m = 0; m <= M; m++) {  // note, we wrote 0 at one past the end.
        uint32_t next_sum = sum + this_P[m];
        this_P[m] = next_sum;
        sum = next_sum;
      }
      P_sum_cumprod *= sum;
      P_sum_cumprod_[n] = P_sum_cumprod;
    }
    P_sum_cumprod_[N] = P_sum_cumprod;
  }

  void ComputeBeta() {
    // see compute_beta_prods() in sampling_ref.py
    uint32_t N = N_, K = K_;
    uint64_t Psum = P_sum_cumprod_[N];
    // Ptop corresponds to topK_P_[0..K-1]

    uint64_t *Ptop_shifted = 0 ; // TODO. topK_P_, with large values placed first

    uint64_t *Ptop_exclusive_sum = 0 ; // TODO. dim = K + 1, contains [0,.. and exclusive-sum...]
    // ?from sort_combinations_, create Ptop_exclusive_sum_

    uint64_t B, remainder;
    uint32_t k;
    for (k = 0; k < K; k++) {
      // We are trying out to see whether we get an admissible B (i.e.,
      // integerized beta) value with exactly k top probabilities exceeding B.  Exactly
      // one such k index will "work".

      uint64_t this_Ptop_shifted = Ptop_shifted[k],
          this_Ptop = Ptop_shifted[k+1];
      uint64_t S1 = Psum - Ptop_exclusive_sum[k];  // corresponds to 1-s_k in the math.
      uint64_t B_k = S1 / (K-k),
          remainder_k = S1 % (K-k);
      bool is_ok = this_Ptop_shifted > B_k && this_Ptop <= B_k;
      if (is_ok) { // should happen exactly once!!
        B = B_k;
        remainder = remainder_k;
        break;
      }
    }
    B_ = B;
    TORCH_CHECK(k < K);  // check that we broke from the loop.
    uint64_t delta_P_sum = 0;
    for (uint32_t k = 0; k < K; k++) {
      uint64_t Ptop = Ptop_shifted[k+1];
      topK_delta_P_[k] = remainder + Ptop - std::min<uint64_t>(Ptop, B);
      delta_P_sum += topK_delta_P_[k];
    }
    uint64_t err = Psum - delta_P_sum - (B * K);
    TORCH_CHECK(err == 0);
    // outputs: B_, and the array delta_P_.
  }

  void ReorderTopK() {
    // Reorders top-K index-tuples.
    // Let:
    //  sort_combinations_[k] = (topK_indexes_[k] << K_bits_ ) + k,  0 <= k < K.
    // sort sort_combinations_.
    // Set topK_indexes_[k] = sort_combinations_[k] >> K_bits_
    // src_k_idx = sort_combinations_[k] & (K-1)
    // p = topK_P_[src_k_idx]; sync;
    // topK_P[src_k_idx] = p  // <---- Same for topK_delta_P_
    //


  }

  void ComputeTopkCumsums() {
    // Computes top-K cumulative sums; these are the cumulative sums of probabilities of
    // all N-tuples of indexes that precede each of the top-K cumulative sums.

    uint32_t M = M_, M_bits = M_bits_, M_mask = (1 << M_bits)  - 1;
    for (uint32_t k = 0; k < K_; k++) {  // this will be separate kernels
      uint64_t P_selected_laterprod = 1,
          cumsum = 0;


      // This involves summations over the N dimension but we do this sequentially
      // as N will only be 2 or at most 3 in practice.

      uint64_t index_combined = topK_indexes_[k];
      for (int32_t n = int32_t(N_) - 1; n >= 0; --n) {
        // 0 <= this_m < M
        uint32_t this_m = (index_combined >> (M_bits * n)) & M_mask,
            this_P_cumsum_idx = (k*(M+1)) + this_m;

        uint32_t this_P_cumsum = P_cumsum_[this_P_cumsum_idx],
            next_P_cumsum = P_cumsum_[this_P_cumsum_idx + 1],
            this_P = next_P_cumsum - this_P_cumsum;

        uint64_t prev_Psum_cumprod = P_sum_cumprod_[n];

        cumsum += prev_Psum_cumprod * P_selected_laterprod * uint64_t(this_P_cumsum);
        P_selected_laterprod *= this_P;
      }
      topK_cumsums_[k] = cumsum;
    }
  }

  void ComputeShiftedSamples() {
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

      // the following is compute_shifted_samples() in python.
      for (uint32_t k = 0; k < K; k++) {
        uint64_t topK_cumsum_reduced = topK_cumsums_[k] - topK_delta_P_cumsum_[k];
        delta_P_sum += (rand >= topK_cumsum_reduced) * topK_delta_P_[k];
      }
      uint64_t rand_shifted = rand + delta_P_sum;
      shifted_samples_[k2] = rand_shifted;

      for (uint32_t k = 0; k < K; k++) { // check_shifted_samples()
        uint64_t topK_disallowed_start = topK_cumsums_[k],
            topK_disallowed_end = topK_disallowed_start + topK_delta_P_[k];
        TORCH_CHECK(!(rand_shifted >= topK_disallowed_start &&
                  rand_shifted < topK_disallowed_end));
        TORCH_CHECK(rand_shifted < P_sum_cumprod_[N]);
      }
    }
  }

  void ComputeIndexesForSamples() {
    uint32_t K = K_, N = N_, M = M_;
    for (uint32_t k2 = 0; k2 < K; k2++) {
      uint64_t cur_sample = shifted_samples_[k2];

      // we'll parallelize over k2.  We have to do the n index sequentially.
      for (uint32_t n = N-1; ; --n) {  // we break in the loop if n == 0.

        uint64_t P_sum_cumprod = P_sum_cumprod_[n];
        uint32_t this_sample = uint32_t(cur_sample / P_sum_cumprod);

        uint32_t *P_cumsum_start = P_cumsum_ + n * (M+1);
        // This uses find_class() which implements a binary search.
        uint32_t this_m_idx = find_class(P_cumsum_start,
                                         (int)0, (int)M, this_sample);

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

  // returns a pseudo random number coprime to M_, as a function of n.
  // this is deterministic within a batch.
  inline uint32_t GetRandomCoprime(uint32_t n) {
    return 1 + M_unique_ * (rand_source_ >> (M_bits_ * n)) % (M_ / M_unique_);
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

         K: An integer, the number of samples required, with 0 < K < N
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
sample_combined_cpu_forward(torch::Tensor probs, // [B][N][M]
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

  TORCH_CHECK(K > 0 && K < N && ((K&(K-1))==0));  // K is sequence length
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

  return std::vector<torch::Tensor>({indexes, combined_indexes, weights});
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_cpu", &sample_combined_cpu_forward,
        "Multi-softmax sampling function (CPU)");
}
