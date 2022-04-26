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
  assert(end > begin);
  while (end > begin + 1) {
    int mid = begin + (end - begin) / 2;
    if (((uint32_t)cumsum[mid]) <= r)
      begin = mid;
    else
      end = mid;
  }
  return begin;
}



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
inline uint32_t zoom(uint32_t r, uint32_t orig_r,
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

class CombinedSampler {
 public:
  CombinedSampler(uint32_t N, uint32_t M, uint32_t K): N_(N), M_(M), K_(K),
                                                       M_unique_(find_prod_unique_prime_factors(M)) {
    assert(N < 5);
    assert((K&(K-1)) == 0);  // require K is a power of 2.
    M_bits_ = 1;
    while ((1 << M_bits_) < M)
      M_bits_++;
    K_bits_ = 1;
    while ((1 << k_bits_) < K)
      K_bits_++;

    p_bits_ = min(uint32_t(54) / K, // so product of N of these is comfortably less than 64, search for "headroom"
                  min(uint32_t(63) - K_bits_) / K,
                  uint32_t(31) - M_bits_); // so we can sort the indexes and probs in one integer.

    // Allocate buffers..
  }



  template <typename Real>
  void LoadP(torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits> &p,
             bool input_is_log) {
    torch::PackedTensorAccessor<Real, 2, torch::RestrictPtrTraits> &p) {
    // loads P into internal buffer as integers, with reordering.
    // Sets M_reorder_

    // TODO.

    // TODO: write zero, one past the end.
  }


  void Compute() {
    ComputeKLargest();


  }


 private:

  uint32_t N_; // number of softmaxes
  uint32_t M_; // size of each softmax
  uint32_t K_; // number of samples we require per softmax.
  uint32_t M_unique_;  // product of unique prime factors of M_
  uint32_t *M_reorder_; // [K].  pseudo-random numbers coprime to M_ that we use to reorder
                        // m indexes at input and output.

  // number of bits used for storing index 0 <= m < M_ when sorting R_; require (1<<M_bits_) >= M_
  uint32_t M_bits_;
  // number of bits used for probabilities; require:
  //  (i) K*p_bits_ <= 54 [an arbitrary choice with 54 << 64,
  //       search for "headroom" to understand]
  //  (ii) (p_bits_*K_bits) <= 63 [1 less than 64 to allow for rounding error].
  //  (iii) also p_bits_ + M_bits_ + 1 <= 32 because of how we
  //      sort [the +1 is in case some probs or sums of probs are slightly more than
  //      1 due to rounding].
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
    of shape [K], contains from highest to lowest the top-K product probabilities;
    these are products of "P" values.
   */
  uint64_t *topK_P_;

  /*
    Contains B_ which is the integerized beta value.
   */
  uint64_t B_;
  /*
    of shape [K], accessed delta_P_[n*K + k], contains original un-reordered *negated* delta_P values as returned
    by compute_beta_prods() in the python version.  These are the amounts of probability mass we remove
    for each of the top-K index-N-tuples.
   */
  uint64_t *neg_delta_P_;
  /*
    of shape [K], delta_p_reordered_ contains the delta_P_ values reordered by smallest to largest
    index-tuples.
   */
  uint64_t *delta_P_reordered_;
  /*
    of shape [K+1], delta_p_reordered_cumsum_ contains the exclusive cumulative sum of delta_P_reordered_.
   */
  uint64_t *delta_P_reordered_cumsum_;


  /*
    combined_cumsums_ is the
   */
  uint64_t *combined_cumsums_;


  uint32_t find_prod_unique_prime_factors(uint32_t i) { // returns smallest number coprime to
    assert(i != 0);
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
        uint3_t p = this_P[m];
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
        P_prod *= this_p
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
    uint64_t Psum = P_sum_cumprod_[N_];
    // Ptop corresponds to topK_P_[0..K-1]

    uint64_t *Ptop_shifted ... ; // topK_P_, with "large-value" placed first    Let Ptop

    uint64_t *Ptop_exclusive_sum = ... ; // dim = K + 1, contains [0,.. and exclusive-sum...]
    // ?from sort_combinations_, create Ptop_exclusive_sum_

    uint64_t B, remainder;
    for (uint64_t k = 0; k < K; k++) {
      // We are trying out to see whether we get an admissible B (i.e.,
      // integerized beta) value with exactly k top probabilities exceeding B.  Exactly
      // one such k index will "work".

      uint64_t Ptop_shifted = Ptop_shifted[k],
          Ptop = Ptop[k+1];
      uint64_t S1 = Psum - Ptop_exclusive_sum[k];  // corresponds to 1-s_k in the math.
      uint64_t B_k = S1 / (K-k),
          remainder_k = S1 % (K-k);
      bool is_ok = Ptop_shifted > B_k && Ptop <= B_k;
      if (is_ok) { // should happen exactly once!!
        B = B_k;
        remainder = remainder_k;
        break;
      }
    }
    B_ = B;
    assert(k < K);  // check that we broke from the loop.
    uint64_t neg_delta_P_sum = 0;
    for (uint32_t k = 0; k < K; i++) {
      uint64_t Ptop = Ptop_shifted[k+1];
      neg_delta_P_[k] = remainder + Ptop - std::min<uint64_t>(Ptop, B);
      neg_delta_P_sum += neg_delta_P_[k];
    }
    err = Psum - neg_delta_P_sum - (B * K);
    assert(err == 0);
    // outputs: B_, neg_delta_P_.
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

    Returns: (indexes, weights)
       indexes: of shape (B, K, N), for each of K samples from a distribution it contains
            an N-tuple of indexes saying which combination of indexes from the
            component distributions were sampled.
       weights: of shape (B, K), gives the weight associated with each sample,
            which will equal max(p, beta) for a beta specific to the batch element,
            i.e. to the product of the distributions (0 < beta <= 1/K).  The
            weights will sum to 1 along the K axis.
*/
torch::Tensor sample_combined_cpu_forward(torch::Tensor probs, // [B][N]
                                          torch::Tensor rand,   // [B][S]
                                          int K) {
  TORCH_CHECK(probs.dim() == 2, "probs must be 2-dimensional");
  TORCH_CHECK(rand.dim() == 2, "rand must be 2-dimensional");
  auto int32_type = torch::kInt32,
      float_type = torch::kFloat32;
  TORCH_CHECK(probs.scalar_type() == float_type);
  TORCH_CHECK(rand.scalar_type() == int32_type);

  int B = probs.size(0),  // batch size
      N = probs.size(1),  // num classes
      S = rand.size(1);    // num sequences

  TORCH_CHECK(K > 0 && K < N);  // K is sequence length
  TORCH_CHECK(rand.size(0) == B);

  TORCH_CHECK(probs.device().is_cpu() && rand.device().is_cpu(),
              "inputs must be CPU tensors");

  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(probs.device());

  // TODO: make empty
  torch::Tensor indexes = torch::zeros({B, S, K}, long_opts);


  auto probs_a = probs.packed_accessor32<float, 2>();
  auto rand_a = rand.packed_accessor32<int32_t, 2>();
  auto indexes_a = indexes.packed_accessor32<int64_t, 3>();

  // At iteration k, cur_classes[0] contains -1; cur_classes[1,2,..k]
  // contains the previously chosen k classes (all distinct), in sorted
  // order from least to greatest; and cur_classes[k+1] contains N.
  std::vector<int32_t> cur_classes(K + 2);

  // cumsum_row contains exclusive cumumulative sum of probabilities,
  // multiplied by (1<<31) and turned to integer, with one extra element
  // containing the total.
  std::vector<uint32_t> cumsum_row(N + 1);

  // At iteration k, cur_cumsum[c] for 0 <= c <= k contains cumulative
  // probabilities after subtracting the probability mass due to the
  // previously chosen classes (sorted by class-index, not iteration).
  // Implicitly, cur_cumsum[k+1] (not present) contains remaining_prob.
  //
  // At iteration k we have already chosen k classes.  We break up the
  // remaining probability mass (with those k classes removed) into k+1
  // remaining intervals (some possibly empty).  Dwscribing how these
  // remaining intervals are embedded in the interval [0,1], with
  // gaps in between, interval i (with 0 <= i <= k) starts at
  // cumsum[b][cur_classes[i]+1], and ends at
  // cumsum[b][cur_classes[i+1]].
  //
  // In cur_cumsum, we store the cumulative sum *excluding* the intervals
  // corresponding to the previously chosen classes.  On iteration k,
  // cur_cumsum[0] will always contain 0 and cur_cumsum[i] for 0 < i <= k
  // will contain the start of interval i, with the intervals belonging to
  // previously chosen classes subtracted.
  std::vector<uint32_t> cur_cumsum(K + 1);

  for (int b = 0; b < B; ++b) {
    auto this_probs_a = probs_a[b];

    uint32_t tot = 0;
    for (int n = 0; n < N; n++) {
      cumsum_row[n] = tot;
      // The + 1 is to ensure it's nonzero.  This is quite a bit smaller than
      // the floating point epsilon, so we're not concerned about the bias from doing
      // "+" instead of "max".
      uint32_t this_prob = (uint32_t) ((((uint32_t)1) << 31) * this_probs_a[n]) + 1;
      tot += this_prob;
    }
    cumsum_row[N] = tot;


    for (int s = 0; s < S; ++s) {
      auto this_indexes_a = indexes_a[b][s];
      cur_classes[0] = -1;
      cur_classes[1] = N;
      cur_cumsum[0] = 0;

      // at iteration k, remaining_prob contains the sum of the probabilities
      // of classes that have not so far been chosen.
      uint32_t remaining_prob = cumsum_row[N];

      // r is a new random value on {0..1<<31-1}.   We only use one random input for
      // each random sequence; we accomplish this by "zooming in" to each interval
      // that we choose.
      uint32_t r = rand_a[b][s],
          r_orig = r;

      r = zoom(r, r, (uint32_t)0, ((uint32_t)1) << 31, remaining_prob);

      for (int k = 0; k < K; ++k) {
        // Note: at this point, r is in the interval {0..remaining_prob-1}
        int i = find_class(cur_cumsum, 0, k + 1, r);
        // CAUTION: none of these asserts actually get compiled, you have to change them all to
        // TORCH_CHECK if you really want to debug.
        assert(i >= 0 && i <= k);
        // i will now be some value 0 <= i <= k, satisfying
        // cur_cumsum[i] <= r < cur_cumsum[i+1], where,
        // implicitly, cur_cumsum[k+1] == remaining_prob, although
        // actually we never access that element and it is not present.

        // class_range_begin, class_range_end, are the (first,
        // one-past-the-last) class indexes of the range of classes know
        // the k'th randomly chosen class is in.  See the comment above
        // about the k+1 intervals.
        int class_range_begin = cur_classes[i] + 1,
            class_range_end = cur_classes[i + 1];

        assert(class_range_end > class_range_begin && class_range_end <= N);

        // shift r by "adding back" the probability mass due to the subset
        // of previously chosen classes that were numbered less than
        // class_range_begin.  Now r can be compared to elements of
        // `cumsum`.
        uint32_t class_range_begin_cumsum = cumsum_row[class_range_begin];
        r = r - cur_cumsum[i] + class_range_begin_cumsum;

        int c = find_class(cumsum_row,
                           class_range_begin,
                           class_range_end, r);
        assert(c >= class_range_begin && c < class_range_end);

        // c is the class chosen, satisfying cumsum_row[c] <= r < cumsum_row[c +
        // 1].  It will be distinct from all previously chosen classes.
        this_indexes_a[k] = c;

        uint32_t this_class_cumsum = (c == 0 ? 0 : cumsum_row[c]),
            next_class_cumsum = cumsum_row[c + 1],
            this_class_prob = next_class_cumsum - this_class_cumsum;

        assert(r >= this_class_cumsum && r < next_class_cumsum);

        remaining_prob -= this_class_prob;

        r = zoom(r, r_orig, this_class_cumsum, next_class_cumsum, remaining_prob);

        // Update cur_classes and cur_cumsum.
        cur_classes[k + 2] = N;
        // TODO: could unroll the next loop (the speed here is mostly an
        // issue only if K is large).  We are inserting this class and its
        // associated elements in cur_cumsum, in the appropriate place in
        // the list, and shifting later elements to the right while
        // subtracting this class's prob from later elements of
        // cur_cumsum.
        for (int k2 = k; k2 > i; --k2) {
          cur_cumsum[k2 + 1] = cur_cumsum[k2] - this_class_prob;
          cur_classes[k2 + 1] = cur_classes[k2];
        }
        cur_cumsum[i + 1] = cur_cumsum[i] + (this_class_cumsum -
                                             class_range_begin_cumsum);
        cur_classes[i + 1] = c;
      }
    }
  }
  return indexes;
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_cpu", &sample_cpu, "Iterative sampling function (CPU)");
}
