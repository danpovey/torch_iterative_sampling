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
  Return the index i into cumsum, with begin <= i < end,
  such that cumsum[i-1] <= r < cumsum[i].
  We assume that cumsum[begin-1] <= r < cumsum[end-1], and we do not
  access cumsum[begin-1] or cumsum[end-1].

  This is like find_class(), but with a -1 offset on the indexes into cumsum.
*/
template <typename IterType> int find_class_offset(
    IterType cumsum, int begin, int end, uint32_t r) {
  assert(end > begin);
  while (end > begin + 1) {
    int mid = begin + (end - begin) / 2;
    if (((uint32_t)cumsum[mid - 1]) <= r)
      begin = mid;
    else
      end = mid;
  }
  return begin;
}




/*
m  Zoom into an interval.  What we are trying to emulate here is something like this:

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


/*
  iterative_sample function.

    cumsum: (inclusive) cumulative probabilities of input classes, of shape (B, N),
        where B is the batch size and N is the number of classes, so element
        cumsum[b][k] is the sum of probs[b][i] for 0 <= i < k.  Must be of
        type int32_t; we recommend to convert probs to int32_t by multiplying
        by (1<<31) and converting to integer, then computing cumsum.
        Any overflow to negative values will not matter, as we convert to
        unsigned for math.

     rand:  random int32_t integers uniformly distributed on {0..1<<31 - 1}, of shape (B,S), where
         S is the number of separate sequences of samples we are choosing from
         each distribution in `cumsum`.
       K: length of the random sequence to draw; must satisfy 0 < K < N.

  Returns:  Tensor of shape (B, S, K) and type torch::kInt64, containing,
            for each B, a squence of K distinct sampled integers (class
            labels) in the range [0..N-1], drawn with probability proportional
            to the differences between `cumsum` elements, but always excluding
            previously drawn classes within the current sequence.
*/
torch::Tensor iterative_sample_cpu(torch::Tensor cumsum,  // [B][N]
                                   torch::Tensor rand,   // [B][S]
                                   int K) {
  TORCH_CHECK(cumsum.dim() == 2, "cumsum must be 2-dimensional");
  TORCH_CHECK(rand.dim() == 2, "rand must be 2-dimensional");
  auto int32_type = torch::kInt32;
  TORCH_CHECK(cumsum.scalar_type() == int32_type);
  TORCH_CHECK(rand.scalar_type() == int32_type);

  int B = cumsum.size(0),  // batch size
      N = cumsum.size(1),  // num classes
      S = rand.size(1);    // num sequences

  TORCH_CHECK(K > 0 && K < N);  // K is sequence length
  TORCH_CHECK(rand.size(0) == B);

  TORCH_CHECK(cumsum.device().is_cpu() && rand.device().is_cpu(),
              "inputs must be CPU tensors");

  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(cumsum.device());

  // TODO: make empty
  torch::Tensor indexes = torch::zeros({B, S, K}, long_opts);


  auto cumsum_a = cumsum.packed_accessor32<int32_t, 2>(),
      rand_a = rand.packed_accessor32<int32_t, 2>();
  auto indexes_a = indexes.packed_accessor32<int64_t, 3>();

  // At iteration k, cur_classes[0] contains -1; cur_classes[1,2,..k]
  // contains the previously chosen k classes (all distinct), in sorted
  // order from least to greatest; and cur_classes[k+1] contains N.
  std::vector<int32_t> cur_classes(K + 2);

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
    auto this_cumsum_a = cumsum_a[b];

    for (int s = 0; s < S; ++s) {
      auto this_indexes_a = indexes_a[b][s];
      cur_classes[0] = -1;
      cur_classes[1] = N;
      cur_cumsum[0] = 0;

      // at iteration k, remaining_prob contains the sum of the probabilities
      // of classes that have not so far been chosen.
       uint32_t remaining_prob = this_cumsum_a[N - 1];

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
        uint32_t class_range_begin_cumsum = (class_range_begin == 0 ? 0 :
                                             this_cumsum_a[class_range_begin - 1]);
        r = r - cur_cumsum[i] + class_range_begin_cumsum;

        int c = find_class_offset(this_cumsum_a,
                                  class_range_begin,
                                  class_range_end, r);
        assert(c >= class_range_begin && c < class_range_end);

        // c is the class chosen, satisfying this_cumsum_a[c-1] <= r <
        // this_cumsum_a[c], treating this_cumsum_a[-1] as 0.
        // It will be distinct from all previously chosen classes.
        this_indexes_a[k] = c;

        uint32_t this_class_cumsum = (c == 0 ? 0 : this_cumsum_a[c - 1]),
            next_class_cumsum = this_cumsum_a[c],
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
  m.def("iterative_sample_cpu", &iterative_sample_cpu, "Iterative sampling function (CPU)");
}
