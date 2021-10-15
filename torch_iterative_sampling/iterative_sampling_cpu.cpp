#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>






/*
  Return the index i into cumsum, with begin <= i < end,
  such that cumsum[i] <= r < cumsum[i + 1].
  We assume that cumsum[begin] <= r < cumsum[end], and we do not
  access cumsum[begin] or cumsum[end].
*/
template <typename IterType, typename ScalarType> int find_class(
    IterType cumsum, int begin, int end, ScalarType r) {
  assert(end > begin);
  while (end > begin + 1) {
    int mid = begin + (end - begin) / 2;
    if (cumsum[mid] <= r)
      begin = mid;
    else
      end = mid;
  }
  return begin;
}




/*
  Randomly discretize the weight `weight` (which we expect to satisfy 0 <= weight <= 1),
  into a value 0 <= ans <= 1 which is limited to M discrete increments.

    weight: a value satisfying 0 <= weight <= 1
       M:  an integer greater than 1, e.g. 256.
       r:  a pointer to a a random value 0 <= *r <= 1, that will determine the random
           component of the discretized value.  At exit, it will be modified to a new
           value 0 <= *r <= 1 that can be treated as a "new" random value (by zooming
           into the interval we chose).
   Returns:
      A discretized value in [0,1] which will be a multiple of 1/(M-1).
 */
template <ScalarType> inline scalar_t discretize_weight(scalar_t weight, int M, scalar_t *r) {
  assert(weight >= 0 && weight <= 1.0);
  assert(*r >= 0 && *r <= 1.0);
  // e.g. if M == 128, we divide the range [0..1] into 127 equal intervals, so that we
  // can represent both 0 and 1.
  // Note: 0 <= weight_scaled <= M - 1.
  scalar_t weight_scaled = weight * (M - 1);
  int lower_val = (int) weight_scaled;  // rounds down.

  scalar_t residual = weight_scaled - lower_val;
  // We know `residual` cannot be exactly equal to 1.0, or lower_val would have
  // been larger by 1.
  assert(residual >= 0.0 && residual < 1.0);
  if (*r < residual) {
    // i.e.: with probability (residual)
    // make r a "new" random value by zooming into the interval [0,residual]
    // Note: since *r < residual and *r >= 0, we know residual > 0, so
    // division by zero is not a problem here.
    *r = *r / residual;
    // Note: we cannot reach here if lower_val == M - 1 (i.e. we cannot return a
    // value greater than 1).  This is because if lower_val equals M - 1,
    // `weight_scaled` must be *exactly* M - 1 (because weight <= 1.0),
    // hence `residual` must be 0.  So cannot have *r < residual, because
    // we know r >= 0.
    return (lower_val + 1) / scalar_t(M - 1);
  } else {
    // alternatively, with probability (1-residual):
    //
    // make r a "new" random value by zooming into the interval [residual,1].
    // Division by zero cannot happen here because we know that residual < 1.0.
    *r = (*r - residual) / (1.0 - residual);
    assert(*r >= 0.0 && *r <= 1.0); // TODO: remove assertion.
    return lower_val / scalar_t(M - 1);
  }
}


template <ScalarType> inline void wrap_if_outside_unit_interval(ScalarType *r) {
  if (*r > 1.0 || r < 0.0) {
    // should be very rare.
    printf("iterative_sampling_cpu.cpp: warning: wrapping %f\n", (float)(*r));
    // mathematically, r should still be in the range [0,1]; we wrap
    // around like this just in case of roundoff errors.
    if (*r < 0.0)
      *r = -*r;
    *r = (*r - (int)*r);
  }
}



/*
  iterative_sample function.

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
torch::Tensor iterative_sample_cpu(torch::Tensor cumsum,
                                   torch::Tensor rand,
                                   int K) {
  TORCH_CHECK(cumsum.dim() == 2, "cumsum must be 2-dimensional");
  TORCH_CHECK(rand.dim() == 2, "rand must be 2-dimensional");

  int B = cumsum.size(0),  // batch size
      N = cumsum.size(1),  // num classes
      S = rand.size(1);    // num sequences

  TORCH_CHECK(K > 0 && K < N);  // K is sequence length
  TORCH_CHECK(rand.size(0) == B);

  TORCH_CHECK(cumsum.device().is_cpu() && rand.device().is_cpu(),
              "inputs must be CPU tensors");

  auto scalar_type = cumsum.scalar_type();  // presumably float or double

  auto opts = torch::TensorOptions().dtype(scalar_type).device(cumsum.device()),
      long_opts = torch::TensorOptions().dtype(torch::kInt64).device(cumsum.device());

  torch::Tensor indexes = torch::empty({B, S, K}, long_opts);

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "iterative_sampling_cpu_loop", ([&] {
        auto cumsum_a = cumsum.packed_accessor32<scalar_t, 2>(),
            rand_a = rand.packed_accessor32<scalar_t, 2>();
        auto indexes_a = indexes.packed_accessor32<int64_t, 3>();


        // At iteration k, cur_classes[0] contains -1; cur_classes[1,2,..k]
        // contains the previously chosen k classes (all distinct), in sorted
        // order from least to greatest; and cur_classes[k+1] contains N.
        std::vector<int32_t> cur_classes(K + 1);

        // At iteration k, cur_cumsum[c] for 0 <= c <= k contains cumulative
        // probabilities after subtracting the probability mass due to the
        // previously chosen classes (sorted by class-index, not iteration).
        // Implicitly, cur_cumsum[k+1] (not present) contains 1 - chosen_sum.
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
        std::vector<scalar_t> cur_cumsum(K);

        for (int b = 0; b < B; ++b) {

          auto this_cumsum_a = cumsum_a[b];

          for (int s = 0; s < S; ++s) {
            auto this_indexes_a = indexes_a[b][s];
            cur_classes[0] = -1;
            cur_classes[1] = N;
            cur_cumsum[0] = 0.0;

            // at iteration k, chosen_sum contains the sum of the probabilities
            // of the classes chosen on iters 0,1,..k-1.
            scalar_t chosen_sum = 0.0;

            // r is a new random value on [0..1].  We only use one random input for
            // each random sequence; we accomplish this by "zooming in" to each interval
            // that we choose, rescaling r each time so that the interval we just chose
            // corresponds to the interval [0,1].
            scalar_t r = rand_a[b][s];

            for (int k = 0; k < K; ++k) {
              // Note: at this point, r is in the interval [0,1-chosen_sum]
              int i = find_class(cur_cumsum, 0, k + 1, r);
              // i will now be some value 0 <= i <= k, satisfying
              // cur_cumsum[i] <= r < cur_cumsum[i+1], where,
              // implicitly, cur_cumsum[k+1] == 1-chosen_sum, although
              // actually we never access that element and it is not present.

              // class_range_begin, class_range_end, are the (first,
              // one-past-the-last) class indexes of the range of classes know
              // the k'th randomly chosen class is in.  See the comment above
              // about the k+1 intervals.
              int class_range_begin = cur_classes[i] + 1,
                  class_range_end = cur_classes[i + 1];

              // shift r by "adding back" the probability mass due to the subset
              // of previously chosen classes that were numbered less than
              // class_range_begin.  Now r can be compared to elements of
              // `cumsum`.
              scalar_t class_range_begin_cumsum = this_cumsum_a[class_range_begin]
                  r = r - cur_cumsum[i] + class_range_begin_cumsum;

              int c = find_class(this_cumsum_a,
                                 class_range_begin,
                                 class_range_end, r);
              // c is the class chosen, satisfying this_cumsum_a[c] <= r <
              // this_cumsum_a[c+1], where implicitly this_cumsum_a[N] == 1.0.
              // It will be distinct from all previously chosen classes.
              this_indexes_a[k] = c;

              scalar_t this_class_prob = (c + 1 == N ? 1.0 : this_cumsum_a[c + 1]) - this_cumsum_a[c];
              r = (r - this_cum_sum_a[c]) / this_class_prob;
              // mathematically, r should be in [0,1]; but make sure of this in case,
              // due to roundoff, it is just outside that interval.
              wrap_if_outside_unit_interval(&r);
              // We can now treat r as a "new" random value on [0,1].


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
              cur_cumsum[i + 1] = cur_cumsum[i] + (this_cumsum_a[c] -
                                                   class_range_begin_cumsum);
              cur_classes[i + 1] = c;


              chosen_sum += this_class_prob;
              // Reduce the random value r so that it is in the range
              // [0..1-chosen_sum], which is the size of the reduced interval
              // after subtracting the probability mass due to previously chosen
              // classes.  On the next iteration, we will search within this
              // reduced interval.
              r = r * (1.0 - chosen_sum);
            }
          }
        }
      }));
  return indexes;
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iterative_sample_cpu", &iterative_sample_cpu, "Iterative sampling function (CPU)");
}
