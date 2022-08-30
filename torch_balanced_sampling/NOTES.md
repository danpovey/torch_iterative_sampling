
 # Some notes on knowledge-bank lookup

[Caution: this NOTES.md is slightly out of date, but still useful in explaining
the basic principles. We make reference in this NOTES.md to doing two sequential
sampling operations, first from the indiviudal distributions and then from the
product of the 1st samples.  In the end we realized that this is not optimal;
what we implemented in sampling_ref.py is based on doing a single sampling
operation, from a joint distribution that's the product of discrete
distributions.  The same basic principles apply but it gets a little more
complicated.  It is all documented in the code though]


This is a sampling operation intended for use in an efficient "knowledge-bank
lookup" operation for neural nets.  The knowledge bank is part of the neural
net's parameters; we avoid using the term "memory" because "memory" generally
refers to previous items in a stream rather than model parameters.



Defining some dimensions with example numbers:

 - M (e.g. 128): size of each softmax input used to access
   the knowledge bank (must be a power of 2, for implementation
   reasons to do with RandomReorder() see below
 - N (e.g. 2): number of softmaxes; should be >1 but quite small, e.g. 2, 3 or 4,
   because the  "knowledge bank" has M^N entries, each of size D.
 - K (e.g. 4): first number of samples
 - L (e.g. 4): second number of samples
 - D (e.g. 256): feature dimension at output of knowledge lookup


  ## Knowledge-bank operation:

  The input would be a tensor of size `(*, N, M)`, e.g. `(*, 2, 128)`, where `*` might be the
  sequence or batch dimension or dimensions.  This is to be interpreted, for each lookup operation,
  as 2 normalized probabilities or log-probabilities of dimension 128.  For exposition
  purposes, assume the input represents probabilities, not log-probs.

  We interpret this as 2 logits of dimension 128; we put that through a softmax
  and interpret the result as a distribution (one that happens to factorize) over
  size 128 x 128.

  The knowledge bank itself is of shape (M**N) x D, e.g. (128*128) x 256,
  interpreted as (M**N) slots each containing a D-dimensional embedding.  The
  output of the knowledge-bank operation is a weighted sum over the slots.  We
  can make this efficient by using a weighted sampling operation to take the
  sum over a very small subset of slots (L slots).

  In order to minimize the variance of the sampling operation, rather than just
  taking L independent samples we sample in a way that always give distinct
  outputs, and the outputs have weights attached.  This is done in such a way
  that the expected output weights are equal to the weights from the softmax.


  ### Un-sampled knowledge-bank lookup

  Below is the un-sampled, softmax-only form of the knowledge-bank operation
  (here specialized for N = 2).  We will later sample this.

<code>
  def access_knowledge_unsampled(
    probs,  # Tensor of shape (*, N, M) e.g. (99, 2, 128), of floats in [0..1]; probs.sum(-1)==1.0
    knowledge, # Shape: (M**N, D)
  ):
    joint_probs = probs[0].reshape(128, 1) * probs[1].reshape(1, 128).reshape(128*128)
    ans = (knowledge_bank * joint_probs.unsqueeze(1)).sum(-2)  # shape (*, D,), e.g. (99, 256,)
    return ans
</code>

 ## Sampled knowledge-bank lookup

<code>
  def access_knowledge_sampled(
    probs,  # Tensor of shape (*, N, M) e.g. (99, 2, 128), of floats in [0..1]; probs.sum(-1)==1.0
    K,      # e.g. K = 4, first num-samples
    L,      # e.g. L = 4, second num-samples
    knowledge, # Shape: (M**N, D)
   ):
    M = probs.shape[-1]
    (i, weight) = SoftSample(probs, K)  # i,weight each of shape: (*, N, K), e.g. (99, 2, 4)
    (i2, weight2) = CombineWeights(i, weight, M)  # i2, weight2 each of shape: (*, K**N), e.g. (99, 16),
                                                  # i2 has elements in [0..M**N - 1]
    (i3, weight3) = SoftSample(weight2, L)  # i3, weight3 each of shape: (*, L), e.g. (99, 4),
                                            # i3 has elements in [0.. K**N - 1]
    (i4, weight4) = (i2[i3]  , weight3)     # i4, weight4 each of shape: (*, L), e.g. (99, 4)
                                            # i4 has elements in [0.. M**N - 1]

    return (knowledge[i4] * weight4.unsqueeze(-1)).sum(dim=-2) # shape: (*, D), e.g. (99, 256)
<code>




 ## SoftSample function, mathematical version

 (i, y) = SoftSample(p, K)

  Inputs:
     p: a vector of shape (*, M) of probabilities 0 <= p <= 1 that sum to 1 along the
        M dimension, with at least K nonzero elements along the M dimension
        [we'll relax the nonzero-element requirement by an epsilon, in the
        practical version.]
     K: an integer with 0 < K < M


 Returns: (i, y), which together represent a sparse output of shape (*, M):
   i: a tensor of int of shape (*, K), of integers in [0..M-1], which will
      be unique along the K dimension.
   y: a tensor of shape (*, K), of floats in [0..1], which will sum to
      one along the K dimension.

   This function will preserve expectations, i.e., E[SoftSample(p)] == p,
   when viewing the output as a sparse tensor of shape (*, M).

 In addition, we would like the SoftSample() operation to produce output that is
 low-variance in some sense.  For now we'll just describe our chosen method,
 which is based on "systematic sampling"*, without much discussion of
 optimality.  I believe it is optimal though, at least when we only consider the diagonal
 of the variance matrix.

 * see: "On the theory of Systematic Sampling" by Madow & Madow, https://www.jstor.org/stable/2236209

 The code will be for just a single sampling operation, i.e. assuming (*) is just the empty shape (,).
<code>
   # We first need to compute a value 0 <= beta <= 1/K such that:
   #   K \beta + \sum_{i: p_i > \beta} (p_i - \beta) = 1     (eqn:0)
   # This divides the probabilities p_i into two groups.  Suppose k values of p_i are
   # > beta.  For these, we sample them with probability 1.  For the rest, we sample
   # them with probability p/beta, and if we choose one of these samples, we return
   # a weight beta instead of p.  There would be (K-k) such samples with value beta,
   # giving total weight (K-k) beta, plus k samples in the sum: \sum_{i: p_i > beta} p_i.
   # We can rearrange this as (eqn:0).
   # The resulting probabilities of sampling various items will be given by r, here
   #   r = min(1, p / \beta)          (eqn:1)
   # See ComputeBeta() [mathematical version] for how beta is computed.
   beta = ComputeBeta(p, K)  # beta: float
   r = min(1, p / beta)   # r: tensor of inclusion probabilities of shape (M,) with sum(r) == K
   t = random integer in {0,1,..(M/2)-1}
   s = 2*t + 1  # s is random in {1,3,5,..M-1}
   rr = Reorder(r, s)
   j = ScheduledSample(rr, K)  # j is a list of integers j_0, j_1, j_{K-1}, with 0 <= j_k < M.
   i = InverseReorderIndexes(j, M, s)

   # divide by inclusion probability (like importance sampling)
   y = p[i] / r[i]
   return (i, y)
 <code>


 Reorder() and InverseReorderIndexes() operation are included to add extra
 randomness-- because the ScheduledSample function, while it gives the correct
 inclusion probabilities, cannot ouptput all possible combinations of input
 symbols.  Reorder() does not allow all possible permutations-- only a subset
 that happens to be easy to implement.  So this whole procedure may lead to very
 slightly larger-than-necessary correlations between inclusion-probabilities of
 different symbols.

================================

 ## ScheduledSampled function [mathematical version]

[See: "On the theory of Systematic Sampling" by Madow & Madow, https://www.jstor.org/stable/2236209]

<code>
  ScheduledSample(p, K):
    # p is a nonnegative vector of size M, containing inclusion-probabilities
    # 0 < p_i <= 1, such that sum(p) == K, with at least K nonzero elements

    # Compute s_i, 0 <= i <= M, as the exclusive sum of p_i, so that
    #  s_i = \sum_{j=0}^{i-1} p_i.
    s = exclusive_sum(p)

    select a uniformly distributed real number 0 <= r < 1
    let i be a vector of size K
    for each k in {0,1,..K-1}:
      set i[k] to the index j in {0,1,...,M-1} such that s_j <= r+k < s_{j+1}
    return i
</code>

 ## Reorder() function:

  y = Reorder(x, s):

    x is a vector of length M, with M a power of 2.
    s is a random integer in {1, 3, ..., M-1}

  For 0 <= i < M:
    y[i] = x[(i*s) % M]

 This function relies on the fact that, because M is a power of 2, multiplying
 by an odd number modulo M is a bijective function.


 ## InverseReorderIndexes() function

  i = InverseReorderIndexes(j, M, s):

  The goal here is to compute the function-inverse of j = (i*s) % M, i.e.
  solve for i in this equation.

<code>
 i = InverseReorderIndexes(j, M, s)
    M is a power of 2
    j is a list of integers of length K, with elements in {0, 1, ..., M-1}
    s is an integer in {1,3,5,...,M-1}.


   Let s_inv = the unique integer in {1,3,...,M-1}, such that (s * s_inv) % M = 1.
   return i = (j * s_inv) % M  # operations applied elementwise
</code>

  On GPU, the fastest way to compute s_inv will likely be exhaustive search.
  Mathematically, we can do:
<code>
       s_inv = (s ** (M/2 - 1)) % M
</code>
  (this relates to Euler's totient function; \phi(M) == M/2 if M is a power of 2).
  A reasonably efficient way to compute this on CPU is:
<code>
       s_inv = compute_inverse(M, s)
</code>
  with compute_inverse as defined below:
<code>
  def compute_inverse(M, u):
    # returns the inverse of u modulo M, if u is coprime to M
    m = 2
    ss = u
    s = u
    while m < M:
       ss = (ss * ss) % M
       s = (s * ss) % M
       m *= 2
    return s
 </code>


 ## Backprop for function SoftSample() [mathematical version]

 The forward pass did:
<code>
   (i, y) = SoftSample(p, K)
</code>
 The function returns (i, y) which are both vectors of length K, of type int and real.
 The backward pass is simply:
<code>
  p_grad = zeros_like(p)
  p_grad[i] = y_grad * y / p[i],
</code>

 The justification for this is as follows.

 The forward pass can be interpreted as:
     z = a random vector whose expected value is [1, 1, 1, 1... ], whose precise
         distribution depends on the input p, it actually contains
          (1/(K*inclusion-probability)) * (1 if included, else 0)
     Y = p z  [where y is the sparse vector with indexes i and specified elements y].

 Then the backprop is:
  p_grad = Y_grad z,
 which can be implemented as:
  p_grad = Y_grad * Y / p

 In expectation over z, this backprop rule is equivalent to just: p_grad =
 Y_grad, which is what we want.  Notice that this backprop rule is different
 from what we'd get if we just naively backpropped the individual instructions;
 in that case we'd zero gradient for those elements with inclusion probability
 1, and larger gradients for the others.  This would not be correct in
 expectation, because we'd be failing to account for the changes in inclusion
 probability.

 A "straight-through" estimate of the derivative would be to use the *expected*
 value of z (i.e. all ones), i.e. to do: p_grad = Y_grad.  But this would require
 us to have derivative values for the zero elements of Y_grad, which would require
 us to do the full (non-sparse) computation, which would defeat the point of
 the sampling.


## ComputeBeta function [mathematical version]

 ComputeBeta(p, K):

 Given a vector of nonzero probabilities 0 <= p <= 1 of size M (that sum
 to one, and with at least K nonzero elements), and an integer 0 < K < M,
 return a value 0 <= \beta <= 1/K such that:

  K \beta + \sum_{i: p_i > \beta} (p_i - \beta) = 1     (see above, this is eqn:0).

 The number of elements k of p such that p_i > beta is going to satisfy
 0 <= k < K (it cannot be K or more because then the LHS of eqn:0 would be >1).
 We will ensure in practice that elements of p are all greater than zero.

 Let q be p after reverse-order sorting (i.e. greatest first), and let s_i be
 the exclusive-sum of q_i, s_k contains the sum of the largest k elements of
 p_i.  Then, if exactly k probabilities are larger than beta, (eqn:0) could only
 be satified if

   s_k + (K-k) beta = 1
   beta = (1 - s_k) / (K-k)

 In addition will require that (k==0 or q_{k-1} > beta) and (q_k <= beta).

<code>
  def compute_beta(p, K):
    q = sorted(p, reverse=True), i.e. sort the elements of p in decreasing order.
    Let s, of dimension M, be the exclusive-sum of elements of q,
    so that s_i = \sum_{j=0}^{i-1} q_i.

    for k in {0,1,..K-1}:
      # note: the constraint sum(r) = K can be expressed as:
      # k + \beta_k (1 - s_k) = K leading to the equation below.
      beta_k = (1 - s_k) / (K - k)        # (eqn:2)
      if (k == 0 or q[k-1] > beta_k) and q[k] <= beta_k:
         return beta_k
</code>


## ComputeBeta function [practical version]:

<code>
  def compute_beta(P, K):
    # [this function returns beta, integerized, but also modifies P slightly.]

    # P is an array of integers P_i that are a fixed-precision representation of
    # p_i, of length M; these are in the range [1..2**31+1], and their sum is
    # not much greater than 2**31 so we can do math in int32.  It may be easier
    # to think of these as un-normalized probbabilities.
    #
    # We return an integer B representing approximately (2**31 * beta), satisfying:
    #
    #      sum(P) ==  K B + \sum_{i: p_i > B} (P_i - B)     (eqn:a1)
    #
    #  ... note: this is after adding n to the one of the smallest elements of P,
    #  with n < K, to ensure exact equality.
    #
    #   ... here, sum(P) takes the role of 1.
    #   Subtracting \sum_{i: P_i > B} (P_i - B) from (eqn:a1), we
    #   can write it more conveniently for the actual sampling operation, as:
    #
    #   sum(min(p_i, B)) - K  = K B                  (eqn:a2)

    R = sorted(P)  # sort in ascending order
    Q = inclusive_sum(R)  # Q[i] = sum_{j=0}^i R[i]
    for k in 0,1,...K-1, in any order:
      # B_k is the value of B if k indexes take the l.h.s. of the "min" expression in min(B, P)
      B_k, remainder_k = Q[M-1-k] // (K - k),  Q[M-1-k] % (K - k)
      if (k==0 or R[M-k] > B_k) and R[M-1-k] <= B_k:
         k_val, B, remainder = k, B_k, remainder_k
         possibly: break
    subtract `remainder` from the M-1-k_val'th element of P
</code>



## SoftSample function, practical version [forward]


 We will be doing the internal computation in finite precision arithmetic, because
 roundoff issues would otherwise be a problem for the algorithm.

 If input_is_log==True, we assume that the input is *normalized* log-probs,
 e.g. the output of log_softmax.  This option is there so that we can
 support float16, because otherwise the backprop of this function might
 generate too-large derivatives.

  For simplicity we assume p is a one-dimensional input.
<code>
  def soft_sample(p: Tensor, K: int, input_is_log: bool):  # returns (index, y)
     if input_is_log:
       p = exp(p)
     # compute unsigned int32 integerized version of p, rounding
     # up (think of this as first adding epsilon=2**-31 to p)
     M = len(p)
     P = floor(p*(2**31) + 1)
     B = compute_beta(P, K)   # [practical version].  B == (2**31 / beta), beta as in (eqn:1)
     inv_beta = float(B) / float(2**31)   # this is 1/beta
     t = a random integer in {0,1,..,M/2-1}
     s = 2*t + 1
     inv_s = the s in {1,3,..M-1} such that (inv_s * s) % M == 1.

     # R is a pseudo-random re-ordering of min(P_i, B)
     R = zeros(M)
     for i in 0..M-1:
        R[i] = min(P[(i*s) % M], B)

     # Note: the rational numbers R / B represent the inclusion probabilities
     # r_i, with sum(r_i) \simeq K.

     Let S be the inclusive-sum of R: S[i] = sum_{j=0}^i R[i]
     Let b be a random integer drawn uniformly from {0, 1, ..., B-1}.

     Allocate "index" and "y" as arrays of length K

     For each i in 0..M, in any order:
       S_prev = (i == 0 ? 0 : S_{i-1}).
       k_prev = (S_prev + b + 1) // B  # integer division, round down or toward zero
       k_cur = (S_i + b + 1) // B      # integer division, round down or toward zero
       if k_cur > k_prev:
         index[k_prev] = (i * inv_s) % M

         # Next implement: [ignoring reordering issues],
         #  y[k_prev] = p_i / r_i,
         # Referring back to (eqn:1)  Since r == min(1, \beta p),
         #  ans = p / r = max(p, 1/beta) (ignoring indexes).
         y[k_prev] = max(p[index[k_prev]], inv_beta)

     Return index, y
 </code>

 The sum of the returned "y" vector be extremely close to 1.0, because:
 with k being the k chosen in ComputeBeta(), [mathematical version]
 r_i will equal 1.0 for k indexes i.  The total of `y` for these k indexes
 is equal to the s_i in ComputeBeta() [mathematical version].
 this satisfies: \beta = (K - k) / (1 - s_k).    (see eqn:2)
 so the total probability mass summed over `y` is as follows:
 Summed over cases where max(p_i, 1/beta) == p_i, sum is:
   s_k
 Summed over cases where max(p_i, 1/beta) == 1/beta, sum is:
   1/beta * (K - k) ==  (1 - s_k)/(K - k) * (K - k) == (1 - s_k)
 So the total of `y` is s_k + (1 - s_k) = 1.


 ## SoftSample function, practical version [backward/backprop]:

  The forward function was:
<code>
   (i, y) = SoftSample(p: Tensor, K: int, input_is_log: bool).
</code>
  The backprop is:
<code>
    p_grad = zeros_like(p)
    if not input_is_log:
      # the "+ 2**-31" is to avoid division by zero; it actually corresponds to
      # how we did the forward pass, where we sampled from (p + 2**-31).
      p_grad[i] = y_grad * y / (p[i] + 2**-31)
    else:
      p_grad[ans_i] = ans_y_grad * ans_y
</code>

Conceptually, we view the forward pass as producing a sparse vector y,
computed as:
<code>
   y = p z
</code>
where p is the input (in probability space, not logs), and z is a sparse
random vector whose expected value is (1,1,1,...).   The backprop is equivalent to:
<code>
  p_grad = z * y_grad,
</code>
and since z = y / p, we can implement this as:
<code>
   p_grad = y_grad * y / p.
</code>
Note: this backprop rule is not the same as what you'd get if you were to
do backprop on the individual statements of the forward pass.  The issue
is that z changes with p, but we can ignore this because the expected value
of z does not change with p.

A "straight-through derivative" approach would just use all-ones in place of
z, i.e. use the expected value of z, but this is not workable in practice here
because we are not going to have derivatives available at the output, w.r.t.
the non-selected indexes in y.
