#include <math.h>  // for log1p, log1pf
#include <torch/extension.h>




/*
  BACKPROP NOTES

  This comment describes how we do the backprop for this sampling algorithm,
  and describes the thought behind it.  The idea is that we treat the probability
  mass as a fluid, and look at how the fluid flows as we change the probabilities;
  this allows us to compute derivatives.

  PRELIMINARIES:

  We are sampling from a categorical distribution, and we want
  the process to be differentiable ("correctly", not using approximations such
  as treating a discontinuous function as being differentiable).  To do this,
  it's necessary for the forward algorithm to sometimes give an output that's
  not at the vertices, i.e. an output other than the one-hot vectors [ 1 0 0
  .. ], [ 0 1 0 ... ], and so on.  The simplest way we can think of is to, some
  of the time, interpolate between these one-hot vectors, i.e. with probability
  `interp_prob` (e.g. interp_prob=0.25), we interpolate between two
  independently chosen one-hot vectors.  The back-propagated derivatives would
  only be nonzero for these 'interpolated' vectors, and zero at the vertices.


  JUSTIFICATION FROM INFORMATION THEORY; COMPARISON WITH GUMBEL SOFTMAX

  Part of the thinking behind this is to avoid 'information leakage' (should
  reference the information bottleneck theory of DNNs, see
  https://arxiv.org/pdf/1503.02406.pdf).  Any time
  the output is a combination of different one-hot vectors, the next layers of
  the network get a hint about which categories were probable in the categorical
  distribution and the more vectors are nonzero, the stronger the hint.  By
  limiting the output to (one vector most of the time) and (two vectors some of
  the time), the idea is that we minimize the amount of information that can pass
  through the sampling algorithm.  In contrast, can show that the Gumbel-Softmax
  approach does not limit the information passing through at all: we can multiply
  the log-probs at the input by a large number, do Gumbel-Softmax, and take the
  log at the output and scale it down, recovering the input with increasing
  precision as the input scale gets small).  Let the input to the network
  be X and the output of the sampling layer be Y.  Let I(X; Y) be the mutual information
  if we just did sampling and no interpolation (this is bounded by the log of
  the number of classes).  We can show that

       I_{sampling}(X; Y)  <=  (1+interp_prob) I(X; Y)

  ... the idea is that when we interpolate, the only information the output
  sees is the identities of the two samples we took (the interpolation weights
  we chose are uncorrelated with the input X, so carry no information), so
  it's equivalent to sampling twice.  This bound is not tight; but the point is,
  the LHS approaches I(X; Y) as interp_prob --> 0, whereas as the Gumbel Softmax
  tau approaches zero we cannot show any such thing (at least, not without
  invoking things like finite precision arithmetic).


  NOTATION

  Suppose the discrete distribution is over symbols S = { a, b, c.. }, with |S|
  = N (the number of symbols).  We have a discrete variable X that can take on
  values x, e.g. x=a, x=b, and so on.  There is an input distribution p(x).
  This corresponds to a single row of the input matrix.  As explained above, we
  plan to sometimes return a one-hot vector and sometimes (with probability
  interp_prob) return an interpolation between two one-hot vectors.  We do this
  by first drawing two discrete samples x1 and x2 (these take on value a, b, and
  so on); and a random interpolated position between them 0 <= d <= 1, where d=0
  corresponds to symbol x1 and d=1 corresponds to being at x2.  Let u_{x1} and
  u_{x2} be the unit vectors along axes x1 and x2, respectively.

  The value we return for this row is an interpolation between x1 and x2:
     y =  (1-f(d)) u_{x1}  + f(d) u_{x2} ,
  where we define the function f as follows, on the domain 0 <= d <= 1
  (it is a continuous piecewise linear function that increases from 0 to 1).
  Define  lower_knee = 0.5*(1-interp_prob), and
          upper_knee = 0.5*(1+interp_prob)
  which are the two discontinuitities in f().  Then:
      f(d) = 0,  if   0 <= d <= lower_knee
      f(d) = 1,  if   upper_knee <= d <= 1
      f(d) = (d - lower_knee) / interp_prob, if   lower_knee < d < upper_knee
                        (0) <-- starting eqation numbers from 0 to avoid
                                renumbering later ones.

  The reason we use this f(d) to make the position only sensitive to
  d in the "middle", rather than the simpler approach of:
  with probability(interp_prob), select d, and let d itself be the
  interpolation position, is to reduce the variance of the derivatives as
  much as possible.  It will turn out later that the values of the
  derivatives vary with this position d.

  In order to differentiate this sampling function, we need to know how d
  changes as the change the probability distribution p(x).  As far as the
  forward propagation is concerned, d is not affected by p(x), so we'd get zero
  derivative.  We'll explain how we compute the derivative regardless of this;
  it involves a physical analogy.

  Imagine there are points labeled with the discrete symbols a, b, c ... ,
  that are the connection points for idealized pipes containing a fluid.
  We have a pipe corresponding to each ordered
  pair x1->x2, e.g. a->a, a->b, a->c, b->a and so on, amounting to N^2 pipes
  in total.  The pipes have unit length and cross-sectional areas p(x1)p(x2)
  for the pipe betwen point x1 to point x2, e.g. the pipe from a to b has
  cross-sectional area p(a)p(b).  You can see that the total volume of fluid
  is equal to 1.0.  (Don't worry how the points and pipes might be laid out in
  3-dimensional space, this is just a concept).

  In our algorithm, we randomly choose the symbols x1 and x2, and then randomly
  choose 0 <= d <= 1.  d represents the position along the pipe from x1 to x2,
  where d=0 would mean we are at x1 and d=1 means we are at x2.  You can think
  of this as placing an ink spot at a randomly chosen location within the total
  volume of fluid; this works because the cross-section area of each pipe equals
  p(x1)p(x2).  Then we consider what happens if we change the probability
  distribution p(x), modifying the diameters of the pipes.  We figure out how
  the fluid would have to flow in order to conform to the resized pipes, and
  that will tell us how our particle moves as p(x) changes.  When
  the fluid flows from one pipe to another at the connection points, although
  the name of the pipe changes discontinuously the output does not (since
  at that point it is determined only by the connection point); so the output
  varies continuously as the location of the particle changes.

  Consider a small section of the pipe that goes from point a to point b (for example):
  the section is from position d to d+\delta_d, where 0 <= d < 1 and \delta_d is small.
  The volume of this section equals (surface area * length) = p(a)p(b)\delta_d.
  We're going to consider small changes in p(a) and p(b), but we can't consider
  these changes independently because we need the total volume to remain equal
  to 1.0.  The easiest way to set this up is to imagine that the probabilities
  vary with time t.  Let p(a,t) by itself represent the value of probability
  p(a) at time t, and p'(a,t) be the derivative w.r.t. t, of p(a), taken at time t.
  Let v represent the volume of our small section, so we can write:

      dv/dt = d/dt  p(a,t)p(b,t)\delta_d
            = (p'(a,t)p(b,t) + p'(b,t)p(a,t)) \delta_d    (1)

  We can obtain a flow consistent with these changes in volume, as follows.
  Using the pair of values a,b as an example, let the following represent
  the rate of volume flow along the pipe from a to b, at position d and time
  t (with positive values indicating a flow from a to b):

     f_{a,b}(d,t) = (1-d) p(a,t)p'(b,t)  -  d p(b,t)p'(a,t).   (2)

  At this point, we just invented this formula out of thin air, as a proposal
  for what the rate of flow would be; but we can show that it's consistent with
  our required changes in volume.  One constraint is that the total flow into
  each of the points (say, point a) is zero.  There are two quite different ways
  to show this; one is that the total flows from the left and right sides of the
  above equation, into each point, cancel; another is that the total flows from
  the left and right sides of the above equation are both equal to zero (this
  requires that \sum_x p'(x,t) = 0, since the distribution must always sum to
  one).

  The next requirement is that the rate of flow into our small section of
  pipe should be equal to (1).  This rate of flow can be written as:

   dv/dt =  f_{a,b}(d,t) - f_{a,b}(d+\delta_d,t) ,

  where the first term represents flow into the section from the "a" side, and
  in the second we subtract flow out of the section to the "b" side.  Expanding this
  using (2), we get:


     dv/dt =  (1-d) p(a,t)p'(b,t)                -  d p(b,t)p'(a,t).
              -(1-(d+\delta d)) p(a,t)p'(b,t)    +  (d+\delta d) p(b,t)p'(a,t).
            = (p(a,t)p'(b,t) +  p(b,t)p'(a,t)) \delta_d

  which is the same as (1), conforming that this flow is consistent with our
  required changes in volume.   We should be able to show that (2) is in some
  sense the minimal flow required to satisfy this condition.
  The volume flow in (2) does not reflect the rate of motion of our particle
  placed in fluid.  We need the flow velocity, which is the volume flow f
  divided by the surface area p(a,t)p(b,t).  Let us use the notation
  d'(d,t) to represent the rate of change of position with time (i.e. velocity),
  so:

         d'(d,t) = ((1-d) p(a,t)p'(b,t)  -  d p(b,t)p'(a,t)) / (p(a,t)p(b,t))
                 = (1-d)p'(b,t)/p(b,t)   -  d p'(a,t)/p(a,t) .       (4)

  Dispensing now with the concept of time, (4) can be interpreted as saying how
  the position d (of our ink spot in the fluid) varies with p(b) and with p(a),
  namely, so that on the pipe from a to b:

    \frac{\partial d}{\partial p(b)} =  (1-d)/p(b)
    \frac{\partial d}{\partial p(a)} = -d/p(a) .           (5)

  ... although we need to be careful with these partial derivative expressions since they
  are only meainingful if there is no total volume change, i.e. \sum_x p'(x) = 0.
  We can actually get rid of the division by p(b) and p(a), which can be numerically
  problematic, by propagating the derivative back to the un-normalized log-probs,
  treating the actual probabilities as a temporary that is optimized out.  That is, if we
  define
          p(x) = exp(l(x) - z),
  where l(x) is the un-normalized logprob for symbol x, and z is the log normalizer
          z = log \sum_x exp(l(x))
  with the summation taken over all symbols x={a,b,c...}, then:

     dp(x)/dl(x) = exp(l(x) - z) = p(x)
     dp(x)/dz = -exp(l(x) - z) = -p(x).              (6)

   Then we can combine (6) with (5) to get:

    \frac{\partial d}{\partial l(b)} =  (1-d)
    \frac{\partial d}{\partial l(a)} = -d
    \frac{\partial d}{z}             = 2d - 1         (7)

  (The partial derivatives w.r.t l(b) and l(a) above do not include the indirect
  terms via z).  Note: the significance of the symbols a and b here is: we
  assume we are on the "pipe" from a to b, i.e. the forward pass we randomly
  sampled with x1=a and x2=b.  So really a and b are just stand-ins for the
  randomly chosen symbols x1 and x2.

  For all symbols x, \frac{\partial z}{\partial l(x)} = p(x) by properties
  of softmax.  So combining this with the expression for \frac{\partial d}{z} in (7), we
  can set the derivatives of our position d w.r.t. the un-normalized log-likelihoods l(x)
  as follows:

    \frac{\partial d}{\partial l(x)} = (2d - 1) p(x), for all x
    \frac{\partial d}{\partial l(b)} +=  (1-d)
    \frac{\partial d}{\partial l(a)} += -d                    (8)

  NOTES ON PROOF:

   The above has been presented as an intuitive argument rather than a proof
   that the derivatives are correct in a specific sense.  Here we sketch out
   how we'd go about provings the correctness of our derivatives.

   We propose to do it without reference to the random inputs, but purely
   as an argument on the distribution of outputs.  Consider a distribution parameterized
   by \theta \in {\cal T}, where we assume {\cal T} \in \Re^M (this
   would correspond to the logits l or the probabilities p, in our example), that has
   a measure P_\theta; we assume the underlying set for the output
   space \Omega = \Re^N.
   Let a derivative for this distribution be a linear function:

    f_{\theta,\omega}:  \Re^N \rightarrow \Re^N

  that maps from the tangent space of {\cal T} to \Re^N (interpreted
  as the tangent space of {\Omega}).
  We propose a definition that says this derivative is admissible if
  for each \theta and each direction of change t_\theta in the tangent
  space of {\cal T} at \theta, and for any \epsilon > 0, we can choose
  a \delta > 0 such that
  the Wasserstein/earch-mover's distance between the "shifted distribution"
     Q_{\theta,\delta t_\theta} and P_{\theta + \delta t_\theta}
  is less than $\epsilon \delta$.  (We can use the Euclidian metric on \Re^N).
  We define the "shifted distribution"
     Q_{\theta,\delta t_\theta}
  as the result of applying the map g:
      x : x + f_{\theta, \omega} (\delta t_\theta)
  to elements of the space \Omega, where the measure Q_{\theta,\delta t_\theta}(\mu)
  for a set \mu is defined as the measure $P_\theta$ of the preimage of \mu under g.

  In general, multiple distinct derivatives in this sense will be admissible.
  Imagine two tanks connected in parallel by two pipes.  We can shift water
  from one tank to the other using either pipe, or both.  We can create a definition
  of an "optimal" derivative by saying a derivative is optimal if, among
  admissible derivative functions, it has the lowest expected energy where
  the energy is defined as the expected squared velocity:

    Energy =  E_\mu [ || f_{\theta, \omega} (\delta t_\theta) ||^2 ]

  where || . || is the Euclidean 2-norm (here taken on directions in the output
  space, which we interpret as velocities).  This must be true
  \theta and every t_\theta in the tangent space at \theta, for the derivative
  function to be optimal.  We can thus refer to the optimal admissible
  derivative of a parameterized probability measure (assuming it admits any
  derivative) as "the" derivative.

  We can show that the energy is minimized if we can express the velocity as
  the derivative of of a potential function.  This potential function would
  be interpreted as a Lagrangian constraint that the probability distribution
  changes in the correct way as we change \theta, in a constrained optimization
  where we are minimizing the expected energy subject to constraints on the
  change in probability distribution.  Since the energy function is strictly
  convex in the space of derivatives, there can be only one such minimum;
  therefore the energy is minimized if we can express the velocity as the
  derivative of some potential function.  Consider our equation (4): repeating
  it here,

         d'(d,t) = (1-d)p'(b,t)/p(b,t)   -  d p'(a,t)/p(a,t) .

  We can interpret d', on the left, as a velocity.  So we want to be able to
  interpret the RHS of the equation above as the derivative of some potential.
  In fact, this is quite easy to do: we can just let the potential function
  $\Phi(x)$ at position x \in \Omega, be:

    \Phi(x) = \sum_i (squared distance of x from i) * p'(i,t)/p(i,t)

  where i ranges over the symbols a,b,c and so on.  The specific details
  would require a little setting up (e.g. there may be a factor of \sqrt{2} to
  take account of in the distances, since the Euclidiean
  distance between two distinct one-hot vectors is \sqrt{2}).
  Thus we can see that our formula for the derivative does indeed, return
  "the" derivative.

  Depending how P varies with \theta, no derivative at all may be admissible.
  One potential reason for this is if P varies discontinuously with \theta.
  Another difficulty can arise when the probability measure at the output
  is "disconnected": you can't fill one tank from another unless the tanks
  are connected by a pipe.


 SUMMARY:

   This section briefly summarizes the forward and backward passes, in a
   mathematically idealized way without considering practicalities like
   roundoff.  Here we consider a single row of the matrix of logprobs.  To
   clarify that x is discrete, we assume it is an integer and rename it
   to i here.

   FORWARD PASS INPUTS:
     Un-normalized log-likelihoods l(i) for 0 <= i < N.
     The interpolation probability 0 < interp_prob <= 1
   FORWARD ALGORITHM
     Compute the log-probs p(i) = exp(l(i)) / \sum_j exp(l(j))
     Randomly choose two symbols i1 and i2 with probabilities proportional
     to p(i).
     Randomly choose a real value d, uniformly distributed on [0,1].
     Compute e = f(d) which is a continuous piecewise linear function
     as defined above (search for f(d)).  Will satisfy 0 <= e <= 1.

     Set o(i) = 0 for all i, then:
     o(i1) += (1-e)
     o(i2) += e
     Store i1, i2, d, and e; and p(i), 0 <= i < N, for the backward pass.
     Return o

   BACKWARD PASS INPUTS:
     The inputs to the backward pass are:
       - The probabilities p(i), 0 <= i < N, stored from the forward pass
       - i1, i2, d, and e, stored from the forward pass
       - The derivative of our loss function w.r.t. the output o;
         call this o_grad(i), for 0 <= i < N.
     The backward pass will return the loss derivatives w.r.t.
     the input logits l(i); call this l_grad(i).
   BACKWARD ALGORITHM:
     if i1 == i2 or e == 0 or e == 1:
       return all-zero derivatives, l_grad(i) = 0 for 0 <= i < N.
     e_grad := o_grad(i2) - o_grad(i1)
     d_grad := e_grad / interp_prob
     Set l_grad(i) := d_grad * (2d - 1) * p(i), for 0 <= i < N.
     l_grad(i1) -= d_grad * d       # i1 corresponds to a in (8)
     l_grad(i2) +=  d_grad * (1-d)  # i2 corresponds to b in (8)

     Return l_grad

  MODIFIED BACKWARD ALGORITHM FOR "STRAIGHT-THROUGH" DERIVATIVES

     Here we describe a modified version of the backward algorithm
     for when we want to interpolate the derivatives of our
     algorithm w.r.t. the "straight-through" derivatives where we
     just treat our o_grad as if it was was p_grad, i.e. we imagine
     that the output was just the softmax output.  This will give
     us a biased but lower-variance estimate of the model parameters'
     derivatives, and might be useful early in training.  Let straight_through_scale
     be the scale on the "straight-through"
     derivatives; and we'll give the "real" derivatives, computed
     as described above, a scale of (1-straight_through_scale).
     For reference, the backprop from p back to l, if we had p_grad,
     would be as follows:
            z_grad = - \sum_i  p(i) p_grad(i)
         l_grad(i) = p(i) (p_grad(i)  + z_grad)
     (this follows from (6))

   So the modified backward algorithm is:

     z_grad = - (\sum_i p(i) o_grad(i)   # <-  for straight-through term, will
                                         # be multiplied by straight_through_scale
     if i1 == i2 or e == 0 or e == 1:
        l_grad(i) = straight_through_scale * p(i) * (o_grad(i) + z_grad), for 0 <= i < N
        return
     e_grad := o_grad(i2) - o_grad(i1)
     d_grad := e_grad / interp_prob
     For 0 <= i < N, set:
       l_grad(i) := p(i) * (straight_through_scale * (o_grad(i) + z_grad) +
                           (1-straight_through_scale) * d_grad * (2d - 1))

     l_grad(i1) -= (1-straight_through_scale) * d_grad * d
     l_grad(i2) += (1-straight_through_scale) * d_grad * (1-d)

     Return l_grad
*/



/*
  Return the index i into cumsum, with begin <= i < end,
  such that cumsum[i] <= r < cumsum[i + 1].
  We assume that cumsum[begin] <= r < cumsum[end], and we do not
  access those two elements.
*/
template <typename IterType, typename ScalarType> int find_class(
    IterType cumsum, int begin, int end, ScalarType r) {
  // First search for the element i with  begin <= i < end,
  // such that
  // cumsum[i] <= r < cumsum[i+1]
  int N = cumsum.size(0),
      begin = 0, end = N;
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
  Return the index i into cumsum, with begin <= i < end,
  such that cumsum[i] <= r < cumsum[i + 1].
  We assume that cumsum[begin] <= r < cumsum[end], and we do not
  access those two elements.
*/
template <ScalarType> inline int find_class_in_vec(
    std::vector<ScalarType> &cumsum, int begin, int end, ScalarType r) {
  while (end > begin + 1) {
    int mid = begin + (end - begin) / 2;
    if (cumsum[mid] <= r)
      begin = mid;
    else
      end = mid;
  }
  return begin + 1;
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

 */
template <ScalarType> inline scalar_t discretize_weight(scalar_t weight, int M, scalar_t *r) {
  assert(weight >= 0 && weight <= 1.0);
  // e.g. if M == 128, we divide the range [0..1] into 127 equal intervals, so that we
  // can represent both 0 and 1.
  scalar_t weight_scaled = weight * (M - 1);
  int lower_val = (int) weight_scaled;  // rounds down.
  if (lower_val >= M - 2) {
    // This should only happen if weight == 1.0 (hence weight_scaled == M - 1).
    lower_val = M - 2;
  }
  scalar_t residual = weight_scaled - lower_val;
  asert(residual >= 0.0 && residual <= 1.0);
  if (*r < residual) {
    *r = *r / residual;  // make it a "new" random value by zooming in.
    return (lower_val + 1) / scalar_t(M - 1);
  } else {
    *r = (*r - residual) / (1.0 - residual);
    return lower_val / scalar_t(M - 1);
    assert(*r >= 0.0 && *r <= 1.0); // TODO: remove assertion.
  }
}


float Exp(float f) {
  return expf(f);
}
double Exp(double f) {
  return exp(f);
}


/*
  Forward of iterative_sampling.  See """... """ comment of `iterative_sampling` in
  iterative_sampling.py for documentation of the behavior of this function.

    cumsum: (exclusive) cumulative probabilities of input classes, of shape (B, N) where B is the
        batch size and N is the number of classes, so element cumsum[b][0] would be 0.0,
        cumsum[b][1] = the first probability, cumsum[b][2] = sum of 1st and 2nd probabilities, and so on.
    alpha:  normalization factors, of shape (B), which is the same as the *
            in `probs`.  Numbers satisfying:
                torch.allclose((1 - (-probs * alpha.unsqueeze(-1)).exp()).sum(dim=-1), K),
             i.e. \sum_i (1 - exp(-alpha * probs_i)) == K.
     rand:  random numbers in range [0,1], of shape (B)
   rand:
     K:   number of samples to choose
     M:   number of levels of discretization of the interval [0,1], for
          the class weights at the output

   Return:
       Returns (indexes, weights), where:
       indexes: a Tensor of shape (B, K) and type torch::kInt64, containing,
               for each B, a series of K distinct sampled integers in the
               range [0,N-1].
       weights: a Tensor of shape (B, K) of type the same as probs (e.g. float),
               containing the discretized weights corresponding to each index,
               converted back to float values in the range [0,1].  These will
               in general be larger than the corresponding weights in `probs`,
               because we divide by the probability of choosing each class,
               in order to preserve expectations.
*/
std::vector<torch::Tensor> iterative_sampling_cpu(torch::Tensor probs,
                                                  torch::Tensor alpha,
                                                  torch::Tensor rand,
                                                  int K,
                                                  int M) {
  TORCH_CHECK(probs.dim() == 2, "cumsum must be 2-dimensional");
  int B = probs.size(0),
      N = probs.size(1);

  TORCH_CHECK(alpha.dim() == 1, "alpha must be 1-dimensional");
  TORCH_CHECK(rand.dim() == 1, "rand must be 1-dimensional");
  TORCH_CHECK(K > 0 && K < N);
  TORCH_CHECK(M > 1);
  TORCH_CHECK(alpha.size(0) == B);
  TORCH_CHECK(rand.size(0) == B);


  TORCH_CHECK(probs.device().is_cpu() && alpha.device().is_cpu() &&
              rand.device().is_cpu(),
              "inputs must be CPU tensors");

  auto scalar_type = probs.scalar_type();

  auto opts = torch::TensorOptions().dtype(scalar_type).device(probs.device()),
      long_opts = torch::TensorOptions().dtype(torch::kInt64).device(probs.device());


  torch::Tensor indexes = torch::zeros({B, N}, long_opts),
      weights = torch::zeros({B, N}, opts);


  AT_DISPATCH_FLOATING_TYPES(scalar_type, "iterative_sampling_cpu_loop", ([&] {
        auto cumsum_a = cumsum.packed_accessor32<scalar_t, 2>(),
            weights_a = weights.packed_accessor32<scalar_t, 2>();
        auto alpha_a = alpha.packed_accessor32<scalar_t, 1>(),
            rand_a = rand.packed_accessor32<scalar_t, 1>();
        auto indexes_a = indexes.packed_accessor32<int32_t, 2>();


        // at iteration k, prev_classes [0] contains -1;
        // prev_classes[1,2,..k] contains the previously chosen k classes (all
        // distinct), in sorted order; and prev_classes[k+1] contains N.
        std::vector<int32_t> prev_classes(K + 1);

        // at iteration k, prev_cumsum[c] for 0 <= c <= k contain the cumulative
        // probabilities after subtracting the probability mass due to the
        // previously chosen classes (sorted by class-index, not iteration).
        //
        // At iteration k we have already chosen k classes.  This
        // breaks up the remaining probability mass (excluding those k classes)
        // into k+1 intervals.  Counting the values accessible in "cumsum",
        // interval i (with 0 <= i <= k) starts at cumsum[b][prev_classes[i]+1],
        // and ends at cumsum[b][prev_classes[i+1]].
        //
        // In cur_cumsum, we store the cumulative sum *excluding* the intervals
        // corresponding to the previously chosen classes.  On iteration k,
        // cur_cumsum[0] will always contain 0 and cur_cumsum[i] for 0 < i <= k
        // will contain the start of interval i, with the intervals belonging to
        // previously chosen classes subtracted.
        std::vector<scalar_t> cur_cumsum(K);

        for (int b = 0; b < B; b++) {
          prev_classes[0] = -1;
          prev_classes[1] = N;
          cur_cumsum[0] = 0.0;

          // at iteration k, chosen_sum contains the sum of the probabilities of
          // classes chosen on iters 0,1,..k-1.
          scalar_t chosen_sum = 0.0;

          auto this_cumsum_a = cumsum_a[b];
          auto this_indexes_a = indexes_a[b];
          auto this_weights_a = weights_a[b];

          scalar_t r = rand_a[b],  // r is a new random value on [0..1].
              alpha = alpha_a[b];   // alpha is the constant that will determine
                                    // the scaling-up of probabilities, to
                                    // preserve expectations.

          for (int k = 0; k < K; ++k) {
            // i will be some value 0 <= i <= k.
            int i = find_class_in_vec(cur_cumsum, 0, k + 1, r);

            // class_range
            int class_range_begin = prev_classes[i] + 1,
                class_range_end = prev_classes[i + 1];
            // shift r...
            r = r - cur_cumsum[i] + this_cumsum_a[class_range_begin];

            int c = find_class(this_cumsum_a,
                               class_range_begin,
                               class_range_end, r);
            // c is the class chosen.
            this_indexes_a[k] = c;

            scalar_t this_class_prob = this_cum_sum_a[c + 1] - this_cum_sum_a[c];
            // We can now treat r as a "new" random value on [0,1].
            r = (r - this_cum_sum_a[c]) / this_class_prob;


            scalar_t prob_of_choosing_class = 1.0 - exp(-alpha * this_class_prob);
            // We assign the class a weight of "class_weight" in the output,
            // which is larger than its actual probability, so the output can have the same expected
            // value as the input.
            scalar_t class_weight = this_class_prob / prob_of_choosing_class,
                discretized_class_weight = discretize_weight(class_weight, M, &r);

            this_weights_a[k] = discretized_class_weight;

            // could unroll this loop.  mostly an issue if K is large.
            for (int k2 = k; k2 > i + 1; --k2) {
              cur_cumsum[k2] = cur_cumsum[k2 - 1] - this_class_prob;
            }
            cur_cumsum[i + 1] = cur_cumsum[i] + (this_cumsum_a[c] -
                                                 this_cumsum_a[class_range_begin]);


            chosen_sum += this_class_prob;
            // Reduce the random value r so that it is in the range [0..1.0-chosen_sum],
            // which is the size of the reduced interval after subtracting the probability
            // mass due to previously chosen classes.
            r = r * (1.0 - chosen_sum);
          }









          scalar_t chosen_cumsum



          auto this_cumsum_a = cumsum_a[b];
          scalar_t d = rand_a[b][2],
              e = (d < lower_bound ? 0.0 :
                   (d > upper_bound ? 1.0 :
                    ((d - lower_bound) / interp_prob)));
          int index1 = -1,
              index2 = -1;
          // the if-statements here are optimizations, we could
          // always compute both index1 and index2 if we wanted.
          if (d < upper_bound)
            index1 = find_class(this_cumsum_a, rand_a[b][0]);
          if (d > lower_bound)
            index2 = find_class(this_cumsum_a, rand_a[b][1]);
          if (index2 < 0)
            index2 = index1;
          if (index1 < 0)
            index1 = index2;
          ans_indexes_a[b][0] = index1;
          ans_indexes_a[b][1] = index2;
          if (index1 == index2 || e == 1 || e == 0) {
            // This 'really' corresponds to index1 if d < lower_bound,
            // and index2 if d > upper_bound.  Also we can reach here
            // because we happen to have sampled the same value twice.
            ans_a[b][index1] = 1.0;
          } else {
            // Interpolate between two values.
            scalar_t e = (d - lower_bound) / interp_prob;  // 0 <= e <= 1.0
            ans_a[b][index1] = 1.0 - e;
            ans_a[b][index2] = e;
          }
        }
      }));
  return std::vector<torch::Tensor>({ans, ans_indexes});
}


/*
   backward of iterative_sampling.  Returns logits_grad; note, `logits` is the
   original log-probs from which `cumsum` was derived by softmax and then
   cumulative summation.
   We don't return a grad for `rand`; actually, such a thing is not even
   possible as the the forward function is a discontinuous function of the
   input.
   These derivatives are not correct from the point of view of a
   deterministic function of the original `logits` and `rand`.  They are
   only correct in a sense involving expectations.
*/
torch::Tensor iterative_sampling_backward_cpu(
    torch::Tensor cumsum,
    torch::Tensor rand,
    torch::Tensor ans_indexes,
    torch::Tensor ans_grad,
    float interp_prob,
    float straight_through_scale) {
  TORCH_CHECK(cumsum.dim() == 2, "cumsum must be 2-dimensional");
  TORCH_CHECK(rand.dim() == 2, "rand must be 2-dimensional");
  TORCH_CHECK(ans_indexes.dim() == 2, "ans_indexes must be 2-dimensional");
  TORCH_CHECK(ans_grad.dim() == 2, "ans_grad must be 2-dimensional");
  TORCH_CHECK(interp_prob > 0.0 && interp_prob <= 1.0);
  TORCH_CHECK(straight_through_scale >= 0.0 && straight_through_scale <= 1.0);


  auto scalar_type = cumsum.scalar_type();
  auto opts = torch::TensorOptions().dtype(scalar_type).device(cumsum.device());

  const int B = cumsum.size(0),
      N = cumsum.size(1);

  TORCH_CHECK(rand.size(0) == B && rand.size(1) == 3);
  TORCH_CHECK(ans_indexes.size(0) == B && ans_indexes.size(1) == 2);
  TORCH_CHECK(ans_grad.size(0) == B && ans_grad.size(1) == N);

  TORCH_CHECK(cumsum.device().is_cpu() && rand.device().is_cpu() &&
              ans_indexes.device().is_cpu() && ans_grad.device().is_cpu());


  // We compute the derivative w.r.t. the original logits (meaning:
  // un-normalized logprobs), even though the input to the original function
  // was a processed form of the logits: the cumulative distribution of
  // the class probabilities, derived from the logits.
  torch::Tensor logits_grad =
      (straight_through_scale == 0.0 ?
       torch::zeros({B, N}, opts) :
       torch::empty({B, N}, opts));

  AT_DISPATCH_FLOATING_TYPES(cumsum.scalar_type(), "iterative_sampling_cpu_backward_loop", ([&] {
        auto cumsum_a = cumsum.packed_accessor32<scalar_t, 2>(),
            rand_a = rand.packed_accessor32<scalar_t, 2>(),
            ans_grad_a = ans_grad.packed_accessor32<scalar_t, 2>(),
            logits_grad_a = logits_grad.packed_accessor32<scalar_t, 2>();
        auto ans_indexes_a = ans_indexes.packed_accessor32<int32_t, 2>();
        scalar_t inv_interp_prob = 1.0 / interp_prob;

        // Search above for the part of the comment headed:
        // MODIFIED BACKWARD ALGORITHM FOR "STRAIGHT-THROUGH" DERIVATIVES
        // to see what we are implementing.
        float lower_bound = (1.0 - interp_prob) * 0.5,
            upper_bound = (1.0 + interp_prob) * 0.5;
        for (int b = 0; b < B; b++) {
          auto this_cumsum_a = cumsum_a[b],
              this_ans_grad_a = ans_grad_a[b],
              this_logits_grad_a = logits_grad_a[b];

          scalar_t z_grad = 0.0;
          if (straight_through_scale != 0.0) {
            scalar_t cur_cum_prob = 0.0;
            for (int i = 0; i < N; i++) {
              scalar_t cum_prob = this_cumsum_a[i],
                  prob = cum_prob - cur_cum_prob,
                  o_grad = this_ans_grad_a[i];
              // Implements: z_grad = - (\sum_i p(i) o_grad(i)
              z_grad -= prob * o_grad;
              cur_cum_prob = cum_prob;
            }
          }
          int32_t i1 = ans_indexes_a[b][0],
              i2 = ans_indexes_a[b][1];
          scalar_t d = rand_a[b][2],
              e = (d < lower_bound ? 0.0 :
                   (d > upper_bound ? 1.0 :
                    ((d - lower_bound) / interp_prob)));
          if (i1 == i2 || e == 0.0 || e == 1.0) {
            if (straight_through_scale == 0.0)
              continue;  // Leave this row of logits_grad at zero
            scalar_t cur_cum_prob = 0.0;
            for (int i = 0; i < N; i++) {
              scalar_t cum_prob = this_cumsum_a[i],
                  prob = cum_prob - cur_cum_prob,
                  o_grad = this_ans_grad_a[i];
              // Implements:
              // l_grad(i) = straight_through_scale * p(i) * (o_grad(i) + z_grad)
              scalar_t l_grad_i = straight_through_scale * prob * (o_grad + z_grad);
              this_logits_grad_a[i] = l_grad_i;
              cur_cum_prob = cum_prob;
            }
          } else {
            scalar_t e_grad = this_ans_grad_a[i2] - this_ans_grad_a[i1],
                d_grad = e_grad * inv_interp_prob,
                d2m1 = (2.0 * d - 1.0);
            if (straight_through_scale == 0.0) {
              scalar_t cur_cum_prob = 0.0;
              for (int i = 0; i < N; i++) {
                scalar_t cum_prob = this_cumsum_a[i],
                    prob = cum_prob - cur_cum_prob;
                // Implements (from "BACKWARD ALGORITHM"):
                //  l_grad(i) := d_grad * (2d - 1) * p(i)
                scalar_t l_grad_i = d_grad * d2m1 * prob;
                this_logits_grad_a[i] = l_grad_i;
                cur_cum_prob = cum_prob;
              }
            } else {
              scalar_t cur_cum_prob = 0.0;
              for (int i = 0; i < N; i++) {
                scalar_t cum_prob = this_cumsum_a[i],
                    prob = cum_prob - cur_cum_prob,
                    o_grad = this_ans_grad_a[i];
                // Implements:
                // l_grad(i) := p(i) * (straight_through_scale * (o_grad(i) + z_grad) +
                //                     (1-straight_through_scale) * d_grad * (2d - 1))
                scalar_t l_grad_i = prob * (straight_through_scale * (o_grad + z_grad) +
                                            d_grad * (d2m1 * (1.0 - straight_through_scale)));
                this_logits_grad_a[i] = l_grad_i;
                cur_cum_prob = cum_prob;
              }
            }
            // Implements:
            // l_grad(i1) -= (1-straight_through_scale) * d_grad * d;
            // l_grad(i2) += (1-straight_through_scale) * d_grad * (1-d)
            this_logits_grad_a[i1] -= (1.0 - straight_through_scale) * d_grad * d;
            this_logits_grad_a[i2] += (1.0 - straight_through_scale) * d_grad * (1.0 - d);
          }
        }
      }));
  return logits_grad;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iterative_sampling_cpu", &iterative_sampling_cpu, "Flow sampling forward function (CPU)");
  m.def("iterative_sampling_backward_cpu", &iterative_sampling_backward_cpu, "Flow sampling backward function (CPU)");
}
