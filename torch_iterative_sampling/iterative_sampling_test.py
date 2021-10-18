import math
import random
import torch
from torch import nn
from torch_iterative_sampling import SamplingBottleneckModule
from typing import List


# below is a small part of the output of this program:
# seq_len=8, minibatch=19500, loss=0.878 vs. ref_loss=0.713, ref_loss_shannon=0.736 class_entropy=6.235, frame_entropy=3.903
# seq_len=4, minibatch=19500, loss=0.918 vs. ref_loss=0.848, ref_loss_shannon=0.844 class_entropy=6.232, frame_entropy=3.162
# seq_len=2, minibatch=19500, loss=0.948 vs. ref_loss=0.924, ref_loss_shannon=0.912 class_entropy=6.227, frame_entropy=2.371l
# seq_len=1, minibatch=19500, loss=0.965 vs. ref_loss=0.965, ref_loss_shannon=0.952 class_entropy=6.205, frame_entropy=0.671


def test_iterative_sampling_train():
    for seq_len in 16, 8, 4, 2, 1:
        print(f"Running test_iterative_sampling_train: seq_len={seq_len}")
        device = torch.device('cuda')
        dim = 256
        hidden_dim = 512
        num_classes = 512
        num_discretization_levels = 512
        m = SamplingBottleneckModule(dim, num_classes,
                                     seq_len=seq_len,
                                     num_discretization_levels=num_discretization_levels,
                                     random_rate=1.0,
                                     epsilon=0.1).to('cuda')

        m_rest = nn.Sequential(nn.Linear(dim, hidden_dim),
                               nn.ReLU(hidden_dim),
                               nn.LayerNorm(hidden_dim),
                               nn.Linear(hidden_dim, dim)).to('cuda')

        m_tot = nn.ModuleList((m, m_rest))

        m_tot.train()
        optim = torch.optim.Adam(m_tot.parameters(), lr=0.005, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

        # only the last dimension here ('dim') needs to match anything.
        feats_shape = (200, 30, dim)

        class_entropy_scale = 0.1  # affects how much we penalize -class_entropy in training.
        frame_entropy_scale = 0.1  # affects how much we penalize -frame_entropy
                                   # in training; but this is clamped so when it
                                   # exceeds a certain value we no longer
                                   # include it in the loss.

        # We figure out a reference loss, ref_loss, that reflects what the loss
        # would be if we were efficiently using the information passed through
        # the channel.  The information consists of the set of class indexes
        # and associated weights.  Since we think the info in the weights will
        # be inefficiently  used, for now we just consider the class indexes.
        # The number of possible combinations of class indexes is:

        #    P = num_classes ** seq_len / math.factorial(seq_len)   (1)

        # Suppose (we're ignoring the values/weights), that the number of
        # distinct outputs the network can have is equal to P from eqn. (1).
        # We can approximately figure out what the loss function would be,
        # based on an analysis of randomized VQ, initialized via one iteration
        # of k-means clustering.
        # Let the dimension of the space be D (this correspond to "dim" above).
        #
        #   On iteration 0 we assign to each class p=1..P-1, a vector v_p^0,
        #  which will have all elements being 1 or -1, chosen randomly.
        #  We then generate a large number of points a_i, with i=0..I-1,
        # from a zero-mean, unit-variance
        # Gaussian in dimension D.  (It's important to choose this distribution,
        # as it's the distribution of our `feats`).  We assign these points to
        # the cluster centers by choosing the v_p^0 (over p) that has the largest
        # dot product with the point; this gives the shortest distance to any v_p^0
        # since it minimizes  (a_i - v_p^0)**2 = a_i.a_i - 2 a_i.v_p^0 + v_p^0.v_p^0,
        # the other 2 terms being constant for a given a_i.  These dot products
        # have distribution N(0, D), i.e. the variance is D.  They are *close*
        # to independently distributed, closer as D gets larger, we'll treat them
        # as independent.
        #
        # According to
        # https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables
        # (see also https://en.wikipedia.org/wiki/Fisher%E2%80%93Tippett%E2%80%93Gnedenko_theorem#Gumbel_distribution)
        ##
        #the mean of the maximum of a size n normal sample, for large n is well approximated by:
        #
        # sqrt(log(n*n / (2*pi*log(n*n/(2*pi)))))  * (1 + gamma/log(n) + o(1/log(n)))   (2)
        # where gamma = Euler-Mascheroni constant = 0.5772156649
        #  .. and ignoring the o(1/log(n)) part, as "small", we treat this as
        #  sqrt(log(n*n / (2*pi*log(n*n/(2*pi)))))  * (1 + gamma/log(n))
        #
        # So we can use the above expression times sqrt(D) [the stddev] as the mean value of
        # the dot products, i.e.:
        #  mean dot_prod, i.e.
        #
        #   E[a_i.v_p^0] \simeq sqrt(D) *  sqrt(log(P*P / (2*pi*log(P*P/(2*pi)))))  * (1 + gamma/log(P))
        #
        # We can use the above expression to lower-bound the expected distance of the mean of the
        # points assigned to any particular cluster p, from the origin.  We can do this because
        # we know (approximately) the expected value in the direction of v_p.  [there will also be
        # some component orthogonal to v_p in generall; this will probably be considerably less though.]
        # The unit vector in the direction of v_p equals v_p / sqrt(D) [since v_p.v_p==D],
        # so we can lower-bound the expected average distance from the origin to the cluster
        # centers on iteration 1 of k-means, as:
        #
        #   sqrt(log(P*P / (2*pi*log(P*P/(2*pi)))))  * (1 + gamma/log(P)).
        #
        # If we use a suboptimal assignments of points to these new cluster centers, by
        # just taking the assignment from iteration 0, we'll be able to show that the remaining
        # variance, i.e. E[point-cluster_center]^2, will equal:
        #
        #  remaining_var = [ D -  log(P*P / (2*pi*log(P*P/(2*pi))))  * (1 + gamma/log(P))**2 ],
        #
        # and normalizing this by dividing by the original total variance D, to match our
        # loss function, we have
        #
        # expected_loss = [ D -  log(P*P / (2*pi*log(P*P/(2*pi))))  * (1 + gamma/log(P))**2 ] / D
        #
        # (where D == dim)

        gamma = 0.5772156649 # Euler-Mascheroni constant
        P = num_classes ** seq_len / math.factorial(seq_len)
        ref_loss = (dim - (math.log(P*P / (2*math.pi*math.log(P*P/(2*math.pi))))  * (1 + gamma/math.log(P)))) / dim


        # Shannon's rate-distortion equation says rate = 1/2 log_2(sigma_x^2 / D),
        # where sigma_x^2 is the input variance and D is the distortion.  Our
        # `loss` can be interpreted as D / sigma_x^2, so we have:
        #   1/loss = exp_2(2*rate) = 2 ** (2 * rate)
        # .. here, the rate is the rate per dimension..
        #  so, loss = 0.5 ** (2 * bits_per_dim)
        #   loss = 0.5 ** (2 * (log(P)/log(2))/dim)
        ref_loss_shannon = 0.5 ** (2 * (math.log(P) / math.log(2))/dim)

        # You might notice that with P large enough, the above expression could potentially be
        # negative, which makes no sense.  The flaw in the argument here arises once P gets
        # close to 2**D, because in that case a significant proportion of the initial clusters
        # v_p^0 start being identical, and the draws are no longer independent.  Even if
        # the v_p^0 were all distinct, e.g. with Gaussian distribution, the "independent" part
        # still fails because there are correlations arising from multiplication by the
        # known point a_i.

        for i in range(20000):

            feats = torch.randn(*feats_shape, device=device)

            output, _, _, _, class_entropy, frame_entropy = m(feats)
            #output = m_rest(output)

            # try to reconstruct the feats, after this information bottleneck.
            loss = ((feats - output) ** 2).sum() / feats.numel()
            if i % 500 == 0:
                loss_val = loss.to('cpu').item()
                print(f"seq_len={seq_len}, minibatch={i}, loss={loss_val:.3f} vs. ref_loss={ref_loss:.3f}, ref_loss_shannon={ref_loss_shannon:.3f} "
                      f"class_entropy={class_entropy.to('cpu').item():.3f}, "
                      f"frame_entropy={frame_entropy.to('cpu').item():.3f}")

            # stop maximizing frame_entropy when it is greater than seq_len.
            frame_entropy = torch.clamp(frame_entropy, max=(math.log(seq_len*2)))

            (loss  - class_entropy_scale * class_entropy - frame_entropy_scale * frame_entropy).backward()
            optim.step()
            optim.zero_grad()
            if i % 1000 == 0:
                scheduler.step()




if __name__ == "__main__":
    torch.set_printoptions(edgeitems=30)
    # Caution!  This is very slow, can take half an hour.
    # Some of the statistical tests require a lot of samples
    test_iterative_sampling_train()
    print("Done")
