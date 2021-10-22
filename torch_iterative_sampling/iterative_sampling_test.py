import math
import random
import torch
from torch import nn
from torch_iterative_sampling import SamplingBottleneckModule, BottleneckPredictor
from typing import List


# below is a small part of the output of this program:
# seq_len=16, minibatch=19500, loss=0.818 vs. ref_loss=0.482, ref_loss_shannon=0.583 class_entropy=6.237, frame_entropy=4.605
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
        num_discretization_levels = 256

        # with epsilon=0.001, the current value, selected output:
        # seq_len=16, minibatch=19500, reconstruction_loss=0.708/valid=0.636 vs. ref_loss=0.583 class_entropy=6.223, frame_entropy=1.729 (limit: 1.386)
        # Average sumsq of feats.grad elements is  tensor(0.7301, device='cuda:0')
        # seq_len=8, minibatch=19500, reconstruction_loss=0.820/valid=0.771 vs. ref_loss=0.736 class_entropy=6.221, frame_entropy=1.551 (limit: 1.386)
        # Average sumsq of feats.grad elements is  tensor(0.5470, device='cuda:0')
        # seq_len=4, minibatch=19500, reconstruction_loss=0.896/valid=0.865 vs. ref_loss=0.844 class_entropy=6.218, frame_entropy=1.323 (limit: 1.040)
        # Average sumsq of feats.grad elements is  tensor(0.3942, device='cuda:0')
        # seq_len=2, minibatch=19500, reconstruction_loss=0.940/valid=0.920 vs. ref_loss=0.912 class_entropy=6.215, frame_entropy=1.016 (limit: 0.693)
        # Average sumsq of feats.grad elements is  tensor(0.3377, device='cuda:0')
        # seq_len=1, minibatch=19500, reconstruction_loss=0.965/valid=0.959 vs. ref_loss=0.952 class_entropy=6.203, frame_entropy=0.352 (limit: 0.347)
        # Average sumsq of feats.grad elements is  tensor(58.5082, device='cuda:0')  <-- this is odd, maybe random.

        # with epsilon=0.0001:
        # ... loss function is a little better, but grad magnitude is maybe 20% larger.  May not be worth it.
        # seq_len=16, minibatch=19500, reconstruction_loss=0.705/valid=0.629 vs. ref_loss=0.583 class_entropy=6.219, frame_entropy=1.521 (limit: 1.386)
        # Average sumsq of feats.grad elements is  tensor(0.9230, device='cuda:0')
        # seq_len=8, minibatch=19500, reconstruction_loss=0.819/valid=0.762 vs. ref_loss=0.736 class_entropy=6.219, frame_entropy=1.434 (limit: 1.386)
        # Average sumsq of feats.grad elements is  tensor(0.6528, device='cuda:0')
        # seq_len=4, minibatch=19500, reconstruction_loss=0.894/valid=0.861 vs. ref_loss=0.844 class_entropy=6.212, frame_entropy=1.158 (limit: 1.040)
        # Average sumsq of feats.grad elements is  tensor(0.5022, device='cuda:0')
        # seq_len=2, minibatch=19500, reconstruction_loss=0.937/valid=0.916 vs. ref_loss=0.912 class_entropy=6.212, frame_entropy=0.966 (limit: 0.693)
        # Average sumsq of feats.grad elements is  tensor(0.3827, device='cuda:0')
        # seq_len=1, minibatch=19500, reconstruction_loss=0.963/valid=0.956 vs. ref_loss=0.952 class_entropy=6.195, frame_entropy=0.346 (limit: 0.347)
        # Average sumsq of feats.grad elements is  tensor(15.7329, device='cuda:0')


        m = SamplingBottleneckModule(dim, num_classes,
                                     seq_len=seq_len,
                                     num_discretization_levels=num_discretization_levels,
                                     random_rate=1.0,
                                     epsilon=0.001).to('cuda')

        predictor_dim = 256
        p = BottleneckPredictor(num_classes, predictor_dim,
                                num_discretization_levels,
                                seq_len, hidden_dim,
                                num_hidden_layers=1).to('cuda')
        test_predictor = False  # can set to False for speed

        m_tot = nn.ModuleList((m, p))

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

        P = num_classes ** seq_len / math.factorial(seq_len)

        # Shannon's rate-distortion equation says rate = 1/2 log_2(sigma_x^2 / D),
        # where sigma_x^2 is the input variance and D is the distortion.  Our
        # `loss` can be interpreted as D / sigma_x^2, so we have:
        #   1/loss = exp_2(2*rate) = 2 ** (2 * rate)
        # .. here, the rate is the rate per dimension..
        #  so, loss = 0.5 ** (2 * bits_per_dim)
        #   loss = 0.5 ** (2 * (log(P)/log(2))/dim)
        ref_loss = 0.5 ** (2 * (math.log(P) / math.log(2)) / dim)

        # You might notice that with P large enough, the above expression could potentially be
        # negative, which makes no sense.  The flaw in the argument here arises once P gets
        # close to 2**D, because in that case a significant proportion of the initial clusters
        # v_p^0 start being identical, and the draws are no longer independent.  Even if
        # the v_p^0 were all distinct, e.g. with Gaussian distribution, the "independent" part
        # still fails because there are correlations arising from multiplication by the
        # known point a_i.

        for i in range(20000):

            feats = torch.randn(*feats_shape, device=device)

            num_seqs = 2
            output, probs, class_indexes, value_indexes, class_entropy, frame_entropy = m(feats, num_seqs=num_seqs)

            # the base_predictor is zero because all frames are independent in
            # this test, there is nothing can meaningfully use to predict them.
            #base_predictor = torch.zeros(*feats.shape[:-1], predictor_dim, device=device)
            base_predictor = feats.detach()

            if test_predictor:
                # class_logprobs, value_logprobs have shape equal to feats.shape[:-1]
                class_logprobs, value_logprobs = p(probs, class_indexes, value_indexes, base_predictor)
                assert class_logprobs.shape == feats.shape[:-1]
                class_avg_logprob = class_logprobs.mean()
                value_avg_logprob = value_logprobs.mean()

            # try to reconstruct the feats, after this information bottleneck.
            loss = ((feats - output) ** 2).sum() / feats.numel()

            # Our limit on the frame entropy is one that's below what the model will choose in practice.
            frame_entropy_limit = 0.5 * math.log(min(seq_len, 8)*1.2)

            if i % 500 == 0:
                loss_val = loss.to('cpu').item()

                # Also compute validation loss by putting model in eval mode.  The data is not real vs. fake, it's
                # all random; the difference is using expectations vs. random samples.
                m.eval()
                output_valid = m(feats, num_seqs=num_seqs)[0]
                loss_valid = (((feats - output_valid) ** 2).sum() / feats.numel()).item()
                m.train()

                print(f"seq_len={seq_len}, minibatch={i}, reconstruction_loss={loss_val:.3f}/valid={loss_valid:.3f} vs. ref_loss={ref_loss:.3f} "
                      f"class_entropy={class_entropy.to('cpu').item():.3f}, "
                      f"frame_entropy={frame_entropy.to('cpu').item():.3f} (limit: {frame_entropy_limit:.3f})")
                if test_predictor:
                    print(f"class_avg_logprob={class_avg_logprob.item()}, value_avg_logprob={value_avg_logprob.item()}")

            if test_predictor:
                loss += -(class_avg_logprob + value_avg_logprob)

            # stop maximizing frame_entropy when it is greater than seq_len.
            frame_entropy = torch.clamp(frame_entropy, max=frame_entropy_limit)

            (loss  - class_entropy_scale * class_entropy - frame_entropy_scale * frame_entropy).backward()
            optim.step()
            optim.zero_grad()
            if i % 1000 == 0:
                scheduler.step()


        feats = torch.randn(*feats_shape, device=device)
        feats.requires_grad = True
        output, _, _, _, class_entropy, frame_entropy = m(feats)
        print("Average sumsq of output is ", (output ** 2).mean().item())

        output_grad = torch.randn_like(output)
        (output * output_grad).sum().backward()
        print("Average sumsq of feats.grad elements is ", (feats.grad ** 2).mean().item())
        # to_values_scale is, after training, always in the interval [0.129..0.132], except for seq_len=1
        # when it is in [0.5..0.7]
        print("to_values_scale is ", m.to_values_scale.item())




if __name__ == "__main__":
    torch.set_printoptions(edgeitems=30)
    torch.set_num_interop_threads(1)
    # Caution!  This is very slow, can take half an hour.
    # Some of the statistical tests require a lot of samples
    test_iterative_sampling_train()
    print("Done")
