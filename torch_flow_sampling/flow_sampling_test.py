# Caution: this will fail occasionally due to cutoffs not being quite large enough.
# As long as it passes most of the time, it's OK.

import random
import torch
from torch_flow_sampling import flow_sample
from typing import List


def test_flow_sampling_basic():
    print("Running test_flow_sampling_basic()")

    B = 30
    N = 32
    logits = torch.randn(B, N)
    loss_grad = torch.randn(B, N)
    interp_prob = 0.5


    logits.requires_grad = True
    sampled = flow_sample(logits, interp_prob)
    assert torch.allclose(sampled.sum(dim=1), torch.tensor([1.0]))


    loss = (sampled * loss_grad).sum()
    loss.backward()

    proportion_interp = ((sampled != 0).sum() / B) - 1.0
    print(f"interp_prob={interp_prob}, proportion_interp={proportion_interp} (should usually be < interp_prob)")


    print("Sampled = ", sampled)
    print("logits_grad = ", logits.grad, ", sum = ", logits.grad.sum())

    return


def test_list_is_zero_mean(l: List[torch.Tensor]) -> None:
    """
    Tests that a provided list of Tensors has zero mean.  The Tensors must all have
    the same shape, and we are testing that the elements in each position are
    zero-mean i.i.d.  The different elements don't necessarily have to have the
    same variance, and there can be correlations between different elements of
    each Tensor.

    Will crash if the statistics don't seem to have their expected values,
    within a reasonable tolerance.  Don't call this function with extremely
    short lists, e.g. less than 10 elements.
    """
    assert len(l) >= 10
    y = torch.stack([ x.reshape(-1) for x in l ], dim=0)
    (N, D) = y.shape  # N is the number of elements in the list, D is the number of elements
                      # in each original  tensor
    y_var  = (y ** 2).mean(dim=0) + 1.0e-20  # shape = (D,).  variance of elements
    y_meansq = y.mean(dim=0) ** 2  # shape = (D,), squared mean.
    y_meansq_div_var = y_meansq / y_var  # shape = (D,), mean^2 / var.
                                         # Average value should be about 1/N
    avg_meansq = y_meansq_div_var.mean().item()
    ratio = (avg_meansq / (1/(N-1)))
    print(f"avg_meansq_div_var = {avg_meansq}, vs. {1/(N-1)}, ratio = {ratio}")
    assert ratio > 0.75 and ratio < 1.25

# Here is an example of how we tested test_list_is_zero_mean() from the python
# command line, after pasting the above code into the command line:
# >>> test_list_is_zero_mean([torch.randn(100,200) for _ in range(100)])
# >>> avg_meansq_div_var = 0.010024802759289742, vs. 0.010101010101010102, ratio = 0.9924554731696843


def test_flow_sampling_linear1():
    print("Running test_flow_sampling_linear1()")

    # Testing that the expected value of our sampling operation is the
    # same as the softmax.

    # Caution: we need to have logits with a relatively low variance, and
    # a fairly small number of classes, and a large number of elements in
    # the list below (for _ in range(...)) for the statistical test to
    # work out.  It can fail if some classes are so rarely sampled
    # that we never see a sample of the class in the entire list-- or
    # at least, if this happens for enough classes, and frequently enough,
    # that it shows up in the globally averaged stats.
    B = 1
    N = 32
    logits = 0.5 * torch.randn(B, N)
    interp_prob = 0.1

    expectation = logits.softmax(dim=1)

    avg_sampled = torch.stack([ flow_sample(logits, interp_prob) for _ in range(300) ], dim=0).mean(dim=0)

    print("Expectation = ", expectation)
    print("vs. Avg_sampled = ", avg_sampled)



def test_flow_sampling_linear():
    print("Running test_flow_sampling_linear()")
    # Testing that the expected value of our sampling operation is the
    # same as the softmax.

    # Caution: we need to have logits with a relatively low variance, and
    # a fairly small number of classes, and a large number of elements in
    # the list below (for _ in range(...)) for the statistical test to
    # work out.  It can fail if some classes are so rarely sampled
    # that we never see a sample of the class in the entire list-- or
    # at least, if this happens for enough classes, and frequently enough,
    # that it shows up in the globally averaged stats.
    B = 30
    N = 32
    logits = 0.5 * torch.randn(B, N)
    interp_prob = 0.5

    expectation = logits.softmax(dim=1)

    # We had to make the list length here quite long (1000) to get the test to
    # (usually) pass.  This has to do with rarely sampled classes.
    sampled = [ flow_sample(logits, interp_prob) - expectation for _ in range(1000) ]

    test_list_is_zero_mean(sampled)



def test_flow_sampling_linear_deriv():
    print("Running test_flow_sampling_linear_deriv()")

    # Tests that for a linear loss funtion, the derivative given by our code is
    # the same, in expectation, as the derivative given by our code.


    # Caution: we need to have logits with a relatively low variance, and
    # a fairly small number of classes, and a large number of elements in
    # the list below (for _ in range(...)) for the statistical test to
    # work out.  It can fail if some classes are so rarely sampled
    # that we never see a sample of the class in the entire list-- or
    # at least, if this happens for enough classes, and frequently enough,
    # that it shows up in the globally averaged stats.
    B = 30
    N = 32
    logits = 0.5 * torch.randn(B, N)
    interp_prob = 0.5

    loss_deriv = torch.randn(B, N)

    logits.requires_grad = True
    expectation = logits.softmax(dim=1)

    loss = (expectation * loss_deriv).sum()
    loss.backward()

    exact_grad = logits.grad.detach()
    logits.grad = None


    if True:
        # test with straight_through_scale=1.0
        loss = (flow_sample(logits, interp_prob, straight_through_scale=1.0) * loss_deriv).sum()
        loss.backward()
        sampled_grad = logits.grad.detach()
        assert torch.allclose(sampled_grad, exact_grad)


    for straight_through_scale in (0.0, 0.5):
        sampled_grads = []
        for i in range(2000):
            loss = (flow_sample(logits, interp_prob, straight_through_scale=straight_through_scale) * loss_deriv).sum()
            loss.backward()
            sampled_grad = logits.grad.detach()
            sampled_grads.append(sampled_grad)
            logits.grad = None

            if i < 5:
                print(f"straight_through_scale={straight_through_scale}, sampled_grads norm = {torch.linalg.norm(sampled_grad)}, exact norm = {torch.linalg.norm(exact_grad)}")


        diff_grads = [ sampled_grad - exact_grad for sampled_grad in sampled_grads ]

        print(f"Testing same-mean property for sampled_grads vs exact_grad with straight_through_scale={straight_through_scale}")
        test_list_is_zero_mean(diff_grads)





if __name__ == "__main__":
    torch.set_printoptions(edgeitems=30)
    test_flow_sampling_basic()
    test_flow_sampling_linear1()
    test_flow_sampling_linear()
    test_flow_sampling_linear_deriv()
    print("Done")
