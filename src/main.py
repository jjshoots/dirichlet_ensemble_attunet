import torch
from dirichlet_ensemble_attunet import DirichletEnsembleAttUNet

if __name__ == "__main__":
    model = DirichletEnsembleAttUNet(
        in_channels=3,
        out_channels=3,
        inner_channels=[2, 4, 6, 8],
        att_num_heads=2,
        num_ensemble=3,
        residual=True,
    )

    # the input is [B, C, H, W]
    # the target is similar, but boolean
    input = torch.randn(1, 3, 512, 512)
    target = torch.randn(1, 3, 512, 512) > 0.0

    # the output is [pos_neg, num_ensemble, B, C, H, W] in [0, +inf]
    output = model(input)
    print(output.shape)
    print(output.min())
    print(output.max())

    # we can binarize the output to obtain a boolean map
    # the result is [B, C, H, W] in {False, True}
    binary_prediction = model.binarize(output)
    print(binary_prediction.shape)
    print(binary_prediction.min())
    print(binary_prediction.max())

    # to train against a target, use the following loss function
    # the output is a pixelwise loss of shape [B, C, H, W]
    pixelwise_loss = model.compute_pixelwise_loss(input, target)
    print(pixelwise_loss.shape)

