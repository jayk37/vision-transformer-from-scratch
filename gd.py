import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm

DEVICE = "cuda"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


def gradient_descent(input, model, loss, iterations=256, learning_rate=0.01, jitter=False):
    """
    Perform gradient ascent to optimize the input image so that the model
    maximizes the activation for a specific target (class neuron).
    """

    # Ensure Gradients are Tracked and Retained for Input Tensor
    input.requires_grad_()
    input.retain_grad()

    # Apply Normalization and Jittering to the Image
    if jitter:
        input = normalize_and_jitter(input)

    for i in tqdm(range(iterations)):
        
        # Zero The Gradients from Previous Iteration
        model.zero_grad()

        # Perform Forward Pass for Current Input
        output = model(input)

        # Compute Loss
        loss_value = loss(output)

        # Compute Gradients of Loss w.r.t. Input Image (retain_graph=True Keeps the Graph for Subsequent Iterations)
        loss_value.backward(retain_graph=True)

        # Perform Gradient Ascent
        if input.grad is not None:
            with torch.no_grad():
                # Normalizing the Gradient to Prevent Exploding Updates
                grad = input.grad / (input.grad.norm() + 1e-8)
                # Updating Input Image using Gradient Ascent
                input += learning_rate * grad

        # Clamp Input Values
        input.data = input.data.clamp(0, 1)

        # Zero Gradients for the Next Iteration
        if input.grad is not None:
            input.grad.zero_()

    return input


def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    Try setting the modeul to `model.features[20]` and the loss to `tensor[0, ind].mean()`
    to see what intermediate activations activate on.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()
