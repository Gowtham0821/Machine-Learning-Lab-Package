# Gradient Inversion Attacks in Federated Learning

A machine learning laboratory project demonstrating how private training
data may be reconstructed from shared model gradients. The
implementation is based on the gradient-matching idea introduced in
**Deep Leakage from Gradients (DLG)** by Zhu, Liu, and Han (NeurIPS
2019).

> **Note:** This is a controlled educational demonstration. It assumes
> access to the model architecture/state and corresponding saved
> gradients; it does not show that every federated-learning system or
> dataset can be reconstructed.

## Problem Statement

Federated learning allows clients to train a shared model without
directly sending raw training examples to a central server. However,
gradients shared during training can still contain information about
those examples. This project explores that privacy risk by attempting to
reconstruct image data from saved gradients.

## Project Objective

-   Generate per-patch gradients from an image on the client side.
-   Save model state and gradient information for the attacker.
-   Optimize dummy image pixels and dummy-label logits so their
    gradients match the observed gradients.
-   Reassemble reconstructed patches into a full RGB image and compare
    it with the reference.

## How It Works

1.  **Prepare the image:** Convert the input image to RGB, resize it to
    `256 × 256`, and normalize pixel values.
2.  **Split into patches:** Divide the image into an `8 × 8` grid,
    giving 64 patches of `32 × 32` pixels each.
3.  **Compute client gradients:** Run each patch through the model and
    calculate loss gradients.
4.  **Optimize dummy data:** The attacker minimizes the distance between
    gradients produced by dummy data and saved target gradients.
5.  **Reconstruct the image:** Place recovered patches at their original
    coordinates to produce the final image.

The project description specifies a small convolutional model (LeakNet),
cross-entropy loss, and L-BFGS optimization. The attack script uses
multiple random restarts and early-stopping controls.

## Repository Contents

  -----------------------------------------------------------------------
  File or folder                      Purpose
  ----------------------------------- -----------------------------------
  `client (1).py`                     Client-side image preparation,
                                      patch processing, gradient
                                      calculation, and export

  `attacker (1).py`                   Gradient-matching optimization and
                                      image reconstruction

  `gradients.pt`                      Serialized model/gradient package
                                      used by the demonstration

  `original.jpg`                      Original input image

  `original_reference.png`            Reference image used for comparison

  `reconstructed.png`                 Reconstructed image produced by the
                                      experiment

  `paper/`                            Reference paper or related research
                                      material

  `progress/`                         Project progress material and
                                      snapshots
  -----------------------------------------------------------------------

File names are shown as they appear in the repository. If you rename the
Python files to `client.py` and `attacker.py`, use those names in the
commands below.

## Requirements

-   Python 3.9 or a compatible Python version
-   PyTorch
-   Pillow
-   NumPy
-   Matplotlib (for optional plotting)

Install the core packages:

``` bash
python -m pip install torch torchvision pillow numpy matplotlib
```

Exact package versions may need to be adjusted for your operating system
and CUDA setup. The scripts may use only a subset of these packages.

## Getting Started

Clone the repository:

``` bash
git clone https://github.com/Gowtham0821/Machine-Learning-Lab-Package.git
cd Machine-Learning-Lab-Package
```

Run the client script first to generate or export the gradient package,
if required:

``` bash
python "client (1).py"
```

Then run the attacker script:

``` bash
python "attacker (1).py"
```

The scripts may expect particular input paths or configuration values.
Check the paths and settings near the top of each script before running.
The repository also contains a saved `gradients.pt` package and image
outputs, so you can inspect the bundled artifacts without rerunning the
experiment.

## Results and Evaluation

The project materials report the following values for the supplied
reference/reconstruction pair:

-   **Mean Squared Error (MSE):** `0.00001337`
-   **Peak Signal-to-Noise Ratio (PSNR):** `48.74 dB`
-   **Patch coverage:** 64 patches arranged in an `8 × 8` grid

These measurements apply to the bundled image pair only. They do not
guarantee similar results for other images, model architectures, labels,
batch sizes, or federated-learning deployments.

## Security Implications

The experiment illustrates that keeping raw data on a client does not
automatically guarantee privacy when model gradients are shared.
Potential mitigation approaches include:

-   Differential privacy
-   Gradient clipping
-   Secure aggregation
-   Limiting unnecessary update exposure and precision

Each defense has assumptions and trade-offs. Secure aggregation can
limit an aggregator's visibility into individual client updates, but
does not by itself eliminate every privacy risk.

## Research Reference

Zhu, L., Liu, Z., & Han, S. (2019). **Deep Leakage from Gradients.**
*Advances in Neural Information Processing Systems (NeurIPS).*\
https://proceedings.neurips.cc/paper_files/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

## Authors

-   Varshin Adhreya A S
-   Gowtham M

## Disclaimer

This project is intended for educational and research purposes to help
understand privacy risks in gradient-sharing machine learning workflows.
Use the techniques responsibly and only with data and systems you are
authorized to test.
