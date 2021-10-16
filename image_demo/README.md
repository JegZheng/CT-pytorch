# Demo code for conditional transport (CT) on image experiments

## Requirements
- pytorch >= 1.2.0
- numpy

## Implementation Details
This code is built on the implementation from the [repo](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan) that implements both DCGAN-like and ResNet GAN architectures. 
In addition, training with standard, Wasserstein, and hinge losses is possible. 


## Example usage
To run the DCGAN backbone:

`$ python main.py --model dcgan --loss ct`

or run the SNGAN backbone:

`$ python main.py --model resnet --loss ct`

Use --help for more options
`$ python main.py --help`