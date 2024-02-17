This repository contains the implementation of the following task:

Implement three VAE models with latent spaces that are: 
(1) continuous, (2) discrete, (3) both.
- Model: There are many ways to implement this model. For example, you may consider whether the encoders' weights are shared or not, or how to combine the two latent spaces.
         For convolutions that upscale the input’s spatial size (for the decoder/generator), you may use nn.ConvTranspose2d . Note, there is no restriction to what layers you may use.

- Dataset: You will use the MNIST dataset. You will randomly color each sample with a different color. You can implement this using a Dataset class that is wrapping the MNIST dataset (this is a suggestion; you can do it however you like if it is correct). The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning.

- Outputs visualization: Generate images from your model and visualize its latentspaces.
                         You can compare different architectures for this purpose (e.g., low/high dimension of latent spaces, expressive/inexpressive encoder, etc.).
                         Demonstrate how to control the color of the output sample using the latent space. We encourage you to get creative with how you combine the latent spaces and visualize them.


Make sure you check out the included report PDF for notes & conclusions.
