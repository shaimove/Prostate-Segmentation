# Prostate-Segmentation
# Shape Regularization for Prostate Segmentation
## Previous work:  

In the following work, we suggest changing the architecture suggested by Ravishankar et al [1]. Ravishankar suggested improving semantic segmentation of simple U-net architecture, by adding shape-regularization obtain by convolutional auto-encoder. 

Let's define the following: I_k image,  S_k ground truth segmentation map, S_k_hat learned segmentation map by vanilla U-net. An auto-encoder with encoder E projecting the segmentation maps into a latent space M and then a decoder R reconstruct the segmentation map from the latent space  M to improved shape-regularized segmentation map.

Illustration of the architecture: 

 ![Image 1](https://github.com/shaimove/Prostate-Segmentation/blob/main/Results/explanation1.png)

The authors of [1] first trained U in an unregularized manner with L2 norm, then they trained an auto-encoder with corrupted ground truth segmentation maps (or by using predicted segmentation maps resulted from early epochs). Then, the auto-encoder was used to fine-tune U with the following loss function:

$$
L_U = 1/N * \sum_{k=1}^{N}(|\hat{S_k} - (R*E)(\hat{S_k})|^2 + \lambda_1 * |E[S_k] - E[\hat{S_k}]|^2 + \lambda_2 * |S_k - \hat{S_k}|^2)
$$

The first term constrains the learning process only to U so the output of the auto-encoder and U-net will be the same. The second term ensures that the encoding of the ground truth and learned segmentation map will be close to each other in the latent space. The last term trains U directly.

## Our Work:  
We suggest replacing the auto-encoder with a generative adversarial network (GAN), specifically with pix2pix architecture, as in Isola et al [2]. 

We suggest improving the training process by replacing the auto-encoder with a discriminator D with a patchGAN architecture when the U-net is used as a generator G. 

The training process will include both the loss from the discriminator and direct loss from the ground truth segmentation map, using Dice loss. 

The first step will be to train an unregularized U-net G, using Dice loss, instead of the L2 loss suggested by the authors of [1], and these results will be compared to the GAN architecture.

The second phase will be to train a generator and discriminator. The discriminator will be trained to discriminate between the ground truth segmentation map G[I_k]=S_k_hat to the segmentation map results from the U-net S_k_hat, given the MRI image I_k

$$
L_D = \mathbf{E}[logD(I_k,S_k)] + \mathbf{E}[log(1-D(I_k,G[I_k]))]
$$

The loss for the generator will include two parts: the loss from the discriminator (adversarial loss). Dice loss between the ground truth segmentation map and generator output. 

$$
L_G = \mathbf{E}[logD(I_k,S_k)] + \lambda*Dice(S_k,G[I_k])
$$

Illustration of the GAN training:

 ![Image 1](https://github.com/shaimove/Prostate-Segmentation/blob/main/Results/explanation2.png)

[1] Ravishankar H., Venkataramani R., Thiruvenkadam S., Sudhakar P., Vaidya V. (2017) Learning and Incorporating Shape Models for Semantic Segmentation. In: Descoteaux M., Maier-Hein L., Franz A., Jannin P., Collins D., Duchesne S. (eds) Medical Image Computing and Computer-Assisted Intervention − MICCAI 2017. MICCAI 2017. Lecture Notes in Computer Science, vol 10433. Springer, Cham. https://doi.org/10.1007/978-3-319-66182-7_24

[2] Isola, Phillip & Zhu, Jun-Yan & Zhou, Tinghui & Efros, Alexei. (2017). Image-to-Image Translation with Conditional Adversarial Networks. 5967-5976. 10.1109/CVPR.2017.632

