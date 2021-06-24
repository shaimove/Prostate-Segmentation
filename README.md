# Prostate-Segmentation
# Shape Regularization for Prostate Segmentation
## Previous work:  

In the following work, we suggest changing the architecture suggested by Ravishankar et al [1]. Ravishankar suggested improving semantic segmentation of a simple ("vanilla") U-net architecture, by adding shape-regularization obtain by a convolutional auto-encoder (AE). 

Let's define the following:  I image,  S  ground truth segmentation map, S hat learned segmentation map by a vanilla U-net, U. An auto-encoder with encoder E projecting the segmentation maps into a latent space M and a decoder R reconstruct the segmentation map from the latent space M to improved shape-regularized segmentation map.

Illustration of the architecture: 

 ![Image 1](https://github.com/shaimove/Prostate-Segmentation/blob/main/Results/explanation1.png)

The authors of [1] first trained U in an unregularized manner (probably with BCE loss). Then, the authors created corrupted segmentation maps from the ground truth segmentation maps (or by using predicted segmentation maps resulted from early epochs), and train the AE to correct the corrupted images to the ground truth segmentation maps. As a final step, the authors freeze the learning of the AE, and using the following loss function and information from the latent space of the AE, they fine-tune U:



The first term constrains the learning process only to U so the output of the auto-encoder and U-net will be the same. The second term ensures that the encoding of the ground truth and learned segmentation map will be close to each other in the latent space. The last term trains U directly.

## Our Work:  
We suggest replacing the auto-encoder with a generative adversarial network (GAN), specifically with pix2pix architecture, as in Isola et al [2]. 

We suggest improving the training process by replacing the auto-encoder with a discriminator D, with a patchGAN architecture when the U-net is the generator G. 

**The first step** will be to train the U-net G, using Adversarial Loss  from the discriminator (BCE with Logits) and Dice loss from semantic segmentation error (the authors didn't specify their metric for initial U-net training, probably Binary Cross Entropy - BCE). 



The Discriminator will train using the adversarial loss (BCE with Logits), given the MRI 2D image, and real/generated segmentation map, the discriminator will be trained to distinguish between the real and generated segmentation map:



Illustration of the GAN training:

 ![Image 1](https://github.com/shaimove/Prostate-Segmentation/blob/main/Results/explanation2.png)



**The second phase** will be to train and discriminator, given the MRI 2D image, to distinguish between the ground truth segmentation maps and a corrupted segmentation maps. The corrupted images are the output of the generator after 25 epochs, the same method was used by the authors of [1] and marked as G[I]_c.



**The third phase**, fine-tuning of the generator (U-net), training the with Dice loss on the semantic segmentation, and adversarial loss (BCE Logits) from the discriminator after shape regularization.  



## How to run the network?:  

First, you need to download the PROMISE12 dataset from the competition website, or the following Kaggle link:

https://www.kaggle.com/aaryapatel98/prostate

if you chose to download the dataset from the Kaggle dataset, you need to use the X_train and Y_train npy files (X_train(Clahe) and X_train(Hist) are the same datasets, with contrast enhancement, which wasn't used in the PROMISE12 original dataset).

Then, you need to split the dataset for training, validation, and test sets (ReadSplitData function from utils.py). 

In parallel to the repository folder, create a folder named PROMISE12, with dataset files, and all results will be saved to this folder. 

if you want to train our version, to the authors of [1] architecture, run MainAE.py, the script will create folders named [unet,ae,unetSR] in PROMISE12 and save the results there. If you want to train our GAN architecture, run MainGAN.py, and the script will save the results to folders named [pix2pix,Disc,pix2pixSR]. 

After training both architectures, run the Test.py script, to compare between the two architectures.

It's suggested to train the network on a computer with a minimum RAM of 32GB (the dataset was uploaded to memory when the dataset was initialized, to reduce latency between batches), and a strong GPU, we used NVIDIA GeForce RTX3090 with 24GB. 

We used PyTorch library, which can be installed with CUDA11 (required for RTX3090), using the following command to Anaconda cmd:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

or pip install to windows command line:

```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Authors:  

Sharon Haimov, a MSc student for Biomedical Engineering ,The Technion, Israel

Yaniv Ziselman, a MSc student for Biomedical Engineering ,The Technion, Israel

## License:  

[1] Ravishankar H., Venkataramani R., Thiruvenkadam S., Sudhakar P., Vaidya V. (2017) Learning and Incorporating Shape Models for Semantic Segmentation. In: Descoteaux M., Maier-Hein L., Franz A., Jannin P., Collins D., Duchesne S. (eds) Medical Image Computing and Computer-Assisted Intervention − MICCAI 2017. MICCAI 2017. Lecture Notes in Computer Science, vol 10433. Springer, Cham. https://doi.org/10.1007/978-3-319-66182-7_24

[2] Isola, Phillip & Zhu, Jun-Yan & Zhou, Tinghui & Efros, Alexei. (2017). Image-to-Image Translation with Conditional Adversarial Networks. 5967-5976. 10.1109/CVPR.2017.632

