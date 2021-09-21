
* [1. Abstract](#1-abstract)
* [2. Proposed-Model](#2-proposed-model)
* [3. Results ](#3-results)
     * 3.1 Computational Result
     * 3.2 Visualizations



# **1 Abstract**
The respiratory motion and the resulting artifacts are considered to be a big problem in abdominal Magnetic Resonance Imaging (MRI), often burdening hospitals with unnecessary cost of time and money. Many deep learning techniques have recently been developed for removing these respiratory motion artifacts. However, fine-details, such as the vessels in liver MRIs, are often over-smoothed by many models, while these details are most important for medical diagnosis. Therefore, more realistic-looking reconstructed images are needed. To achieve this goal, in this paper, we design a Generative Adversarial Networks (GAN)-based model, and use the perceptual loss as part of our training loss to further enhance the images' perceptual quality. To ensure the proper convergence of the GAN model, we add a pre-training step before the adversarial training. In addition, due to the lack of image pairs of the clean image and its corresponding motion distorted image, we develop a motion simulation method via the k-space to mimic the motion-generated artifacts. Benefiting from all these methods, our proposed model achieves high perceptual quality with an MSE of 0.002, an SSIM of 0.942, a PSNR of 33.894 and a perceptual distance metric of 0.015. Finally, it generates motion-reduced images with clearer and better fine-details, thus providing radiologists with more realistic MRI images for improving diagnosis.

# **2 Proposed-Model**
<p align="center">
  <img src= "https://user-images.githubusercontent.com/23268412/134108619-84f31500-474f-4e28-aa47-626d78b45a68.png" />
  <br>
    <em>Figure 1. The architecture of the proposed percepGAN-preG model.</em>
</p>


# **3 Results**


<p align="center">
  <img src= "https://user-images.githubusercontent.com/23268412/134109168-4a7b7dd2-67a2-4a72-a850-c94a2c096042.png" />
  <br>
    <em>Figure 2. Motion reduction examples by the proposed model on three types of motion simulation: A. random motion; B. interweaving motion and C. sinusoidal motion. The first row: clean images; second row: images with specific simulated motion; third row: motion reduced images by the proposed model.</em>
</p>

