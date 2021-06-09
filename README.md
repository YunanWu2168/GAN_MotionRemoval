# Motion Artifact Reduction on Liver MRI Using Generative Adversarial Networks

Team member: Yunan Wu, Xijun Wang.

Class: 2021-SPRING-CS-497: DEEP GENERATIVE MODELS

* [1. Introduction](#1-introduction)
    * 1.1 Objective
    * 1.2 Problem Description
    * 1.3 Related Work
* [2. Materials](#2-materials)
    * 2.1 Dataset
    * 2.2 Motion Simulation
* [3. Methods](#3-methods)
     * 3.1 Denoising Model
     * 3.2 The Generative Adversial Networks
     * 3.3 The Perceptive Loss
     * 3.4 Statistical Analysis
     * 3.5 Experimental Design
* [4. Results ](#4-results)
     * 4.1 Computational Result
     * 4.2 Visualizations
* [5. Discussion](#5-discussion)
     * 5.1 Discussion
     * 5.2 Limitations and Future Work


# **1 Introduction**

**1.1 Objective**

Magnetic resonance imaging (MRI) is the most commonly used imaging modality for diagnosis of liver cancer and other liver diseases. Liver MRI images often suffer degraded quality from ghosting or blurring artifacts caused by patient respiratory or bulk motion. The liver MRI sequence is continuously acquired before and after the intravenous administration of gadolinium-based contrast agents at several time points and requires patients to hold their breath for 15-25 seconds during each imaging acquisition. The critical timing required for this sequence allows for only a few seconds of patient breathing time between imaging acquisitions which causes patient fatigue and thus respiratory motion artifacts which degrade image quality through ghosting artifact or blurring of the images. As a result, the repeated scanning is needed. However, the repeated scanning is inefficient due to the high cost and the extra examination time. In addition, it is difficult for patients to lie down motionlessly for up to 10 minutes, so the image quality from the second scanning is hard to promise. In order to account for all these issues, methods need to be developed to reduce these motion artifacts and facilitate both patients and radiologists in the future research.  

**1.2 Problem Description**

We want to propose a better method to reduce the motion artifacts on the liver MRI caused by breath. This project has several challenges. First of all, there are no ground truth (i.e., clean images) for the liver MRIs with real motions. We need to simulate the motion artifacts on the clean image in the K-space to mimic the real motion, which is hard because the respiratory motions are caused by many complex factors. Second, the reconstructed image should be clear enough to show the details of livers. Many methods we found before from other literatures didn’t have a good denoising performance on the liver MRIs. Therefore, we need to propose a generative method that not only denoises the image, but also keeps the critical tissue structures and the details of livers, i.e. more realistic and higher perceptual quality. Third, the method needs to be generalizable when tested on the real images.  

**1.3 Related Work**

Several preprocessing and postprocessing strategies have been applied to mitigate motion artifacts in liver MRI. The simplest approach is to instruct the patients how to hold the breath before the examination. In clinical MRI examinations, patients are given written and visual instructions to practice breath-holding required for imaging which can help reduce the number of repeat imaging [1] acquisitions to correct artifacts. However, once the contrast agent is injected, it is not practical to stop and repeat the MRI image acquisition if patient respiratory motion occurs because of the timing requirements and gadolinium toxicity. Advanced image reconstruction techniques such as compressed sensing (CS) have been used to remove respiratory motion artifacts [2]–[5], where signals are reconstructed from the highly under-sampled signal acquisitions to produce an MR image within a shortened scan time. Radial K-space sampling combined with CS and parallel imaging enabled free breathing acquisitions of liver MRI [4]. Although the CS has been proved to help motion reduction, it was limited by the acceleration rate and availability from different MRI manufacturers and platforms. Also, patients with irregular breathing patterns are not ideal for use with the CS technique. Therefore, the development of image-based post-processing methods on the MRI images will be useful to retrospectively mitigate motion artifacts and improve image quality.

Recent years, deep learning (DL) approaches have been developed in medical imaging use cases including image reconstruction and artifact reduction [6], [7], motion detection and correction [8], and image quality control [9], [10]. DL methods utilized convolutional neural networks (CNNs) to extract features of different types of artifacts and correct them in the brain [11], [12], abdominal [13]–[15] and cardiac imaging [8]. Only a few studies had worked on the motion reductions on liver MRI. Tamada et al. proposed a denoising CNN on multi-phase magnitude-only image patches that learned the artifact patterns as residual feature maps and then subtracted them from the original images to obtain the motion reduced images [14]. Kromrey et al. developed deep learning filters in CNNs for multi-arterial phase acquisitions and improved image quality on severely degraded images [15]. However, the motion artifacts used for model training in both studies were simulated based on specific re-ordering of K-space lines, which cannot represent the full spectrum of motion artifact patterns. It is necessary to create a more generalized CNN model that is able to reduce various degrees and types of motion artifacts occurring on liver MRIs. 

[Need some related work in GAN model]



# **2 Materials**

**2.1 Dataset**


**2.2 Motion Simulation**

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121229286-5e029380-c853-11eb-865d-edd42d0f77ed.png" />
  <br>
    <em>Figure 1. The process of motion simulation in K-space.</em>

</p>

For model training process, pairs of images without motion (i.e. clean images) and with simulated motion artifacts (i.e. simulated motion images) were generated by manipulating the K-space data of the clean images and/or transformed images. The signal phases or the order of a certain range of K-space data were altered to simulate different types and degrees of motion artifacts. The simulation steps were described in Fig. 1.  (i) A given image, which can be either a clean image or a transformed image, was converted to its K-space data by Fast Fourier Transform (FFT); (ii) K-space data of the clean image and/or transformed image was manipulated according to three different rules to form a new K-space.  (iii) A simulated motion image was reconstructed from the new K-space using inversed FFT (iFFT). The simulation process can be generalized in:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121228881-e9c7f000-c852-11eb-9972-a9c954689e10.png" />
</p>

where F and F^{-1}  denote FFT and iFFT, Y and Y_m represent the clean image and the image with simulated motion artifacts, and [∙] is the simulated K-space with motion. 
Three different types of motion were simulated to mimic realistic motion artifacts and their corresponding residual artifact images between the clean images and simulated motion images were shown in Fig. 2.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121230368-c7cf6d00-c854-11eb-8f51-750990d420d7.png" />
  <br>
    <em>Figure 2. Examples of three types of simulated motion artifacts: (ii) rotational motion, (iii) sinusoidal motion and (iv) random motion.</em>

</p>

(1) Rotational Motion 
	
One of the major motion artifacts is bulk body movement, which causes incoherent ghosting and blurring on images. The clean image was rotated to create a rotated image with a random rotational angle between -20˚ and 20˚, which then went through FFT to form the rotated K-space\ F\left(Y_r\right). A new K-space [FY]rotation (i.e.,  [FY]sim in Eq.1) was generated as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121230707-33b1d580-c855-11eb-9545-d1bf4048a6c2.png" />
</p>
where F\left(Y_r\right) represents the K-space domain of the rotated image, k is the K-space coordinate in the phase-encoding direction (-\pi<k<\pi), n is the number of continuous K-space lines (0< n < 128) in F\left(Y_r\right), and m is an index of K-space location (0 < m < 384). In each simulated scenario, a random number of n K-space lines from F\left(Y_r\right), starting at a randomly chosen K-space index m, were filled into the new K-space  [FY]rotation, and the rest of the original K-space F\left(Y\right) lines were kept in [FY]rotation.

(2) Sinusoidal Motion

To simulate periodic human respiratory cycles, sinusoidal motion patterns were generated by changing the duration, frequency and phase of the simulated sinusoidal wave. The new k-space [FY]sin (i.e.,  [FY]sim in Eq.1) was generated by altering the signal phase of the original K-space F\left(Y\right), defined as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121230982-78d60780-c855-11eb-8205-0f6560330e8e.png" />
</p>
, where \emptyset(k) denotes the phase shift error added to a given K-space line k along the phase-encoding direction, and \emptyset(k) is defined as:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121231028-87242380-c855-11eb-8d27-85b56e844de4.png" />
</p>

, where k_{max} is the range of center K-space lines that were preserved without adding phase shift errors, and k_{max} was randomly chosen from \pi/10 to \pi/2. ∆ is the number of pixels (0 < ∆ < 20), depicting the severity of motion, \alpha is the frequency of the respiratory cycle (0.1<\alpha<5 Hz), and \beta is the phase of the respiratory wave (0<\beta<\pi/4). Phase shift defined by \emptyset(k) was added to each K-space line k that was located outside the center range of ({-k}_{max}, k_{max}).

(3) Random Motion 

To simulate the irregular non-periodic respiratory motion, a new K-space with random motion \left[F\left(Y\right)\right]_{rand}\ (i.e.,  [FY]sim in Eq.1) was generated by adding signal phase shifts {\emptyset\left(k\right)}_{rand} to 10%-50% randomly selected peripheral K-space lines, whereas the center 4% - 10% of the original K-space F\left(Y\right) lines were kept intact [12]. Different percentage of the preserved center K-space lines represented different levels of motion severity. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121231199-bd61a300-c855-11eb-8444-89551e949432.png" />
</p>



# **3 Methods**

**3.1 The Denoising Model**

The deep residual network with densely connected multi-resolution block (DRN-DCMB) network was first applied to reduce the motion artifacts on the liver MRI. As shown in Fig. 3, the architecture of the deep residual network with densely connected multi-resolution blocks (DRN-DCMB) was composed of 3 multi-resolution blocks, followed by a basic convolutional block, which included a 3×3 convolutional layer, a batch normalization (BN) layer and a Leaky rectified linear unit (LeakyReLU) activation function. The BN layers were used to promote faster training and make the nonlinear activation functions viable. LeakyReLU was chosen due to its small slope for negative values in order to fix the “dying ReLU” problem and speed up the training. The U-net inner structure of each multi-resolution block consisted of a down-sampling path and an up-sampling path. Four 2×2 max-pooling layers were used in the down-sampling path to generate deeper levels of feature maps for extraction of local image details, such as vessels and tumors in the liver. Likewise, four 2×2 up-pooling layers were used to restore the feature maps back to the same size of the input image at each level. In particular, the feature maps from the down-sampling layers were concatenated to their corresponding up-sampling layers. By doing so, the global information such as organs and motion artifacts extracted by the down-sampling levels were combined with the local information exacted on the same levels along the up-sampling path. The 3 multi-resolution blocks were densely connected, in which the output from one previous multi-resolution block was concatenated to the inputs of all subsequent multi-resolution blocks. Dense connection between multi-resolution blocks preserved important learning features and therefore accelerated training speed without relearning redundant features.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121233763-9789cd80-c858-11eb-88da-ac712dd380ad.png" />
  <br>
    <em>Figure 3. The architecture of the DRN-DCMB model.</em>
</p>

In addition, residual learning [16] was used during the training process to learn the residual differences (i.e., artifacts) between the simulated motion image and the clean image. The model output image was generated by subtracting the residual map from the input image. Lastly, the fully convolutional layers in the DRN-DCMB model allowed the input image size to vary, for example, small image patches (64×64) randomly sampled from the full-size image were used as the input for training purpose, and the full-size images (512×512) were used as the input during the testing process. All trainable parameters were updated by minimizing the pixel-by-pixel mean square error (MSE) loss between the clean image and the DRN-DCMB model output image.

 
**3.2 The Generative Adversial Networks**

To further improve the perceptual quality of generated liver results, we decide to make use of the GAN framework, for its ability of generating the realistic images. We are keeping the DRN-DCMB as our generator network (Figure 1), and build the discriminator network shown in Figure 5.

<p align="center">
  <img src="https://user-images.githubusercontent.com/23268412/121280059-3fc08600-c89b-11eb-851c-25354df77278.png" />
  <br>
    <em>Figure 5. The architecture of the proposed discriminator network.</em>
</p>


**3.3 The Perceptive Loss**

**3.4 Statistical Analysis**

MSE and structural similarity index (SSIM) (35) were measured in the training dataset and the testing dataset-I with simulated motion. The mean squared difference between the clean image and the model output motion-reduced image was measured by MSE. SSIM is a widely used perceptual metric that addresses the differences in the structural, luminance, and contrast between two images.


<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/121234615-87beb900-c859-11eb-8971-b7db120b4845.png" />
</p>

, where Y and (Y)^  denotes the clean and model output motion-reduced image, respectively; μ is the mean intensity and δ  is the standard deviation of an image; c_1=0.01 and c_2 = 0.03 are the constants. 

MSE and SSIM values were calculated for each type and across all types of the simulated motion artifacts in training dataset and testing dataset. The SSIM and MSE values were compared between the output images generated by different models using the pair t-test. A p value < 0.05 indicated significant differences in comparison. All statistical analyses were performed using the SciPy library in Python 3.7.   


**3.5 Experimental Design**

The liver MRI image volumes that had no obvious motion artifacts (i.e., “clean image”) in 10 patients were selected in the training process , of which 8 was used for training and 2 for validation. Three different types of the simulated motion artifacts were added to these clean images.  During the training of the DRN-DCMB model, 20 small image patches (size: 64×64) were randomly generated from each full-size image, leading to a total of 150,000 patches (20 patches × 25 slices × 3motion types × 10 randomized parameters × 8 patients) for the training dataset, and 30,000 patches (20 patches × 25 slices ×3 motion types × 10 randomized parameters × 2 patients) for the validation dataset. Other training parameters included: batch size = 64, early stopping at the 34th epoch, and a learning rate initialized from 0.0001 using the Adam optimization algorithm. The training time was 3.7 hours for training the DRN-DCMB model.
During the training process of the GAN model, the full-size simulated motion images (size: 512×512) were used as the input, with 6,000 images (25 slices ×3 motion types x 10 randomized parameters ×8 patients) used for training and 1500 images (25 slices ×3 motion types × 10 randomized parameters × 2 patients) used for validation. Other training parameters included: batch size = 8, early stopping at the 9th epoch, and a learning rate initialized from 0.00005 using the Adam optimization algorithm. The training time was 2.6 hours for training the proposed model.

The testing datasets were used to evaluate the model performance, which consisted of 3750 clean images with simulated motion acquired from 5 patients (25 slices ×3 motion types × 10 randomized parameters ×5 patients). 


# **4 Results**


# **5 Discussion**


