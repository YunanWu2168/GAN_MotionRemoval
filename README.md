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

We want to propose a better method to reduce the motion artifacts on the liver MRI caused by breath. This project has several challenges. First of all, there are no ground truth (i.e., clean images) for the liver MRIs with real motions. We need to simulate the motion artifacts on the clean image in the K-space to mimic the real motion, which is hard because the respiratory motions are caused by many complex factors. Second, the reconstructed image should be clear enough to show the details of livers. Many methods we found before from other literatures didn’t have a good denoising performance on the liver MRIs. Therefore, we need to propose a generative method that not only denoises the image, but also keeps the critical tissue structures. Third, the method needs to be generalizable when tested on the real images.  

**1.3 Related Work**

Several preprocessing and postprocessing strategies have been applied to mitigate motion artifacts in liver MRI. The simplest approach is to instruct the patients how to hold the breath before the examination. In clinical MRI examinations, patients are given written and visual instructions to practice breath-holding required for imaging which can help reduce the number of repeat imaging [1] acquisitions to correct artifacts. However, once the contrast agent is injected, it is not practical to stop and repeat the MRI image acquisition if patient respiratory motion occurs because of the timing requirements and gadolinium toxicity. Advanced image reconstruction techniques such as compressed sensing (CS) have been used to remove respiratory motion artifacts [2]–[5], where signals are reconstructed from the highly under-sampled signal acquisitions to produce an MR image within a shortened scan time. Radial K-space sampling combined with CS and parallel imaging enabled free breathing acquisitions of liver MRI [4]. Although the CS has been proved to help motion reduction, it was limited by the acceleration rate and availability from different MRI manufacturers and platforms. Also, patients with irregular breathing patterns are not ideal for use with the CS technique. Therefore, the development of image-based post-processing methods on the MRI images will be useful to retrospectively mitigate motion artifacts and improve image quality.
Recent years, deep learning (DL) approaches have been developed in medical imaging use cases including image reconstruction and artifact reduction [6], [7], motion detection and correction [8], and image quality control [9], [10]. DL methods utilized convolutional neural networks (CNNs) to extract features of different types of artifacts and correct them in the brain [11], [12], abdominal [13]–[15] and cardiac imaging [8]. Only a few studies had worked on the motion reductions on liver MRI. Tamada et al. proposed a denoising CNN on multi-phase magnitude-only image patches that learned the artifact patterns as residual feature maps and then subtracted them from the original images to obtain the motion reduced images [14]. Kromrey et al. developed deep learning filters in CNNs for multi-arterial phase acquisitions and improved image quality on severely degraded images [15]. However, the motion artifacts used for model training in both studies were simulated based on specific re-ordering of K-space lines, which cannot represent the full spectrum of motion artifact patterns. It is necessary to create a more generalized CNN model that is able to reduce various degrees and types of motion artifacts occurring on liver MRIs. 

[Need some related work in GAN model]



# **2 Materials**

**2.1 Dataset**



# **3 Methods**


# **4 Results**


# **5 Discussion**


