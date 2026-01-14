# CT-images-Homogenization

## Motivation
**Computed Tomography (CT)** images are widely used in medical research and clinical practice. However, images acquired from different scanners or using different acquisition and reconstruction settings can look substantially different, even when they belong to the same patient. These differences often appear as variations in image texture and contrast that are unrelated to the underlying anatomy or pathology. 

These discrepancies can significantly affect image interpretability for both clinicians and machine learning and deep learning models. In particular, models trained on data from a single scanner or acquisition protocol often struggle to generalize to images from different sources, limiting their reliability and real-world applicability.

Therefore, there is a **strong need for a single, consistent, and reliable image representation** that minimizes acquisition-induced variability while preserving clinically relevant anatomical and textural information. Achieving such harmonization is a **crucial step toward enabling reproducible analysis and the safe deployment of data-driven methods in real-world clinical settings**.

## Idea
Given the variability introduced by different CT acquisition procedures, we model CT harmonization as a **domain translation problem** driven by texture differences.

**CT** images are first **grouped into domain**s based on the distribution of their radiomic texture features. Each **domain represents a consistent acquisition setup**. Through statistical analysis, we identify scanner type and reconstruction kernel as the primary sources of variability; consequently, **each scanner–kernel combination defines a domain**.

We then **train a deep learning model capable of one-to-many domain translation**, allowing **any CT image from a source domain to be transformed into its corresponding representation in a target domain**. The model focuses on aligning texture characteristics across domains while preserving the underlying anatomical structures.

## Dataset
**LIDC-IDRI** (The Lung Image Database Consortium and Image Database Resource Initiative) https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

## Texture-Aware StarGAN architecture 
Our approach builds upon the StarGAN architecture, a multi-domain image-to-image translation framework designed to transform images across different styles using a single unified model. 
While StarGAN provides a strong baseline for cross-domain translation, it does not explicitly account for the complex texture variations that characterize CT images acquired with different scanners and reconstruction kernels.

To address this limitation, we extend the original architecture by **introducing a multi-scale texture loss function**, which explicitly enforces texture consistency across domains at multiple spatial scales. This addition enables the model to better capture and harmonize kernel-induced texture differences while preserving anatomical structures.

<img width="500" height="400" alt="Texture-Aware-StarGAN-image_only_tx_loss" src="https://github.com/user-attachments/assets/62124675-56c8-4d6a-a3ae-766d121c2036" />

Focusing on the Texture and Cycle Loss block, the architecture incorporates the following components:
- a **cycle consistency loss**, which enforces the reconstruction of the original image after a round-trip translation, ensuring that domain-invariant and anatomical characteristics are preserved;
- a **multi-scale texture extractor** that computes textural representations using Gray-Level Co-occurrence Matrices (GLCMs) across multiple spatial resolutions and angular directions;
- an **aggregation module that dynamically combines these multi-scale texture representations into a single scalar objective**, referred to as the **Multi-Scale Texture Loss**, which guides the model toward texture-consistent harmonization across domains.

<img width="500" height="400" alt="tx_loss_box" src="https://github.com/user-attachments/assets/7b9a056d-bcd0-4029-9437-6effb410288a" />

## Evaluation methods
In order to evaluate the performances of this model, we used three complementary analysis:
- **Deep Features Alignment**: measure of the similarity between the real and generated images through the Fréchet Inception Distance (FID).
  **GOAL:**evaluate the quality and diversity of generated images;
- **Radiomic Features Alignment**: statistical comparison of radiomic features of input and target domains before and after harmonization.
  **GOAL:**assess the impact of harmonization on radiomic feature differences across input and target domains;
- **Edge Evaluation**: comparison of lungs structure extracted from the image before and after harmonization.
  **GOAL:**evaluate the network's ability to preserve structural details during generation.



