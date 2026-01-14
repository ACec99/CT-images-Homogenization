# CT-images-Homogenization

## Motivation
**Computed Tomography (CT)** images are widely used in medical research and clinical practice. However, images acquired from different scanners or using different acquisition and reconstruction settings can look substantially different, even when they belong to the same patient. These differences often appear as variations in image texture and contrast that are unrelated to the underlying anatomy or pathology. \\
These discrepancies can significantly affect image interpretability for both clinicians and machine learning and deep learning models. In particular, models trained on data from a single scanner or acquisition protocol often struggle to generalize to images from different sources, limiting their reliability and real-world applicability.

Therefore, there is a **strong need for a single, consistent, and reliable image representation** that minimizes acquisition-induced variability while preserving clinically relevant anatomical and textural information. Achieving such harmonization is a **crucial step toward enabling reproducible analysis and the safe deployment of data-driven methods in real-world clinical settings**.
