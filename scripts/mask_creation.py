import numpy as np
import pydicom
import os
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from preprocessing_base_functions import step1_python, process_mask

def create_lungs_mask(patientID, scan):

    try:

        print("I'm building the mask for patient", patientID)
        im, m1, m2, spacing = step1_python(scan)

        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2

        return dilatedMask

    except:

        print('bug in ' + patientID)
        raise