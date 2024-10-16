import numpy as np # linear algebra
import pydicom
import os

import scipy.ndimage
from  scipy.ndimage import zoom , rotate, binary_dilation, generate_binary_structure
import scipy
import math
import re

from skimage import measure
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from scipy.ndimage import binary_dilation
from scipy.ndimage.interpolation import zoom
import numpy as np
import warnings

# Load the scans in given folder path

def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: abs(float(x.ImagePositionPatient[2])))
    return slices

def get_pixels_hu(slices):
    if len(slices) == 1:
        image = slices[0].pixel_array
    else:
        image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
        
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
            
    return np.array(image, dtype=np.int16)

                                                                                                ### START: FUNCTIONS RELATED TO MASK CREATION ###

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1], label[-1 - cut_num, -1, 0],
                    label[-1 - cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1 - cut_num, 0, mid], label[-1 - cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)

def fill_hole(bw):
    """
    Fill 3D holes in the binary image.

    This function identifies the corner components of the 3D image and removes them.
    It then inverts the image to fill the holes (i.e., the background becomes the foreground and vice versa).

    Parameters:
    bw (ndarray): The input 3D binary image.

    Returns:
    ndarray: The processed 3D binary image with holes filled.
    """
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    """
    Binarize an image per slice based on certain conditions.

    This function applies a Gaussian filter to the image and then selects the proper components based on their area and eccentricity.
    The function also checks if corner pixels are identical, and if so, it applies the Gaussian filter to the slice.

    Parameters:
    image (ndarray): The input image.
    spacing (tuple): The spacing of the image.
    intensity_th (int, optional): The intensity threshold. Default is -600.
    sigma (int, optional): The sigma value for the Gaussian filter. Default is 1.
    area_th (int, optional): The area threshold. Default is 30.
    eccen_th (float, optional): The eccentricity threshold. Default is 0.99.
    bg_patch_size (int, optional): The size of the background patch. Default is 10.

    Returns:
    ndarray: The binarized image.
    """
    bw = np.zeros(image.shape, dtype=bool)

    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma,
                                                       truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.gaussian_filter(image[i].astype('float32'), sigma,
                                                       truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw

def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def step1_python(scan):
    #case = load_scan(case_path)
    scan_pixels = get_pixels_hu(scan)
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    bw = binarize_per_slice(scan_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68, 7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return scan_pixels, bw1, bw2, spacing

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 2 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask

def load_mask(path):
    mask_dcm = os.listdir(path)[0]
    mask = pydicom.dcmread(path + '/' + mask_dcm)
    return mask

                                                                                                ### END: FUNCTIONS RELATED TO MASK CREATION ###


def resample(image, scan, new_spacing):
    # Determine current pixel spacing
    if len(scan) == 1: # we're resampling a single slice
        pixel_spac_np = list(float(value) for value in scan[0].PixelSpacing)
        spacing = np.array(pixel_spac_np, dtype=np.float32)
    else:
        pixel_spac_np = list(float(value) for value in scan[0].PixelSpacing)
        spacing = np.array([float(scan[0].SliceThickness)] + pixel_spac_np, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = zoom(image, real_resize_factor, mode='nearest')

    return image, real_resize_factor

def cut_scan(image, coords, ref_dims, scale_factor):
    
    img_depth = int(image.shape[0])
    img_width = int(image.shape[1])
    img_height = int(image.shape[2])
    
    ref_width = int(int(ref_dims['x']) * ref_dims['scale_factor_x'])
    ref_height = int(int(ref_dims['y']) * ref_dims['scale_factor_y'])
    ref_depth = int(int(ref_dims['z']) * ref_dims['scale_factor_z'])
    
    # we must have square (width and height must be the same)
    if ref_width > ref_height:
        ref_height = ref_width
    elif ref_height > ref_width:
        ref_width = ref_height
        
    #print("the reference width is:", ref_width)
    #print("the reference height is:", ref_height)
    #print("the reference depth is:", ref_depth)
    
    ### STARTING AND ENDING POINT OF Z-AXIS (DEPTH OF THE VOLUME) ###
    
    start_z = int(int(coords[2]) * scale_factor[0])
    end_z = int(int(coords[5]) * scale_factor[0])
    dim_z = (end_z - start_z) + 1
    
    if dim_z == img_depth: # it's one of the volumes with a reconstructed mask
        depth_center = coords[6]
        half_ref_depth = (ref_depth // 2) + 1 # we take the first closest and highest integer
        start_z = depth_center - half_ref_depth
        end_z = depth_center + half_ref_depth
    else:
        if dim_z != ref_depth:
            diff_depths = ref_depth - dim_z
            half_diff_depths = diff_depths // 2 
            int_or_not_d = diff_depths % 2
            if int_or_not_d == 0:
                new_start_z = start_z - half_diff_depths
                if new_start_z < 0:
                    start_z = 0
                    end_z = end_z + half_diff_depths + abs(new_start_z)
                else:
                    new_end_z = end_z + half_diff_depths
                    if new_end_z >= img_depth:
                        surplus_z = new_end_z - (img_depth - 1) # I subtract 1 because there isn't the position img_depth (the last position is img_depth - 1)
                        start_z = new_start_z - surplus_z
                        end_z = img_depth - 1
                    else:
                        start_z = new_start_z
                        end_z = new_end_z
            else:
                new_start_z = start_z - (half_diff_depths + 1)
                if new_start_z < 0:
                    start_z = 0
                    end_z = end_z + half_diff_depths + abs(new_start_z)
                else:
                    new_end_z = end_z + half_diff_depths
                    if new_end_z >= img_depth:
                        surplus_z = new_end_z - (img_depth - 1)
                        start_z = new_start_z - surplus_z
                        end_z = img_depth - 1
                    else:
                        start_z = new_start_z
                        end_z = new_end_z
    
    #print("the difference among starting and ending point of the z axis is:", end_z - start_z)
    
    ### STARTING AND ENDING POINT OF X-AXIS (LENGTH OF THE LUNG) ###
    
    start_x = int(int(coords[0]) * scale_factor[1])
    end_x = int(int(coords[3]) * scale_factor[1])
    
    #print("the start x at first is:", start_x)
    #print("the end x at first is:", end_x)
    
    dim_x = (end_x - start_x) + 1
    
    if dim_x != ref_width:
        diff_width = ref_width - dim_x
        #print("diff_width:", diff_width)
        half_diff_width = diff_width // 2 
        #print("half_diff_width:", half_diff_width)
        int_or_not_w = diff_width % 2
        if int_or_not_w == 0:
            new_start_x = start_x - half_diff_width
            if new_start_x < 0:
                start_x = 0
                end_x = end_x + half_diff_width + abs(new_start_x)
            else:
                new_end_x = end_x + half_diff_width
                if new_end_x >= img_width:
                    surplus_x = new_end_x - (img_width - 1)
                    start_x = new_start_x - surplus_x
                    end_x = img_width - 1
                else:
                    start_x = new_start_x
                    end_x = new_end_x
        else:
            new_start_x = start_x - (half_diff_width + 1)
            if new_start_x < 0:
                start_x = 0
                end_x = end_x + half_diff_width + abs(new_start_x)
            else:
                new_end_x = end_x + half_diff_width
                if new_end_x >= img_width:
                    surplus_x = new_end_x - (img_width - 1)
                    start_x = new_start_x - surplus_x
                    end_x = img_width - 1
                else:
                    start_x = new_start_x
                    end_x = new_end_x
    
    #print("the difference among starting and ending point of the x axis is:", end_x - start_x)
    #print("the start_x is:", start_x)
    #print("the end_x is:", end_x)
    
    ### STARTING AND ENDING POJNT OF Y-AXIS (HEIGHT OF THE LUNG) ###
    
    start_y = int(int(coords[1]) * scale_factor[2])
    end_y = int(int(coords[4]) * scale_factor[2])
    
    #print("start_y at first is:", start_y)
    #print("end_y at first is:", end_y)
    
    dim_y = (end_y - start_y) + 1
    
    if dim_y != ref_height:
        diff_heights = ref_height - dim_y
        half_diff_heights = diff_heights // 2 
        int_or_not_h = diff_heights % 2
        if int_or_not_h == 0:
            new_start_y = start_y - half_diff_heights
            if new_start_y < 0:
                start_y = 0
                end_y = end_y + half_diff_heights + abs(new_start_y)
            else:
                new_end_y = end_y + half_diff_heights
                if new_end_y >= img_height:
                    surplus_y = new_end_y - (img_height - 1)
                    start_y = new_start_y - surplus_y
                    end_y = img_height - 1
                else:
                    start_y = new_start_y
                    end_y = new_end_y
        else:
            new_start_y = start_y - (half_diff_heights + 1)
            if new_start_y < 0:
                start_y = 0
                end_y = end_y + half_diff_heights + abs(new_start_y)
            else:
                new_end_y = end_y + half_diff_heights
                if new_end_y >= img_height:
                    surplus_y = new_end_y - (img_height - 1)
                    start_y = new_start_y - surplus_y
                    end_y = img_height - 1
                else:
                    start_y = new_start_y
                    end_y = new_end_y
    
    #print("the difference among starting and ending point of the y axis is:",  end_y - start_y)
    #print("start_y:", start_y)
    #print("end_y:", end_y)
    
    ### CROP IMAGE ###
    
    crop_img = np.full((ref_depth, ref_width, ref_height), -1024)
    
    #missing_slices = False
    padding = False
    num_pix_added_sx = 0
    num_pix_added_dx = 0
    
    if start_z < 0:
        start_z = 0
    
    if end_x > img_width-1:
        padding = True
        difference = end_x - (img_width-1)
        res = difference % 2
        if res == 0:
            num_pix_added_sx = difference / 2
            num_pix_added_dx = difference / 2
        else:
            num_pix_added_sx = difference // 2
            num_pix_added_dx = (difference // 2) + 1
    elif start_x < 0:
        padding = True 
        difference = abs(start_x)
        res = difference % 2
        if res == 0:
            num_pix_added_sx = difference / 2
            num_pix_added_dx = difference / 2
        else:
            num_pix_added_sx = difference // 2
            num_pix_added_dx = (difference // 2) + 1
    
    num_pix_added_dx = int(num_pix_added_dx)
    num_pix_added_sx = int(num_pix_added_sx)
            
    for s in range(image.shape[0]):
        if s >= start_z and s <= end_z:
            idx = s - start_z
            if padding == True:
            
                ### additive columns ###
                
                additive_cols_sx = np.full((img_height, num_pix_added_sx), -1024)
                additive_cols_dx = np.full((img_height, num_pix_added_dx), -1024)
                if start_x < 0:
                    lungs = image[s][start_y:end_y+1 , 0:end_x+1] 
                else:
                    lungs = image[s][start_y:img_height , start_x:img_width] 
                central_part_temp = np.append(additive_cols_sx,lungs, axis=1)
                central_part = np.append(central_part_temp,additive_cols_dx, axis=1)
                
                ### additive rows ###
                
                additive_rows_sx = np.full((num_pix_added_sx, img_width+num_pix_added_sx+num_pix_added_dx), -1024)
                additive_rows_dx = np.full((num_pix_added_dx, img_width+num_pix_added_sx+num_pix_added_dx), -1024)
                upper_part = np.append(additive_rows_sx, central_part, axis=0)
                crop_img[idx] = np.append(upper_part, additive_rows_dx, axis=0)
            else:
                crop_img[idx] = image[s][start_y:end_y+1 , start_x:end_x+1] 
    
    crop_img_padd = np.full((ref_depth,512,512), -1024)
    
    for i, img in enumerate(crop_img):
        
        outer_shape = (512,512)
        inner_shape = img.shape
        outer_array = np.full(outer_shape,-1024)
        inner_array = img
        
        start_index = tuple((np.array(outer_shape) - np.array(inner_shape)) // 2)
        end_index = tuple(start_index + np.array(inner_shape))

        outer_array[start_index[0]:end_index[0], start_index[1]:end_index[1]] = inner_array
        
        crop_img_padd[i] = outer_array
    
    return crop_img_padd


def cut_image(image, coords, ref_dims, scale_factor):
    img_width = image.shape[0]
    img_height = image.shape[1]

    ref_width = int(int(ref_dims['x']) * ref_dims['scale_factor_x'])
    ref_height = int(int(ref_dims['y']) * ref_dims['scale_factor_y'])

    # we must have square (width and height must be the same)
    if ref_width > ref_height:
        ref_height = ref_width
    elif ref_height > ref_width:
        ref_width = ref_height

    ### STARTING AND ENDING POINT OF X-AXIS (LENGTH OF THE LUNG) ###

    start_x = int(int(coords[0]) * scale_factor[0])
    end_x = int(int(coords[2]) * scale_factor[0])

    dim_x = (end_x - start_x) + 1

    if dim_x != ref_width:
        diff_width = ref_width - dim_x
        # print("diff_width:", diff_width)
        half_diff_width = diff_width // 2
        # print("half_diff_width:", half_diff_width)
        int_or_not_w = diff_width % 2
        if int_or_not_w == 0:
            new_start_x = start_x - half_diff_width
            if new_start_x < 0:
                start_x = 0
                end_x = end_x + half_diff_width + abs(new_start_x)
            else:
                new_end_x = end_x + half_diff_width
                if new_end_x >= img_width:
                    surplus_x = new_end_x - (img_width - 1)
                    start_x = new_start_x - surplus_x
                    end_x = img_width - 1
                else:
                    start_x = new_start_x
                    end_x = new_end_x
        else:
            new_start_x = start_x - (half_diff_width + 1)
            if new_start_x < 0:
                start_x = 0
                end_x = end_x + half_diff_width + abs(new_start_x)
            else:
                new_end_x = end_x + half_diff_width
                if new_end_x >= img_width:
                    surplus_x = new_end_x - (img_width - 1)
                    start_x = new_start_x - surplus_x
                    end_x = img_width - 1
                else:
                    start_x = new_start_x
                    end_x = new_end_x

    ### STARTING AND ENDING POJNT OF Y-AXIS (HEIGHT OF THE LUNG) ###

    start_y = int(int(coords[1]) * scale_factor[1])
    end_y = int(int(coords[3]) * scale_factor[1])

    dim_y = (end_y - start_y) + 1

    if dim_y != ref_height:
        diff_heights = ref_height - dim_y
        half_diff_heights = diff_heights // 2
        int_or_not_h = diff_heights % 2
        if int_or_not_h == 0:
            new_start_y = start_y - half_diff_heights
            if new_start_y < 0:
                start_y = 0
                end_y = end_y + half_diff_heights + abs(new_start_y)
            else:
                new_end_y = end_y + half_diff_heights
                if new_end_y >= img_height:
                    surplus_y = new_end_y - (img_height - 1)
                    start_y = new_start_y - surplus_y
                    end_y = img_height - 1
                else:
                    start_y = new_start_y
                    end_y = new_end_y
        else:
            new_start_y = start_y - (half_diff_heights + 1)
            if new_start_y < 0:
                start_y = 0
                end_y = end_y + half_diff_heights + abs(new_start_y)
            else:
                new_end_y = end_y + half_diff_heights
                if new_end_y >= img_height:
                    surplus_y = new_end_y - (img_height - 1)
                    start_y = new_start_y - surplus_y
                    end_y = img_height - 1
                else:
                    start_y = new_start_y
                    end_y = new_end_y

    ### CROP IMAGE ###
    sub_array = image[start_x:end_x, start_y:end_y]
    """plt.imshow(sub_array, cmap=plt.cm.gray)
    plt.show()"""
    crop_img = np.full((ref_height, ref_width), -1024)
    insert_start_row = (crop_img.shape[0] - sub_array.shape[0]) // 2
    insert_start_col = (crop_img.shape[1] - sub_array.shape[1]) // 2

    # Inserire il sub_array nell'array grande
    crop_img[insert_start_row:insert_start_row + sub_array.shape[0],
             insert_start_col:insert_start_col + sub_array.shape[1]] = sub_array
    """plt.imshow(crop_img, cmap=plt.cm.gray)
    plt.show()"""

    """crop_img = np.full((ref_height, ref_width), -1024)

    padding = False
    num_pix_added_sx = 0
    num_pix_added_dx = 0

    if end_x > img_width - 1:
        print("sono nella condizione end_x > img_width")
        padding = True
        difference = end_x - (img_width - 1)
        res = difference % 2
        if res == 0:
            num_pix_added_sx = difference / 2
            num_pix_added_dx = difference / 2
        else:
            num_pix_added_sx = difference // 2
            num_pix_added_dx = (difference // 2) + 1
    elif start_x < 0:
        print("sono nella condizione start_x < 0")
        padding = True
        difference = abs(start_x)
        res = difference % 2
        if res == 0:
            num_pix_added_sx = difference / 2
            num_pix_added_dx = difference / 2
        else:
            num_pix_added_sx = difference // 2
            num_pix_added_dx = (difference // 2) + 1

    num_pix_added_dx = int(num_pix_added_dx)
    num_pix_added_sx = int(num_pix_added_sx)

    if padding == True:

        ### additive columns ###

        additive_cols_sx = np.full((img_height, num_pix_added_sx), -1024)
        additive_cols_dx = np.full((img_height, num_pix_added_dx), -1024)
        if start_x < 0:
            print("sono nella condizione start_x < 0")
            lungs = image[start_y:end_y + 1, 0:end_x + 1]
        else:
            print("sono nella condizione start_x > 0")
            lungs = image[start_y:img_height, start_x:img_width]

        central_part_temp = np.append(additive_cols_sx, lungs, axis=1)
        central_part = np.append(central_part_temp, additive_cols_dx, axis=1)

        ### additive rows ###

        additive_rows_sx = np.full((num_pix_added_sx, img_width + num_pix_added_sx + num_pix_added_dx), -1024)
        additive_rows_dx = np.full((num_pix_added_dx, img_width + num_pix_added_sx + num_pix_added_dx), -1024)
        upper_part = np.append(additive_rows_sx, central_part, axis=0)
        crop_img = np.append(upper_part, additive_rows_dx, axis=0)
    else:
        crop_img = image[start_y:end_y + 1, start_x:end_x + 1]"""

    return crop_img