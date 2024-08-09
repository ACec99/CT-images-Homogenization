import torch
import torch.nn as nn


def soft_binning_einsum(x, centers, sigma=0.5):
    x = x.unsqueeze(-1)

    centers = centers.to(x.device)

    exponent = -((x - centers) ** 2) / (2 * sigma ** 2)

    return torch.exp(exponent)


# theta 0°
def shift_image(x, d):
    result = torch.roll(x, shifts=-d, dims=3)
    result[:, :, :, -d:, :] = 0
    return result


# theta 45°
def shift_image45(x, d):
    result = torch.roll(x, shifts=(d, -d), dims=(2, 3))
    result[:, :, 0:d, :, :] = 0
    result[:, :, :, -d:, :] = 0
    return result


# theta 90°
def shift_image90(x, d):
    result = torch.roll(x, shifts=d, dims=2)
    result[:, :, 0:d, :, :] = 0
    return result


# theta 135°
def shift_image135(x, d):
    result = torch.roll(x, shifts=(d, d), dims=(2, 3))
    result[:, :, 0:d, :, :] = 0
    result[:, :, :, 0:d, :] = 0
    return result


def soft_binned_glcm_einsum_approx(x, d, theta, num_levels, min_r=-1, max_r=1):
    # Generate centers for soft binning
    centers = torch.linspace(min_r, max_r, num_levels)#.to(x.device)

    # Soft binning for all centers
    I_bins = soft_binning_einsum(x, centers)  # Shape: [batch, channels, height, width, num_levels]
    # I_bins = [soft_binning(x, center, 0.005).unsqueeze(-1) for center in centers]

    # Perform image shifting
    # I_s_bins = torch.cat([shift_image(I_bin, d) for I_bin in I_bins], dim=4)  # .permute(1, 4, 2, 3, 0)
    # I_bins = torch.cat(I_bins, dim=4)  # .permute(1, 4, 2, 3, 0)
    if theta == 0:
        I_s_bins = shift_image(I_bins, d)  # .permute(1, 4, 2, 3, 0)
    elif theta == 45:
        I_s_bins = shift_image45(I_bins, d)
    elif theta == 90:
        I_s_bins = shift_image90(I_bins, d)
    elif theta == 135:
        I_s_bins = shift_image135(I_bins, d)

    # Compute occurrences using einsum
    occurrences = torch.einsum('bchwj,bchwk->bcjk', I_bins, I_s_bins)

    # Make GLCM symmetrical
    glcm = occurrences + occurrences.permute(0, 1, 3, 2)

    # Normalize GLCM
    glcm_sum = glcm.sum(dim=(2, 3), keepdim=True)

    # Replace zeros in glcm_sum with a small value to avoid division by zero
    glcm_sum = torch.where(glcm_sum == 0, torch.ones_like(glcm_sum), glcm_sum)

    glcm /= glcm_sum
    return glcm


def compute_haralick_features(glcm):
    num_gray_levels = glcm.shape[2]

    # Create two 1D tensors representing the indices along each dimension
    I = torch.arange(0, num_gray_levels).unsqueeze(1).to(glcm.device)  # Column vector
    J = torch.arange(0, num_gray_levels).unsqueeze(0).to(glcm.device)  # Row vector

    weights = (I - J) ** 2
    weights = weights.reshape((1, 1, num_gray_levels, num_gray_levels)).to(glcm.device)
    contrast = torch.sum(glcm * weights, dim=(2, 3))

    return contrast


# texture grid extractor
def _extract_grid(image):
    haralick_grid = []
    for i in [1, 3, 5, 7]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=0, num_levels=256, min_r=-1, max_r=1)))
    for i in [1, 2, 4, 6]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=45, num_levels=256, min_r=-1, max_r=1)))
    for i in [1, 3, 5, 7]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=90, num_levels=256, min_r=-1, max_r=-1)))
    for i in [1, 2, 4, 6]:
        haralick_grid.append(compute_haralick_features(
            soft_binned_glcm_einsum_approx(image, d=i, theta=135, num_levels=256, min_r=-1, max_r=-1)))
    return torch.cat(haralick_grid, dim=0).view(image.size(0), 1, 4, 4)


def _texture_loss(fake_im, real_im, opt, grid_extractor, model=None):
    textures_real = grid_extractor(real_im)
    textures_fake = grid_extractor(fake_im)

    delta_grids = (torch.abs(textures_fake - textures_real)).view(opt.batch_size, 1, 4, 4)
    normalized_criterion = (delta_grids - delta_grids.min()) / (delta_grids.max() - delta_grids.min())
    out_attention, map, weight = model(normalized_criterion)

    loss_cycle_texture = torch.abs(torch.mean(torch.sum(out_attention, dim=(2, 3))))
    return loss_cycle_texture, map, weight
