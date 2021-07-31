import os
import cv2
import argparse
import numpy as np
import scipy.signal
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt


def config_parse():
    parser = argparse.ArgumentParser(description='Parameters Parser')
    parser.add_argument('--input_path', type=str, required=True, help='Specify the dir where the img seq are stored.')
    parser.add_argument('--p', type=int, default=4, help='Specify the exponent parameter.')
    parser.add_argument('--gsig', type=float, default=0.2, help='Specify the spread of the global Gaussian.')
    parser.add_argument('--lsig', type=float, default=0.5, help='Specify the spread of the local Gaussian.')
    parser.add_argument('--patch_size', type=int, default=21, help='Specify the patch size.')
    parser.add_argument('--step_size', type=int, default=2, help='Specify the stride size.')
    parser.add_argument('--exp_thres', type=float, default=0.01, help='Specify the exposure threshold to determine under- and over-exposed patches.')
    parser.add_argument('--cons_thres', type=float, default=0.1, help='Specify the IMF threshold.')
    parser.add_argument('--strt_thres', type=float, default=0.8, help='Specify the structure consistency threshold.')
    args = parser.parse_args()

    return args


def reorder_by_lum(seq_imgs):
    seq_imgs = np.double(seq_imgs)
    seq_sum_lum = np.sum(np.sum(np.sum(seq_imgs, axis=0, keepdims=True), axis=1, keepdims=True), axis=2, keepdims=True).squeeze()
    lum_sorts = np.argsort(seq_sum_lum)
    imgs_copy = np.copy(seq_imgs)
    for i in range(len(lum_sorts)):
        seq_imgs[:, :, :, i] = imgs_copy[:, :, :, lum_sorts[i]]

    return seq_imgs


def down_sample(seq_imgs, max_size=None):
    if max_size == None:
        max_size = 512

    seq_imgs = np.double(seq_imgs)
    [img_height, img_width] = seq_imgs.shape[:2]

    if img_height >= img_width and img_height > max_size:
        sample_factor = img_height / max_size
        down_spl_imgs = np.zeros((max_size, int(np.floor(img_width / sample_factor)), seq_imgs.shape[-2], seq_imgs.shape[-1]))
        for i in range(seq_imgs.shape[-1]):
            down_spl_imgs[:, :, :, i] = cv2.resize(seq_imgs[:, :, :, i], (int(np.floor(img_width / sample_factor)), max_size), cv2.INTER_LINEAR)
    elif img_height < img_width and img_width > max_size:
        sample_factor = img_width / max_size
        down_spl_imgs = np.zeros((int(np.floor(img_height / sample_factor)), max_size, seq_imgs.shape[2], seq_imgs.shape[-1]))
        for i in range(seq_imgs.shape[-1]):
            down_spl_imgs[:, :, :, i] = cv2.resize(seq_imgs[:, :, :, i], (max_size, int(np.floor(img_height / sample_factor))), cv2.INTER_LINEAR)
    else:
        down_spl_imgs = seq_imgs

    return down_spl_imgs


def select_ref_idx(seq_imgs, win_size=None, expos_thres=None):
    if win_size == None:
        win_size = 3

    if expos_thres == None:
        expos_thres = 0.01

    seq_imgs = np.double(seq_imgs)
    seq_imgs = reorder_by_lum(seq_imgs)
    [_, _, size_3, size_4] = seq_imgs.shape

    if size_4 == 3:
        ref_idx = 1
    else:
        window = np.ones((win_size, win_size, 3))
        window = window / window.sum()
        positive = np.zeros((size_4, 1))
        for i in range(size_4):
            conved_img = scipy.signal.convolve(seq_imgs[:, :, :, i], window, 'valid')
            positive[i] = np.sum(np.sum((conved_img < expos_thres) | (conved_img > 1 - expos_thres)))
        ref_idx = np.argmin(positive)

    return ref_idx


def imf_consistency(mean_intens, ref_img_idx, consistency_thres):
    imf_map = np.zeros_like(mean_intens)
    imf_map[:, :, ref_img_idx] = np.ones(mean_intens.shape[:2])

    ref_mean_intens = mean_intens[:, :, ref_img_idx]
    for i in range(mean_intens.shape[-1]):
        if i != ref_img_idx:
            temp_mean_intens = exposure.match_histograms(mean_intens[:, :, i], ref_mean_intens)
            diff = np.abs(temp_mean_intens - ref_mean_intens)
            imf_map[:, :, i] = diff <= consistency_thres

    return imf_map


def spd_mef(args, seq_imgs):
    exp_param = args.p
    glb_gauss = args.gsig
    lcl_gauss = args.lsig
    patch_size = args.patch_size
    step_size = args.step_size
    exp_thres = args.exp_thres
    cons_thres = args.cons_thres
    strt_thres = args.strt_thres

    C = 0.03 ** 2 / 2  # From Structural Similarity (MEF-SSIM)

    window = np.ones((patch_size, patch_size))
    window_3d = np.repeat(np.expand_dims(window, axis=2), 3, axis=2)
    window = window / window.sum()
    window_3d = window_3d / window_3d.sum()

    seq_imgs = np.double(seq_imgs)
    [size_1, size_2, size_3, size_4] = seq_imgs.shape
    x_idx_max = size_1 - patch_size + 1
    y_idx_max = size_2 - patch_size + 1

    ref_img_idx = select_ref_idx(seq_imgs)

    # Genarating Pseudo Exposures
    exp_img_num = 2 * size_4 - 1
    seq_exp_imgs = np.zeros((size_1, size_2, size_3, exp_img_num))
    seq_exp_imgs[:, :, :, :3] = seq_imgs

    count = 0
    for i in range(size_4):
        if i != ref_img_idx:
            exp_img_tmp = exposure.match_histograms(seq_exp_imgs[:, :, :, ref_img_idx], seq_exp_imgs[:, :, :, i])
            exp_img_tmp = np.maximum(np.minimum(exp_img_tmp, 1), 0)
            seq_exp_imgs[:, :, :, count + size_4] = exp_img_tmp
            count += 1

    # Computing Statistics
    glb_mean_intens = np.zeros((x_idx_max, y_idx_max, exp_img_num))  # Global Mean Intensity
    for i in range(exp_img_num):
        exp_img = seq_exp_imgs[:, :, :, i]
        glb_mean_intens[:, :, i] = np.ones((x_idx_max, y_idx_max)) * exp_img.mean()

    temp = np.zeros((x_idx_max, y_idx_max, size_3))
    lcl_mean_intens = np.zeros((x_idx_max, y_idx_max, exp_img_num))  # Local Mean Intensity
    lcl_intens_square = np.zeros((x_idx_max, y_idx_max, exp_img_num))

    for i in range(exp_img_num):
        for j in range(size_3):
            temp[:, :, j] = scipy.signal.correlate2d(seq_exp_imgs[:, :, j, i], window, 'valid')
        lcl_mean_intens[:, :, i] = temp.mean(axis=2)
        lcl_intens_square[:, :, i] = lcl_mean_intens[:, :, i] ** 2

    sig_strg_square = np.zeros((x_idx_max, y_idx_max, exp_img_num))  # Signal Strength from Variance
    for i in range(exp_img_num):
        for j in range(size_3):
            temp[:, :, j] = scipy.signal.correlate2d(seq_exp_imgs[:, :, j, i] ** 2, window, 'valid') - lcl_intens_square[:, :, i]
        sig_strg_square[:, :, i] = temp.mean(axis=2)
    sig_strength = np.sqrt(np.maximum(sig_strg_square, 0))
    sig_strength = sig_strength * np.sqrt(patch_size ** 2 * size_3) + 0.001  # Signal Strength

    # Computing Structural Consistency Map
    stru_consist_map = np.zeros((x_idx_max, y_idx_max, size_4, size_4))
    for i in range(size_4):
        for j in range(i+1, size_4):
            cross_intens = lcl_mean_intens[:, :, i] * lcl_mean_intens[:, :, j]
            cross_strg = scipy.signal.convolve(seq_exp_imgs[:, :, :, i] * seq_exp_imgs[:, :, :, j], window_3d, 'valid').squeeze() - cross_intens
            stru_consist_map[:, :, i, j] = (cross_strg + C) / (sig_strength[:, :, i] * sig_strength[:, :, j] + C)  # The third term in MEF-SSIM
    stru_consist_map = np.maximum(stru_consist_map, 0)

    stru_ref_map = stru_consist_map[:, :, ref_img_idx, :].squeeze() + stru_consist_map[:, :, :, ref_img_idx]
    stru_ref_map[:, :, ref_img_idx] = np.ones((x_idx_max, y_idx_max))  # Add Reference
    stru_ref_map[stru_ref_map <= strt_thres] = 0
    stru_ref_map[stru_ref_map > strt_thres] = 1
    intens_idx_map = (lcl_mean_intens[:, :, ref_img_idx] < exp_thres) | (lcl_mean_intens[:, :, ref_img_idx] > 1 - exp_thres)
    intens_idx_map = np.repeat(np.expand_dims(intens_idx_map, axis=2), size_4, axis=2)
    stru_ref_map[intens_idx_map] = 1
    struct_elem = np.zeros((41, 41))
    n = 11
    for i in range(struct_elem.shape[0]):
        n = n - 1 if np.abs(i - int(struct_elem.shape[1] / 2)) > 10 else 0
        for j in range(np.abs(n), struct_elem.shape[1] - np.abs(n)):
            struct_elem[i, j] = 1
    for i in range(size_4):
        stru_ref_map[:, :, i] = cv2.morphologyEx(stru_ref_map[:, :, i], cv2.MORPH_OPEN, struct_elem.astype(np.uint8))

    imf_ref_map = imf_consistency(lcl_mean_intens[:, :, :size_4], ref_img_idx, cons_thres)

    ref_map = stru_ref_map * imf_ref_map

    exp_ref_map = np.zeros((x_idx_max, y_idx_max, 2 * size_4 - 1))
    exp_ref_map[:, :, :size_4] = ref_map

    count = 0
    for i in range(size_4):
        if i != ref_img_idx:
            exp_ref_map[:, :, count + size_4] = 1 - exp_ref_map[:, :, i]
            count += 1

    # Computing Weighing Map
    mean_intens_map = np.exp(-0.5 * ((glb_mean_intens - 0.5) ** 2 / glb_gauss ** 2 + (lcl_mean_intens - 0.5) ** 2 / lcl_gauss ** 2))  # Mean Intensity Weighing Map
    mean_intens_map = mean_intens_map * exp_ref_map
    normalizer = np.sum(mean_intens_map, axis=2)
    mean_intens_map = mean_intens_map / np.repeat(np.expand_dims(normalizer, axis=2), exp_img_num, axis=2)

    stru_consist_map = sig_strength ** exp_param  # Signal Structure Weighting Map
    stru_consist_map = stru_consist_map * exp_ref_map + 0.001
    normalizer = np.sum(stru_consist_map, axis=2)
    stru_consist_map = stru_consist_map / np.repeat(np.expand_dims(normalizer, axis=2), exp_img_num, axis=2)

    max_exp = sig_strength * exp_ref_map  # Desired Signal Strength
    max_exp = np.max(max_exp, axis=2)

    # Computing Index Matrix for Main Loop
    idx_matrix = np.zeros((x_idx_max, y_idx_max, size_4))
    idx_matrix[:, :, ref_img_idx] = ref_img_idx

    for i in range(size_4):
        if i < ref_img_idx:
            idx_matrix[:, :, i] = exp_ref_map[:, :, i] * i + exp_ref_map[:, :, i + size_4] * (i + size_4)
        elif i > ref_img_idx:
            idx_matrix[:, :, i] = exp_ref_map[:, :, i] * i + exp_ref_map[:, :, i + size_4 - 1] * (i + size_4 - 1)

    # Main Loop for SPD-MEF
    final_img = np.zeros((size_1, size_2, size_3))
    count_map = np.zeros((size_1, size_2, size_3))
    count_window = np.ones((patch_size, patch_size, size_3))
    x_idx_tmp = [x for x in range(x_idx_max)]
    x_idx = x_idx_tmp[:x_idx_max:step_size]
    x_idx.append(x_idx_tmp[x_idx[-1] + 1:x_idx_max][0])
    y_idx_tmp = [y for y in range(y_idx_max)]
    y_idx = y_idx_tmp[:y_idx_max:step_size]
    y_idx.append(y_idx_tmp[y_idx[-1] + 1:y_idx_max][0])

    offset = patch_size
    for row in range(len(x_idx)):
        for col in range(len(y_idx)):
            i = x_idx[row]
            j = y_idx[col]
            blocks = seq_exp_imgs[i:i + offset, j:j + offset, :, list(idx_matrix[i, j, :].astype(np.uint8))]
            r_block = np.zeros((patch_size, patch_size, size_3))
            for k in range(size_4):
                r_block = r_block + stru_consist_map[i, j, k] * (blocks[:, :, :, k] - lcl_mean_intens[i, j, k]) / sig_strength[i, j, k]
            if np.linalg.norm(r_block.flatten()) > 0:
                r_block = r_block / np.linalg.norm(r_block.flatten()) * max_exp[i, j]
            r_block = r_block + np.sum(mean_intens_map[i, j, :] * lcl_mean_intens[i, j, :])
            final_img[i:i + offset, j:j + offset, :] = final_img[i:i + offset, j:j + offset, :] + r_block
            count_map[i:i + offset, j:j + offset, :] = count_map[i:i + offset, j:j + offset, :] + count_window

    final_img = final_img / count_map
    final_img = np.maximum(np.minimum(final_img, 1), 0)

    return final_img


def main():
    args = config_parse()
    seq_img_path = args.input_path
    exp_img = np.array(Image.open(os.path.join(seq_img_path, os.listdir(seq_img_path)[0])))
    seq_img_RGB = np.zeros((exp_img.shape[0], exp_img.shape[1], exp_img.shape[2], len(os.listdir(seq_img_path))))
    for i in range(len(os.listdir(seq_img_path))):
        seq_img_RGB[:, :, :, i] = np.double(
            np.array(Image.open(os.path.join(seq_img_path, os.listdir(seq_img_path)[i])))) / 255.0

    seq_img_RGB = reorder_by_lum(seq_img_RGB)
    seq_img_RGB = down_sample(seq_img_RGB, 1024)
    enhanced_img = spd_mef(args, seq_img_RGB)

    plt.imsave('fused_img.png', enhanced_img)
    plt.title('Fused IMG')
    plt.imshow(enhanced_img)
    plt.show()


if __name__ == "__main__":
    main()







