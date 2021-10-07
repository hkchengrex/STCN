import cv2
import numpy as np

import torch
from dataset.range_transform import inv_im_trans
from collections import defaultdict

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_seg(x):
    x = detach_to_cpu(x)
    x = inv_seg_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

# Predefined key <-> caption dict
key_captions = {
    'im': 'Image', 
    'gt': 'GT', 
}

"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""
def get_image_array(images, grid_shape, captions={}):
    h, w = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([w*cate_counts, h*(rows_counts+1), 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        dy = 40
        for i, line in enumerate(caption.split('\n')):
            cv2.putText(output_image, line, (10, col_cnt*w+100+i*dy),
                     font, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(col_cnt+0)*w:(col_cnt+1)*w,
                         (row_cnt+1)*h:(row_cnt+2)*h, :] = img
            
        col_cnt += 1

    return output_image

def base_transform(im, size):
        im = tensor_to_np_float(im)
        if len(im.shape) == 3:
            im = im.transpose((1, 2, 0))
        else:
            im = im[:, :, None]

        # Resize
        if im.shape[1] != size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

        return im.clip(0, 1)

def im_transform(im, size):
        return base_transform(inv_im_trans(detach_to_cpu(im)), size=size)

def mask_transform(mask, size):
    return base_transform(detach_to_cpu(mask), size=size)

def out_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)

def pool_pairs(images, size, so):
    req_images = defaultdict(list)

    b, s, _, _, _ = images['gt'].shape

    # limit number of images to save disk space
    b = max(2, b)

    GT_name = 'GT'
    for b_idx in range(b):
        GT_name += ' %s\n' % images['info']['name'][b_idx]

    for b_idx in range(b):
        for s_idx in range(s):
            req_images['RGB'].append(im_transform(images['rgb'][b_idx,s_idx], size))
            if s_idx == 0:
                req_images['Mask'].append(np.zeros((size[1], size[0], 3)))
                if not so:
                    req_images['Mask 2'].append(np.zeros((size[1], size[0], 3)))
            else:
                req_images['Mask'].append(mask_transform(images['mask_%d'%s_idx][b_idx], size))
                if not so:
                    req_images['Mask 2'].append(mask_transform(images['sec_mask_%d'%s_idx][b_idx], size))
            req_images[GT_name].append(mask_transform(images['gt'][b_idx,s_idx], size))
            if not so:
                req_images[GT_name + '_2'].append(mask_transform(images['sec_gt'][b_idx,s_idx], size))

    return get_image_array(req_images, size, key_captions)