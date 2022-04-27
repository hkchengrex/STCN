import os
from os import path
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import hickle as hkl
from PIL import Image

from progressbar import progressbar
from multiprocessing import Pool


def search_options(options, name):
    for option in options:
        if path.exists(path.join(option, name)):
            return path.join(option, name)
    else:
        return None

def process_vid(vid):
    backward_path = search_options(all_options, vid[:-4]+'_backward.hkl')
    if backward_path is not None:
        backward_mapping = hkl.load(backward_path)
    else:
        backward_mapping = None
    total_result = None

    for option in all_options:
        if not path.exists(path.join(option, vid)):
            continue
        print(option, vid)

        result = hkl.load(path.join(option, vid))

        if total_result is None:
            total_result = result.astype(np.float32)
        else:
            total_result += result.astype(np.float32)

    # argmax and to idx
    k = total_result.shape[1]
    total_result = np.argmax(total_result, axis=1)  

    # Remap the indices to the original domain
    if backward_mapping is not None:
        idx_masks = np.zeros_like(total_result, dtype=np.uint8)
        # zero always maps to zero
        backward_mapping = [0] + backward_mapping
        for i in range(k):
            idx_masks[total_result==i] = backward_mapping[i]
    else:
        idx_masks = total_result.astype(np.uint8)

    # Save the results
    names = hkl.load(search_options(all_options, vid[:-4]+'_names.hkl'))
    this_out_path = path.join(out_path, 'Annotations', vid[:-4])
    os.makedirs(this_out_path, exist_ok=True)
    for f in range(idx_masks.shape[0]):
        img_E = Image.fromarray(idx_masks[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(this_out_path, names[f].replace('.jpg','.png')))


if __name__ == '__main__':
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument('--yv', default='../YouTube')
    parser.add_argument('--list', nargs="+")
    parser.add_argument('--output')
    parser.add_argument('--num_proc', default=4, type=int)
    args = parser.parse_args()

    yv_path = args.yv
    out_path = args.output

    # Simple setup
    os.makedirs(out_path, exist_ok=True)
    palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

    all_options = args.list
    vid_count = defaultdict(int)

    for option in all_options:
        vid_in_here = sorted(os.listdir(option))
        vid_in_here = [v for v in vid_in_here if 'backward' not in v and 'names' not in v and '.hkl' in v]
        for vid in vid_in_here:
            vid_count[vid] += 1

    all_vid = []
    count_to_vid = defaultdict(int)
    for k, v in vid_count.items():
        count_to_vid[v] += 1
        all_vid.append(k)

    for k, v in count_to_vid.items():
        print('Videos with count %d: %d' % (k, v))

    all_vid = sorted(all_vid)
    print('Total number of videos: ', len(all_vid))

    pool = Pool(processes=args.num_proc)
    for _ in progressbar(pool.imap_unordered(process_vid, all_vid), max_value=len(all_vid)):
       pass
    # for v in progressbar(all_vid, max_value=len(all_vid)):
    #     print(v)
    #     process_vid(v)

    pool.close()
    pool.join()