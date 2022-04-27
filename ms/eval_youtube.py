import os
from os import path
from argparse import ArgumentParser
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import hickle as hkl
from PIL import Image

from model.eval_network import STCN
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from util.tensor_util import unpad
from inference_core_yv import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--yv_path', default='../YouTube')

parser.add_argument('--output_all', help=
"""
We will output all the frames if this is set to true.
Otherwise only a subset will be outputted, as determined by meta.json to save disk space.
For ensemble, all the sources must have this setting unified.
""", action='store_true')

parser.add_argument('--output')
parser.add_argument('--split', help='valid/test', default='valid')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--use_km', action='store_true', help='Employ kernelized memory')
# parser.add_argument('--save_np', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--res', default=480, type=int)
# parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', action='store_true')
args = parser.parse_args()

yv_path = args.yv_path
out_path = args.output
args.save_np = True
args.amp = True

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Load the json if we have to
if not args.output_all:
    with open(path.join(yv_path, args.split, 'meta.json')) as f:
        meta = json.load(f)['videos']

# Setup Dataset
test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split, res=args.res)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = args.top

if args.use_km:
    kmn_sigma = 7
else:
    kmn_sigma = None

prop_model = STCN().cuda().eval()
prop_model.load_state_dict(prop_saved)

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb']
        msk = data['gt'][0]
        info = data['info']
        name = info['name'][0]
        num_objects = len(info['labels'][0])
        gt_obj = info['gt_obj']
        size = info['size']

        # Load the required set of frames
        req_frames = None
        if not args.output_all:
            req_frames = []
            objects = meta[name]['objects']
            for key, value in objects.items():
                req_frames.extend(value['frames'])

            # Map the frame names to indices
            req_frames_names = set(req_frames)
            req_frames = []
            for fi in range(rgb.shape[1]):
                frame_name = info['frames'][fi][0][:-4]
                if frame_name in req_frames_names:
                    req_frames.append(fi)
            req_frames = sorted(req_frames)

        if args.flip:
            rgb = torch.flip(rgb, dims=[-1])
            msk = torch.flip(msk, dims=[-1])

        # Frames with labels, but they are not exhaustively labeled
        frames_with_gt = sorted(list(gt_obj.keys()))

        processor = InferenceCore(prop_model, rgb, num_objects=num_objects, top_k=top_k,
                                    mem_every=args.mem_every, include_last=args.include_last,
                                    req_frames=req_frames, kmn_sigma=kmn_sigma)
        # min_idx tells us the starting point of propagation
        # Propagating before there are labels is not useful
        min_idx = 99999
        for i, frame_idx in enumerate(frames_with_gt):
            min_idx = min(frame_idx, min_idx)
            # Note that there might be more than one label per frame
            obj_idx = gt_obj[frame_idx][0].tolist()
            # Map the possibly non-continuous labels into a continuous scheme
            obj_idx = [info['label_convert'][o].item() for o in obj_idx]

            # Append the background label
            with_bg_msk = torch.cat([
                1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
                msk[:,frame_idx],
            ], 0).cuda()

            # We perform propagation from the current frame to the next frame with label
            if i == len(frames_with_gt) - 1:
                processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
            else:
                processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

        # Do unpad -> upsample to original size (we made it 480p)
        if args.save_np:
            out_prob = torch.zeros((processor.t, num_objects+1, *size), dtype=torch.float32, device='cuda')

        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            if args.save_np:
                out_prob[ti] = prob[:,0]
            out_masks[ti] = torch.argmax(prob, dim=0)

        if args.flip:
            if args.save_np:
                out_prob = torch.flip(out_prob, dims=[-1])
            out_masks = torch.flip(out_masks, dims=[-1])

        if args.save_np:
            out_prob = ((out_prob.detach().cpu().numpy()*255)).astype(np.uint8)
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        # Remap the indices to the original domain
        idx_masks = np.zeros_like(out_masks)
        for i in range(1, num_objects+1):
            backward_idx = info['label_backward'][i].item()
            idx_masks[out_masks==i] = backward_idx

        if args.save_np:
            label_backward = [info['label_backward'][i].item() for i in range(1, num_objects+1)]
            frame_names = [info['frames'][fi][0] for fi in range(processor.t)]
            if not args.output_all:
                frame_names = [frame_names[fi] for fi in req_frames]
                out_prob = out_prob[req_frames]
            hkl.dump(out_prob, path.join(out_path, name+'.hkl'), mode='w', compression='lzf')
            hkl.dump(label_backward, path.join(out_path, name+'_backward.hkl'), mode='w')
            hkl.dump(frame_names, path.join(out_path, name+'_names.hkl'), mode='w')

        # Save the results
        this_out_path = path.join(out_path, 'Annotations', name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(idx_masks.shape[0]):
            if f >= min_idx:
                if args.output_all or (f in req_frames):
                    img_E = Image.fromarray(idx_masks[f])
                    img_E.putpalette(palette)
                    img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))

        del rgb
        del msk
        del processor
