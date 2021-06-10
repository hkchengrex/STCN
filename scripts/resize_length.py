import sys
import os
import cv2

from progressbar import progressbar

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# max_length = 500
min_length = 384

def process_fun():

    for f in progressbar(os.listdir(input_dir)):
        img = cv2.imread(os.path.join(input_dir, f))
        h, w, _ = img.shape

        # scale = max(h, w) / max_length
        scale = min(h, w) / min_length

        img = cv2.resize(img, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(f)), img)

if __name__ == '__main__':

    os.makedirs(output_dir, exist_ok=True)
    process_fun()

    print('All done.')