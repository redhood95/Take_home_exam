import argparse
import os
import itertools
import math
import sys

import numpy as np


sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2 as cv


# import timeit

from skimage.io import imsave

from utils import positive_integer, subarray, show_quiver


def main():
    # parse the command line arguments to attributes of 'args'

    parser.add_argument('--anchor-frame', dest='anchor_frame_path', required=True, type=str,
                        help='Path to the frame that will be predicted.')
    parser.add_argument('--target-frame', dest='target_frame_path', required=True, type=str,
                        help='Path to the frame that will be used to predict the anchor frame.')
    parser.add_argument('--frame-width', dest='frame_width', required=True, type=positive_integer,
                        help='Frame width.')
    parser.add_argument('--frame-height', dest='frame_height', required=True, type=positive_integer,
                        help='Frame height.')
    parser.add_argument('--norm', dest='norm', required=False, type=float, default=1,
                        help='Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.')
    parser.add_argument('--block-size', dest='block_size', required=True, type=positive_integer,
                        help='Size of the blocks the image will be cut into, in pixels.')
    parser.add_argument('--search-range', dest='search_range', required=True, type=positive_integer,
                        help="Range around the pixel where to search, in pixels.")
    parser.add_argument('--pixel-accuracy', dest='pixel_acc', type=positive_integer, default=1, required=False,
                        help="1: Integer-Pel Accuracy (no interpolation), "
                             "2: Half-Pel Integer Accuracy (Bilinear interpolation")
    args = parser.parse_args()

    # Pixel map of the frames in the [0,255] interval
    target_frm = np.fromfile(args.target_frame_path, dtype=np.uint8, count=args.frame_height * args.frame_width)
    anchor_frm = np.fromfile(args.anchor_frame_path, dtype=np.uint8, count=args.frame_height * args.frame_width)
    target_frm = np.reshape(target_frm, (args.frame_height, args.frame_width))
    anchor_frm = np.reshape(anchor_frm, (args.frame_height, args.frame_width))

    # store frames in PNG for our records
    os.system('mkdir -p frames_of_interest')
    imsave('frames_of_interest/target.png', target_frm)
    imsave('frames_of_interest/anchor.png', anchor_frm)

    ebma = EBMA_searcher(N=args.block_size,
                         R=args.search_range,
                         p=args.norm,
                         acc=args.pixel_acc)

    predicted_frm, motion_field = \
        ebma.run(anchor_frame=anchor_frm,
                 target_frame=target_frm)

    # store predicted frame
    imsave('frames_of_interest/predicted_anchor.png', predicted_frm)

    motion_field_x = motion_field[:, :, 0]
    motion_field_y = motion_field[:, :, 1]

    # show motion field
    show_quiver(motion_field_x, motion_field_y[::-1])

    # store error image
    error_image = abs(np.array(predicted_frm, dtype=float) - np.array(anchor_frm, dtype=float))
    error_image = np.array(error_image, dtype=np.uint8)
    imsave('frames_of_interest/error_image.png', error_image)

    # Peak Signal-to-Noise Ratio of the predicted image
    mse = (np.array(error_image, dtype=float) ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    print 'PSNR: %s dB' % psnr


class EBMA_searcher():
    """
    Estimates the motion between to frame images
     by running an Exhaustive search Block Matching Algorithm (EBMA).
    Minimizes the norm of the Displaced Frame Difference (DFD).
    """

    def __init__(self, N, R, p=1, acc=1):
        """
        :param N: Size of the blocks the image will be cut into, in pixels.
        :param R: Range around the pixel where to search, in pixels.
        :param p: Norm used for the DFD. p=1 => MAD, p=2 => MSE. Default: p=1.
        :param acc: 1: Integer-Pel Accuracy (no interpolation),
                    2: Half-Integer Accuracy (Bilinear interpolation)
        """

        self.N = N
        self.R = R
        self.p = p
        self.acc = acc

    def run(self, anchor_frame, target_frame):
        """
        Run!
        :param anchor_frame: Image that will be predicted.
        :param target_frame: Image that will be used to predict the target frame.
        :return: A tuple consisting of the predicted image and the motion field.
        """

        acc = self.acc
        height = anchor_frame.shape[0]
        width = anchor_frame.shape[1]
        N = self.N
        R = self.R
        p = self.p

        # interpolate original images if half-pel accuracy is selected
        if acc == 1:
            pass
        elif acc == 2:
            target_frame = cv.resize(target_frame, dsize=(width * 2, height * 2))
        else:
            raise ValueError('pixel accuracy should be 1 or 2. Got %s instead.' % acc)

        # predicted frame. anchor_frame is predicted from target_frame
        predicted_frame = np.empty((height, width), dtype=np.uint8)

        # motion field consisting in the displacement of each block in vertical and horizontal
        motion_field = np.empty((int(height / N), int(width / N), 2))

        # loop through every NxN block in the target image
        for (blk_row, blk_col) in itertools.product(xrange(0, height - (N - 1), N),
                                                    xrange(0, width - (N - 1), N)):

            # block whose match will be searched in the anchor frame
            blk = anchor_frame[blk_row:blk_row + N, blk_col:blk_col + N]

            # minimum norm of the DFD norm found so far
            dfd_n_min = np.infty

            # search which block in a surrounding RxR region minimizes the norm of the DFD. Blocks overlap.
            for (r_col, r_row) in itertools.product(range(-R, (R + N)),
                                                    range(-R, (R + N))):
                # candidate block upper left vertex and lower right vertex position as (row, col)
                up_l_candidate_blk = ((blk_row + r_row) * acc, (blk_col + r_col) * acc)
                low_r_candidate_blk = ((blk_row + r_row + N - 1) * acc, (blk_col + r_col + N - 1) * acc)

                # don't search outside the anchor frame. This lowers the computational cost
                if up_l_candidate_blk[0] < 0 or up_l_candidate_blk[1] < 0 or \
                                low_r_candidate_blk[0] > height * acc - 1 or low_r_candidate_blk[1] > width * acc - 1:
                    continue

                # the candidate block may fall outside the anchor frame
                candidate_blk = subarray(target_frame, up_l_candidate_blk, low_r_candidate_blk)[::acc, ::acc]
                assert candidate_blk.shape == (N, N)

                dfd = np.array(candidate_blk, dtype=np.float16) - np.array(blk, dtype=np.float16)

                candidate_dfd_norm = np.linalg.norm(dfd, ord=p)

                # a better matching block has been found. Save it and its displacement
                if candidate_dfd_norm < dfd_n_min:
                    dfd_n_min = candidate_dfd_norm
                    matching_blk = candidate_blk
                    dy = r_col
                    dx = r_row

            # construct the predicted image with the block that matches this block
            predicted_frame[blk_row:blk_row + N, blk_col:blk_col + N] = matching_blk

            print str((blk_row / N, blk_col / N)) + '--- Displacement: ' + str((dx, dy))

            # displacement of this block in each direction
            motion_field[blk_row / N, blk_col / N, 1] = dx
            motion_field[blk_row / N, blk_col / N, 0] = dy

        return predicted_frame, motion_field


##IMPLEMENTATION OF LOSS FUNCTION


### IMPLEMENTATION of sum of squared Difference
def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    for y in range(kernel_half, h - kernel_half):
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):

                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])
                        ssd += ssd_temp * ssd_temp

                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            depth[y, x] = best_offset * offset_adjust

    Image.fromarray(depth).save('depth.png')




#####Sum of absolute Difference

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img[10:330, 10:870]
img2 = img[20:340, 20:880]

start = time.clock()
d = cv2.absdiff(img1, img2)
s = d.sum()
t = time.clock() - start
print 'with absdiff ', t
print s

start = time.clock()
s = cv2.norm(img1, img2, cv2.NORM_L1)
t = time.clock() - start
print 'with norm L1 ',  t
print s



#### normalized-cross-correlation-to-measure-similarites
from skimage import io, feature
from scipy import ndimage
import numpy as np

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

im = io.imread('faces.jpg', as_grey=True)

im1 = im[16:263, 4:146]
sh_row, sh_col = im1.shape
im2 = im[16:263, 155:155+sh_col]

# Registration of the two images
translation = feature.register_translation(im1, im2, upsample_factor=10)[0]
im2_register = ndimage.shift(im2, translation)

d = 1

correlation = np.zeros_like(im1)

for i in range(d, sh_row - (d + 1)):
    for j in range(d, sh_col - (d + 1)):
        correlation[i, j] = correlation_coefficient(im1[i - d: i + d + 1,
                                                        j - d: j + d + 1],
                                                    im2[i - d: i + d + 1,
                                                        j - d: j + d + 1])

io.imshow(correlation, cmap='gray')
io.show()
