from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np



def run(img1, img2):
    pass

"""
# Enhanced Correlation Coefficient (ECC)
    The ECC image alignment algorithm introduced in OpenCV 3 is based on a 2008 paper titled Parametric Image Alignment
    using Enhanced Correlation Coefficient Maximization by Georgios D. Evangelidis and Emmanouil Z. Psarakis.
    They propose using a new similarity measure called Enhanced Correlation Coefficient (ECC) for estimating
    the parameters of the motion model. There are two advantages of using their approach.

ref. http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

"""
def ecc(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert image to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Find size of images
    img1_size = img1.shape
    img2_size = img2.shape

    # Define the motion model
    wrap_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if wrap_mode == cv2.MOTION_HOMOGRAPHY:
        wrap_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        wrap_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations
    number_of_iterations = 5000
    # Specify the threshold of the icrement
    termination_eps = 1e-10

    # Define termation criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, wrap_matrix) = cv2.findTransformECC(img1_gray, img2_gray, wrap_matrix, wrap_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Show final results
    cv2.imshow("Image 1", im1)
    cv2.imshow("Image 2", im2)
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)
