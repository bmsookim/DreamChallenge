from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
#from matplotlib import pyplot as plt

methods = {
    'orb' : None,
    'sift': None,
    'surf': None
}

def extract_features(img1, img2, method='sift'):
    if methods[method] == None:
        if method == 'orb':
            methods['orb']  = cv2.ORB_create()
        elif method == 'sift':
            methods['sift'] = cv2.xfeatures2d.SIFT_create()
        elif method == 'surf':
            methods['surf'] = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError("Invalid method name: {0}".format(method))

    kp1, des1 = methods[method].detectAndCompute(img1, None)
    kp2, des2 = methods[method].detectAndCompute(img2, None)

    return kp1, kp2, des1, des2


"""
# Brute-Force Matcher
    Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with
    all other features in second set using some distance calculation. And the closest one is returned.
"""
def bruteforce(img1, img2, method='sift'):
    features = extract_features(img1, img2, method)
    (kp1, kp2, des1, des2) = features

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    return features, matches


"""
# FLANN based Matcher
    FLANN stands for Fast Library for Approximate Nearest Neighbors.
    It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and
    for high dimensional features.
    It works more faster than BFMatcher for large datasets.
    We will see the second example with FLANN based matcher.
"""
def flann(img1, img2, method='sift'):
    features = extract_features(img1, img2, method)
    (kp1, kp2, des1, des2) = features

    FLANN_INDEX_KDTREE  = 0
    index_params  = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowes'paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]

    return features, matches, matchesMask

"""
Draw Matching Result
"""
"""
def draw(img1, img2, features, matches, matchesMask=None):
    (kp1, kp2, des1, des2) = features

    if matchesMask is not None:
        draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    else:
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

    plt.imshow(img3,),plt.show()
"""
