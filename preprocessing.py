from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Preprocess import alignment
from Preprocess import matcher

def main():
    pass

def test(config):
    import cv2

    input_dir = config.get('data', 'input_dir')

    img1_path = '/'.join([input_dir, '1.png'])
    img2_path = '/'.join([input_dir, '2.png'])

    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    """
    features, matches =  matcher.bruteForce(img1, img2, method='orb')
    matcher.draw(img1, img2, features, matches)
    features, matches, matchesMask =  matcher.flann(img1, img2, method='sift')
    matcher.draw(img1, img2, features, matches, matchesMask)

    features, matches, matchesMask =  matcher.flann(img1, img2, method='surf')
    matcher.draw(img1, img2, features, matches, matchesMask)
    """
    alignment.ecc(img1_path, img2_path)

if __name__ == '__main__':
    import ConfigParser
    import argparse

    # program argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--test', action='store_true',
            help='Development Test Mode using dev_data')

    args = parser.parse_args()

    # program configuration
    config = ConfigParser.ConfigParser()

    if not args.test:
        config.read('config/preprocessing.ini')
        main(config)
    else:
        config.read('config/preprocessing.test.ini')
        test(config)
