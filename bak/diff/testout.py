import numpy as np
import pandas as pd
import os
import cv2
import dicom
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

# returns the id list of the given patient id
def return_id_path_list(get_id):
    patient_path = './data/1-100/CKUGH-'+get_id
    patient_dir = [f for f in listdir(patient_path) if isfile(join(patient_path, f))]
    return patient_path, patient_dir

# The trimming is done without the y trimming for alignment!
def trim(im):
    ret,thresh = cv2.threshold(im,0,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    max_index = np.argmax(areas)
    areas.remove(max(areas))

    cnt=contours[max_index]
    mask = np.zeros(im.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)

    im = cv2.bitwise_and(im, im, mask=mask)

    x,y,w,h = cv2.boundingRect(cnt)

    return im[:, x:x+w]

# displays gray-scale images
def imshow_gray(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'), cmap = 'gray')
    plt.gca().axis('off')

# Returns CC, MLO view aligned by width
def aligned_patient_dict(dcm_list, patient_path):
    patient_dict = {
            'LCC' : [],
            'LMLO' : [],
            'RCC' : [],
            'RMLO' : [],
    }

    for each_dcm in dcm_list :
        read_dicom = dicom.read_file(join(patient_path, each_dcm))
        key = read_dicom.ImageLaterality + read_dicom.ViewPosition
        pixels = read_dicom.pixel_array
        image = cv2.convertScaleAbs(pixels, alpha=(255.0/pixels.max(axis=1).max(axis=0)))
        ## low_value_indices = image < 50
        ## image[low_value_indices] = 0
        image = trim(image)
        if(read_dicom.ImageLaterality == 'R'):
            image = cv2.flip(image, 1)
        patient_dict[key] = image
        print image.shape

    # patient_dict['LCC'] = np.array(cv2.resize(patient_dict['LCC'], \
    #         (patient_dict['RCC'].shape[1], patient_dict['LCC'].shape[0]), \
    #         interpolation = cv2.INTER_AREA), dtype=np.int16)
    # patient_dict['LMLO'] = np.array(cv2.resize(patient_dict['LMLO'], \
    #         (patient_dict['RMLO'].shape[1], patient_dict['LMLO'].shape[0]), \
    #         interpolation = cv2.INTER_AREA), dtype=np.int16)

    if patient_dict['LCC'].shape[1] > patient_dict['RCC'].shape[1] :
        patient_dict['LCC'] = np.array(patient_dict['LCC'][:, 0:patient_dict['RCC'].shape[1]], dtype=np.int16)
    else :
        patient_dict['RCC'] = np.array(patient_dict['RCC'][:, 0:patient_dict['LCC'].shape[1]], dtype=np.int16)
    if patient_dict['LMLO'].shape[1] > patient_dict['RMLO'].shape[1] :
        patient_dict['LMLO'] = np.array(patient_dict['LMLO'][:, 0:patient_dict['RMLO'].shape[1]], dtype=np.int16)
    else :
        patient_dict['RMLO'] = np.array(patient_dict['RMLO'][:, 0:patient_dict['LMLO'].shape[1]], dtype=np.int16)
    return patient_dict

# Plot the 4 views in one screen
def plot_in_once(patient_dict):
    view_list = ['RCC', 'LCC', 'RMLO', 'LMLO']

    for i in range(len(view_list)):
        plt.subplot(2, 2, i+1)
        plt.title(view_list[i])
        imshow_gray(patient_dict[view_list[i]])
    plt.show()

# Drop the 4 views in current directory as img files
def drop_in_once(patient_dict, series=1):
    view_list = ['RCC', 'LCC', 'RMLO', 'LMLO']

    for view in view_list:
        cv2.imwrite('./'+view+str(series)+'.png', patient_dict[view])

# Drop contours of 4 views in current directory as img files
def drop_cont_in_once():
    view_list = ['RCC', 'LCC', 'RMLO', 'LMLO']

    thresh = 127
    maxValue = 255

    for view in view_list:
        src = cv2.imread('./'+view+'.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
        th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(dst, contours, -1, (255, 255, 255), 3)
        cv2.imwrite('./'+view+'_cont.png', dst)

# Get dotted map for inference
def get_dot_map_dict(patient_dict):
    dot_dict = {
            'LCC' : [],
            'LMLO' : [],
            'RCC' : [],
            'RMLO' : [],
    }

    view_list = ['RCC', 'LCC', 'RMLO', 'LMLO']

    for view in view_list:
        img = patient_dict[view]

        w, h = img.shape[1], img.shape[0]
        blank_array = [[0] * w for i in range(h)]
        blank_array = np.array(blank_array, dtype=np.int16)

        X_split = 11
        Y_split = 35

        X = w/X_split
        Y = h/Y_split

        for x in range(X_split):
            for y in range(Y_split):
                area = img[y*Y:(y+1)*Y, x*X:(x+1)*X]
                blank_array[y*Y:(y+1)*Y, x*X:(x+1)*X] = area.mean()

        dot_dict[view] = blank_array

    return dot_dict

# Get subtracted map for inference
def get_diff_dict(patient_dict):
    diff_dict = {
            'LCC' : [],
            'LMLO' : [],
            'RCC' : [],
            'RMLO' : [],
    }

    view_list = ['CC', 'MLO']

    def other_lat(lat):
        if(lat == 'L'):
            return 'R'
        else:
            return 'L'

    for view in view_list:
        for lat in ['R', 'L']:
            diff_dict[lat + view] = patient_dict[lat+view] - patient_dict[other_lat(lat)+view]
            not_subtracted_indices = diff_dict[lat+view] == patient_dict[lat+view]
            diff_dict[lat + view][not_subtracted_indices] = 0
            high_value_indices = diff_dict[lat+view] > 150
            diff_dict[lat + view][high_value_indices] = 0
            # diff_dict[lat+view] = (diff_dict[lat+view] - diff_dict[lat+view].mean())*1.5 + diff_dict[lat+view]

    return diff_dict


# thresholded map of 4 views
def get_thres_dict(patient_dict):
    thres_dict = {
            'LCC' : [],
            'LMLO' : [],
            'RCC' : [],
            'RMLO' : [],
    }

    view_list = ['LCC', 'LMLO', 'RCC', 'RMLO']

    for view in view_list:
        thres_dict[view] = patient_dict[view]
        low_value_indices = thres_dict[view] < 50
        thres_dict[view][low_value_indices] = 0

    return thres_dict

# stack up the gray-scale images
def stack_img(patient_dict, diff_dict): # (patient_dict, mass_dict, diff_dict)
    stacked_dict = {
            'LCC' : [],
            'LMLO' : [],
            'RCC' : [],
            'RMLO' : [],
    }
    view_list = ['LCC', 'LMLO', 'RCC', 'RMLO']
    for view in view_list:
        im_layers = []
        img_B = patient_dict[view]
        img_G = diff_dict[view] # substitute this with 'mass' detection
        img_R = diff_dict[view]
        im_layers.append(img_B)
        im_layers.append(img_G)
        im_layers.append(img_R)

        stacked_dict[view] = np.stack(im_layers, axis=-1)

    return stacked_dict

def outline_dict(patient_dict):
    outlined_dict = {
            'LCC' : [],
            'RCC' : [],
            'LMLO' : [],
            'RMLO' : [],
    }

    view_list = ['LCC', 'LMLO', 'RCC', 'RMLO']
    for view in view_list:
        blurred = cv2.GaussianBlur(patient_dict[view], (3,3), 0)

        # sobel X
        result = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        # Laplacian
        # result = cv2.Laplacian(blurred, cv2.CV_64F)
        outlined_dict[view] = result

    return outlined_dict

############### MAIN FUNCTION ##################
for i in range(1, 9):
    patient_path, dcm_list = return_id_path_list('000'+str(i))
    patient_dict = aligned_patient_dict(dcm_list, patient_path)

    print("Checking that each view dimension is equal .. ")
    print patient_dict['LMLO'].shape, patient_dict['RMLO'].shape
    print patient_dict['LCC'].shape, patient_dict['RCC'].shape

    # drop original images
    drop_in_once(patient_dict, 1)

    # drop outline images
    # outline = outline_dict(patient_dict)
    # drop_in_once(outline, 2)

    # drop stacked images
    patient_diff = get_diff_dict(patient_dict)
    result = stack_img(patient_dict, patient_diff)
    drop_in_once(result, i)
