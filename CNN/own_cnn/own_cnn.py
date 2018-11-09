"""
file own_cnn.py
author: Petri Lamminaho
Simple conv net.
one conv layer relu layer and (max) pooling layer
Code is based example and code from
https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
"""

import skimage.data
import numpy as np
import matplotlib
import sys

# loading image
img = skimage.data.camera()
# Converting the image into gray.
img = skimage.color.rgb2gray(img)


def __convolution(image, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    for r in np.uint16(np.arange(filter_size / 2.0,
                                       img.shape[0] - filter_size / 2.0 + 1)):
        for c in np.uint16(np.arange(filter_size / 2.0,
                                           img.shape[1] - filter_size / 2.0 + 1)):
            curr_region = img[r - np.uint16(np.floor(filter_size / 2.0)):r + np.uint16(
                np.ceil(filter_size / 2.0)),
                          c - np.uint16(np.floor(filter_size / 2.0)):c + np.uint16(
                              np.ceil(filter_size / 2.0))]

            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.


    final_result = result[np.uint16(filter_size / 2.0):result.shape[0] - np.uint16(filter_size / 2.0),
                   np.uint16(filter_size / 2.0):result.shape[1] - np.uint16(filter_size / 2.0)]
    return final_result



def convolution(image, conv_filter):
    print("Image size:",image.shape)
    print("Filter size:", conv_filter.shape)

    features_map = np.zeros((image.shape[0] - conv_filter.shape[1] + 1,
                             image.shape[1] - conv_filter.shape[1] + 1,
                             conv_filter.shape[0]))
    print(features_map.shape)

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print(" Using filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.
        if len(curr_filter.shape) > 2:
            conv_map = __convolution(img[:, :, 0], curr_filter[:, :, 0])  # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[
                -1]):  # Convolving each channel with the image and summing the results.
                conv_map = conv_map + __convolution(img[:, :, ch_num],
                                            curr_filter[:, :, ch_num])
        else:  # There is just a single channel in the filter.
            conv_map = __convolution(img, curr_filter)
        features_map[:, :, filter_num] = conv_map
    return features_map

def relu(feature_map):
    relu_output = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0, feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_output[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_output

def pooling(feature_map, size=2, stride=2 ):
    pool_out = np.zeros((np.uint16((feature_map.shape[0] - size + 1) / stride),
                            np.uint16((feature_map.shape[1] - size + 1) / stride),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0, feature_map.shape[0] - size - 1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1] - size - 1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r + size, c:c + size]])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out



# create con filters

layer1_filters = np.zeros((2, 3, 3))
layer1_filters[0,:, : ] = np.array([[[1, 1, 1],
                                   [-1, -1, -1],
                                   [0, 0, 0]]])
layer1_filters[1,:, : ] = np.array([[
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]]])

feature_map = convolution(img, layer1_filters)
relu_map = relu(feature_map)
map_relu_pool = pooling(relu_map)


# save image/result
fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
ax0.imshow(img).set_cmap("gray")
ax0.set_title("input image ")
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
matplotlib.pyplot.close(fig0)



fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax1[0, 0].imshow(feature_map[:, :, 0]).set_cmap("gray")
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])
ax1[0, 0].set_title("L1-Map1")

ax1[0, 1].imshow(feature_map[:, :, 1]).set_cmap("gray")
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])
ax1[0, 1].set_title("L1-Map2")


ax1[1, 0].imshow(relu_map[:, :, 0]).set_cmap("gray")
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])
ax1[1, 0].set_title("L1-Map1ReLU")

ax1[1, 1].imshow(relu_map[:, :, 1]).set_cmap("gray")
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])
ax1[1, 1].set_title("L1-Map2ReLU")


ax1[2, 0].imshow(map_relu_pool[:, :, 0]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 0].set_title("L1-Map1ReLUPool")

ax1[2, 1].imshow(map_relu_pool[:, :, 1]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 1].set_title("L1-Map2ReLUPool")



matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
matplotlib.pyplot.close(fig1)







