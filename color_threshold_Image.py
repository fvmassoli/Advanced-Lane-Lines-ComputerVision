import numpy as np
import cv2

class ColorThresholdImage(object):
    def __init__(self, img, thresh_H=(15, 80), thresh_S=(80, 255)):
        self.img = img
        self.thresh_H = thresh_H
        self.thresh_S = thresh_S

    def get_color_threshold_combined(self, ):
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS).astype(np.float)
        H = hls[:, :, 0]
        L = hls[:, :, 1]
        S = hls[:, :, 2]
        h_channel = np.zeros_like(H)
        h_channel[(H > self.thresh_H[0]) & (H < self.thresh_H[1])] = 1
        s_channel = np.zeros_like(S)
        s_channel[(S > self.thresh_S[0]) & (S < self.thresh_S[1])] = 1
        color_combined = np.zeros_like(s_channel)
        color_combined[(s_channel == 1) & (h_channel == 1)] = 1
        return color_combined