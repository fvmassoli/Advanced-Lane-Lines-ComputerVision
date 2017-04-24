import numpy as np
import cv2
import matplotlib.pyplot as plt


class ColorThreshold(object):
    def __init__(self, thresh_H=(15, 80), thresh_S=(80, 255)):
        self.thresh_H = thresh_H
        self.thresh_S = thresh_S
        self.img = None

    def _get_hls_image(self, ):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS).astype(np.float)

    def _get_s_binary(self, ):
        s_channel = np.zeros_like(self.S)
        s_channel[(self.S > self.thresh_S[0]) & (self.S < self.thresh_S[1])] = 1
        return s_channel

    def _get_h_binary(self, ):
        h_channel = np.zeros_like(self.H)
        h_channel[(self.H > self.thresh_H[0]) & (self.H < self.thresh_H[1])] = 1
        return h_channel

    def get_color_threshold_combined(self, img):
        self.img = img
        hls_image = self._get_hls_image()
        self.H = hls_image[:, :, 0]
        self.S = hls_image[:, :, 1]
        self.L = hls_image[:, :, 2]
        s_channel = self._get_s_binary()
        h_channel = self._get_h_binary()
        color_combined = np.zeros_like(s_channel)
        color_combined[(s_channel == 1) & (h_channel == 1)] = 1
        return color_combined

    def draw_binaries(self, ):
        h_binary = self._get_h_binary()
        s_binary = self._get_s_binary()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        ax1.set_title('H channel')
        ax1.imshow(h_binary, cmap='gray')
        ax2.set_title('S channel')
        ax2.imshow(s_binary, cmap='gray')