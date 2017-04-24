import numpy as np
import cv2
import matplotlib.pyplot as plt


class ColorThreshold(object):
    def __init__(self, images, thresh_H=(15, 80), thresh_S=(80, 255)):
        self.images = images
        self.thresh_H = thresh_H
        self.thresh_S = thresh_S
        self._loop_over_images()

    def _loop_over_images(self, ):
        self.hls_images = []
        self.H = []
        self.L = []
        self.S = []
        for img in self.images:
            hls_img = self._get_hls_image(img)
            self.hls_images.append(hls_img)
            self.H.append(hls_img[:, :, 0])
            self.L.append(hls_img[:, :, 1])
            self.S.append(hls_img[:, :, 2])

    def _get_hls_image(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    def _get_s_binaries(self, ):
        s_channels = []
        for i in range(len(self.hls_images)):
            s_channel = np.zeros_like(self.S[0])
            s_channel[(self.S[i] > self.thresh_S[0]) & (self.S[i] < self.thresh_S[1])] = 1
            s_channels.append(s_channel)
        return s_channels

    def _get_h_binaries(self, ):
        h_channels = []
        for i in range(len(self.hls_images)):
            h_channel = np.zeros_like(self.S[0])
            h_channel[(self.H[i] > self.thresh_H[0]) & (self.H[i] < self.thresh_H[1])] = 1
            h_channels.append(h_channel)
        return h_channels

    def get_color_threshold_combined(self, ):
        s_channels = self._get_s_binaries()
        h_channels = self._get_h_binaries()
        colors_combined = []
        color_combined = np.zeros_like(s_channels[0])
        for i in range(len(s_channels)):
            color_combined[(s_channels[i] == 1) & (h_channels[i] == 1)] = 1
            colors_combined.append(color_combined)
        return colors_combined

    def draw_binaries(self, ):
        h_binaries = self._get_h_binaries()
        s_binaries = self._get_s_binaries()
        for i in range(len(self.hls_images)):
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
            ax1.set_title('H channel')
            ax1.imshow(h_binaries[i], cmap='gray')
            ax2.set_title('S channel')
            ax2.imshow(s_binaries[i], cmap='gray')