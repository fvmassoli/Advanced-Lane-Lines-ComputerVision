import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class WarpImage(object):
    def __init__(self, src=np.float32([[240, 720], [580, 460], [735, 460], [1200, 720]]),
                 dst=np.float32([[300, 720], [300, 0], [1100, 0], [1100, 720]])):
        self.src = src
        self.dst = dst

    def warp_image(self, img):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        warped_imge = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped_imge, Minv

    def draw_images(self, img, warped_image):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        ax1.imshow(img, cmap='gray')
        rect = patches.Polygon(self.src, closed=True, fill=False, edgecolor='y')
        ax1.add_patch(rect)
        ax2.imshow(warped_image, cmap='gray')
        plt.show()

    def plot_histogram(self, warped_image):
        histogram = np.sum(warped_image[int(warped_image.shape[0] / 2):, :], axis=0)
        plt.plot(histogram)
