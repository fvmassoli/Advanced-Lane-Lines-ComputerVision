import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class WarpImage(object):
    def __init__(self, img, src=np.float32([[240, 720], [580, 460], [735, 460], [1200, 720]]),
                 dst=np.float32([[300, 720], [300, 0], [1100, 0], [1100, 720]])):
        self.img = img
        self.src = src
        self.dst = dst

    def warp_image(self, ):
        img_size = (self.img.shape[1], self.img.shape[0])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped_imge = cv2.warpPerspective(self.img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped_imge

    def draw_images(self, warped_image):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        ax2.imshow(self.img, cmap='gray')
        rect = patches.Polygon(self.src, closed=True, fill=False, edgecolor='b')
        ax2.add_patch(rect)
        ax1.imshow(warped_image, cmap='gray')
        plt.show()

    def get_image_histo(self, warped_image):
        histogram = np.sum(warped_image[int(warped_image.shape[0] / 2):, :], axis=0)
        return histogram