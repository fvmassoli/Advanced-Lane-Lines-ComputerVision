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
        self.warped_image = None

    def warp_image(self, ):
        img_size = (self.img.shape[1], self.img.shape[0])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        warped_imge = cv2.warpPerspective(self.img, self.M, img_size, flags=cv2.INTER_LINEAR)
        self.histogram = np.sum(warped_imge[int(warped_imge.shape[0] / 2):, :], axis=0)
        self.warped_image = warped_imge
        return warped_imge

    def draw_images(self, ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        ax1.imshow(self.img, cmap='gray')
        rect = patches.Polygon(self.src, closed=True, fill=False, edgecolor='y')
        ax1.add_patch(rect)
        ax2.imshow(self.warped_image, cmap='gray')
        plt.show()

    def plot_histogram(self, ):
        histogram = np.sum(self.warped_image[int(self.warped_image.shape[0] / 2):, :], axis=0)
        plt.plot(histogram)
