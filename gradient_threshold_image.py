import numpy as np
import cv2

class GradientThresholdImage(object):
    def __init__(self, img, kernel=3, thres_x=(20, 200), thres_y=(45, 150), thres_mag=(35, 255),
                 thres_dir=(0.7, 1.3)):
        self.img = img
        self.kernel = kernel
        self.thres_x = thres_x
        self.thres_y = thres_y
        self.thres_mag = thres_mag
        self.thres_dir = thres_dir
        self.gray = self._gray_scale()

    def _abs_sobel_x_thresh(self, ):
        sobel = self._evaluate_absolute_sobel_operator(1, 0)
        sobel_scaled = self._scale(sobel)
        sobel_binary = np.zeros_like(sobel_scaled)
        sobel_binary[(sobel_scaled >= self.thres_x[0]) & (sobel_scaled <= self.thres_y[1])] = 1
        return sobel_binary

    def _abs_sobel_y_thresh(self, ):
        sobel = self._evaluate_absolute_sobel_operator(0, 1)
        sobel_scaled = self._scale(sobel)
        sobel_binary = np.zeros_like(sobel_scaled)
        sobel_binary[(sobel_scaled >= self.thres_y[0]) & (sobel_scaled <= self.thres_y[1])] = 1
        return sobel_binary

    def _mag_thresh(self, ):
        sobelx = self._evaluate_absolute_sobel_operator(1, 0)
        sobely = self._evaluate_absolute_sobel_operator(0, 1)
        grad_magnitue = np.sqrt(sobelx ** 2 + sobely ** 2)
        grad_scaled = self._scale(grad_magnitue)
        grad_binary = np.zeros_like(grad_scaled)
        grad_binary[(grad_scaled >= self.thres_mag[0]) & (grad_scaled <= self.thres_mag[1])] = 1
        return grad_binary

    def _dir_threshold(self, ):
        sobelx = self._evaluate_absolute_sobel_operator(1, 0)
        sobely = self._evaluate_absolute_sobel_operator(0, 1)
        dir_grad = np.arctan2(sobely, sobelx)
        dir_binary = np.zeros_like(dir_grad)
        dir_binary[(dir_grad >= self.thres_dir[0]) & (dir_grad <= self.thres_dir[1])] = 1
        return dir_binary

    #     def _blur_image(self, img):
    #         kernel_size=9
    #         blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    #         return blur

    def _gray_scale(self, ):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        return gray

    def _evaluate_absolute_sobel_operator(self, x, y):
        sobel = cv2.Sobel(self.gray, cv2.CV_64F, x, y, self.kernel)
        return np.abs(sobel)

    def _scale(self, sobel):
        return np.uint8(255. * sobel / np.max(sobel))

    def get_gradient_threshold_combined(self, ):
        gradx = self._abs_sobel_x_thresh()
        grady = self._abs_sobel_y_thresh()
        dir_binary = self._dir_threshold()
        mag_binary = self._mag_thresh()
        grad_combined = np.zeros_like(dir_binary)
        grad_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return grad_combined