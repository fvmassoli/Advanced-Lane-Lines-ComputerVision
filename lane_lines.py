import numpy as np
import matplotlib.pyplot as plt
import cv2


class LaneLines():
    def __init__(self, nwindows=9, margin=100, minpix=50):
        self.nonzero = None
        self.nonzeroy = None
        self.nonzerox = None
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.img_shape = (1280, 720)

    def lane_lines_full_search(self, img, binary_warped):
        left_lane_inds = []
        right_lane_inds = []
        self.binary_warped = binary_warped
        self.left_lane_inds = []
        self.right_lane_inds = []
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])

        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / self.nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = \
            ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low) &
             (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = \
            ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) &
             (self.nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(self.left_lane_inds)
        right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx = self.nonzerox[left_lane_inds]
        self.lefty = self.nonzeroy[left_lane_inds]
        self.rightx = self.nonzerox[right_lane_inds]
        self.righty = self.nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        self.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        self.fit_leftx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.fit_rightx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

    def draw_lane_lines(self, binary_warped):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        self.left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        #         out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        #         out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(self.left_fitx, ploty, color='green')
        plt.plot(self.right_fitx, ploty, color='green')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    def plot(self, plot_img=True):
        out_img = np.dstack(
            (self.binary_warped, self.binary_warped, self.binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        out_img[self.nonzeroy[self.left_lane_inds],
                self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds],
                self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        left_line_window1 = np.array(
            [np.transpose(np.vstack([self.fit_leftx - self.margin, self.fity]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([self.fit_leftx + self.margin, self.fity])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([self.fit_rightx - self.margin, self.fity]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([self.fit_rightx + self.margin, self.fity])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        if plot_img:
            plt.imshow(result)
            plt.plot(self.fit_leftx, self.fity, color='yellow', linewidth=5.0)
            plt.plot(self.fit_rightx, self.fity, color='yellow', linewidth=5.0)
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
        else:
            return result

    def calculate_curvature(self, real=False):
        # Around middle of the image, I think it will give use better
        # results since we can get better value from perspective
        # transform
        y_eval = 360
        ploty = np.linspace(0, 719, num=720)
        quadratic_coeff = 3e-4  # arbitrary quadratic coefficient

        if real:
            ym_per_pix = 30 / 720
            xm_per_pix = 3.7 / 700

            left_fit_cr = np.polyfit(self.lefty * ym_per_pix,
                                     self.leftx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(self.righty * ym_per_pix,
                                      self.rightx * xm_per_pix, 2)
        else:
            left_fit_cr = self.left_fit
            right_fit_cr = self.right_fit
            # right_curverad = self.right_fit
            ym_per_pix = 1.0
            xm_per_pix = 1.0

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                               left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                                right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def translate_to_real_world_image(self, image, binary_warped, Minv):

        warped = binary_warped
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(
            np.vstack([self.fit_leftx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(
            np.vstack([self.fit_rightx, self.ploty])))])

        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, Minv, self.img_shape)
        left_curvature, right_curvature = self.calculate_curvature(real=True)
        cv2.putText(image, "Curvature: " + str(int(left_curvature)) + '(m)',
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)
        cv2.putText(image, str(self.calculate_relative_position()),
                    (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

    def calculate_relative_position(self, xm_per_pix=3.7 / 700):
        middle = 360
        left = self.calculate_val(self.left_fit, middle)
        right = self.calculate_val(self.right_fit, middle)
        car_middle_pixel = int((left + right) / 2)
        screen_off_center = middle - car_middle_pixel
        return xm_per_pix * screen_off_center

    def calculate_val(self, fit, val):
        return fit[0] * val ** 2 + fit[1] * val + fit[2]





