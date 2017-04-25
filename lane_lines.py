import numpy as np
import matplotlib.pyplot as plt
import cv2


class LaneLines():
    def __init__(self, nwindows=9, margin=100, minpix=50, max_nb_acquired_lanes=15):
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.img_shape = (1280, 720)
        self.full_search_mode = True
        # containers for the x and y positions of nonzero pixels of the image
        self.nonzero = None
        self.nonzeroy = None
        self.nonzerox = None
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.old_lanes = []
        self.max_nb_acquired_lanes = max_nb_acquired_lanes

    def search(self, binary_warped):
        self.current_image = binary_warped
        self.set_search_mode(binary_warped)
        if len(self.old_lanes) >= self.max_nb_acquired_lanes:
            self.old_lanes.pop(0)
        # self.old_lanes.append(copy.copy(self.lane))
        self.old_lanes.append(self)
        self.smooth_fit(binary_warped)

    def set_search_mode(self, binary_warp):
        '''
        Start a full search for lane lines or uses previously knowledge 
        '''
        if self.full_search_mode:
            self.start_full_search(binary_warp)
        else:
            self.fast_search(binary_warp)

    def start_full_search(self, binary_warped):
        self.left_fit = []
        self.right_lane_inds = []
        self.binary_warped = binary_warped
        self.nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])

        histogram = np.sum(self.binary_warped
                           [self.binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(self.binary_warped.shape[0] // self.nwindows)
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []
        self.left_lane_inds = []
        self.right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = self.binary_warped.shape[
                            0] - (window + 1) * window_height
            win_y_high = self.binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((self.nonzeroy >= win_y_low) &
                              (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) &
                              (self.nonzerox < win_xleft_high)).nonzero()[0]

            good_right_inds = ((self.nonzeroy >= win_y_low) &
                               (self.nonzeroy < win_y_high) &
                               (self.nonzerox >= win_xright_low) &
                               (self.nonzerox < win_xright_high)).nonzero()[0]

            # self.left_lane_inds = np.concatenate(good_left_inds)
            # self.right_lane_inds = np.concatenate(good_right_inds)
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(
                    np.mean(self.nonzerox[good_right_inds]))

        self.left_lane_inds = np.concatenate(left_lane_inds)
        self.right_lane_inds = np.concatenate(right_lane_inds)

        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        if len(self.leftx) < 25 or len(self.rightx) < 25:
            # self = self.old
            self.leftx = self.old_leftx
            self.lefty = self.old_lefty
            self.rightx = self.old_rightx
            self.righty = self.old_righty

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)
        self.fity = np.linspace(0, self.binary_warped.shape[0] - 1, self.binary_warped.shape[0])
        self.fit_leftx = self.left_fit[0] * self.fity ** 2 + self.left_fit[1] * self.fity + self.left_fit[2]
        self.fit_rightx = self.right_fit[0] * self.fity ** 2 + self.right_fit[1] * self.fity + self.right_fit[2]
        self.old_leftx = self.leftx
        self.old_rightx = self.rightx
        self.old_lefty = self.lefty
        self.old_righty = self.righty
        self.exhaustive = False

    def fast_search(self, binary_warped):
        self.binary_warped = binary_warped
        self.nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        self.left_lane_inds = ((self.nonzerox > (self.left_fit[0] *
                                                 (self.nonzeroy ** 2) + self.left_fit[1] * self.nonzeroy +
                                                 self.left_fit[2] - self.margin)) &
                               (self.nonzerox < (self.left_fit[0] * (self.nonzeroy ** 2) +
                                                 self.left_fit[1] * self.nonzeroy +
                                                 self.left_fit[2] + self.margin)))

        self.right_lane_inds = ((self.nonzerox > (self.right_fit[0] *
                                                  (self.nonzeroy ** 2) + self.right_fit[1] * self.nonzeroy +
                                                  self.right_fit[2] - self.margin)) & (self.nonzerox <
                                                                                       (self.right_fit[0] * (
                                                                                       self.nonzeroy ** 2) +
                                                                                        self.right_fit[1] *
                                                                                        self.nonzeroy + self.right_fit[
                                                                                            2] + self.margin)))

        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        if len(self.leftx) < 25 or len(self.rightx) < 25:
            self.exhaustive_search(binary_warped)

        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        self.fity = np.linspace(0, self.binary_warped.shape[0] - 1, self.binary_warped.shape[0])
        self.fit_leftx = self.left_fit[0] * self.fity ** 2 + self.left_fit[1] * self.fity + self.left_fit[2]
        self.fit_rightx = self.right_fit[0] * self.fity ** 2 + self.right_fit[1] * self.fity + self.right_fit[2]

    def plot(self):
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
        return result, self.fit_leftx, self.fit_rightx, self.fity

    #         plt.imshow(result)
    #         plt.plot(self.fit_leftx, self.fity, color='yellow', linewidth=5.0)
    #         plt.plot(self.fit_rightx, self.fity, color='yellow', linewidth=5.0)
    #         plt.xlim(0, 1280)
    #         plt.ylim(720, 0)

    def smooth_fit(self, binary_warped):
        self.current_image = binary_warped
        leftx, lefty = [], []
        rightx, righty = [], []

        for i, lane in enumerate(self.old_lanes):
            leftx.extend(lane.leftx)
            lefty.extend(lane.lefty)
            rightx.extend(lane.rightx)
            righty.extend(lane.righty)
        leftx = np.array(leftx)
        rightx = np.array(rightx)
        lefty = np.array(lefty)
        righty = np.array(righty)
        self.leftx, self.lefty = leftx, lefty
        self.rightx, self.righty = rightx, righty
        self.fity = np.linspace(0, self.current_image.shape[0] - 1, self.current_image.shape[0])
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        self.fit_leftx = np.array(self.left_fit[0] * self.fity ** 2 + self.left_fit[1] * self.fity + self.left_fit[2])
        self.fit_rightx = np.array(
            self.right_fit[0] * self.fity ** 2 + self.right_fit[1] * self.fity + self.right_fit[2])
        self._set_variables(leftx, rightx, lefty, righty, self.left_fit, self.right_fit, self.fit_leftx,
                            self.fit_rightx)

    def _set_variables(self, leftx, rightx, lefty, righty, left_fit, right_fit, fit_leftx, fit_rightx):
        self.leftx, self.lefty = leftx, lefty
        self.rightx, self.righty = rightx, righty
        self.left_fit, self.right_fit = left_fit, right_fit
        self.fit_leftx, self.fit_rightx = fit_leftx, fit_rightx

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

    def translate_to_real_world_image(self, image, Minv):
        warped = self.binary_warped
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(
            np.vstack([self.fit_leftx, self.fity]))])
        pts_right = np.array([np.flipud(np.transpose(
            np.vstack([self.fit_rightx, self.fity])))])

        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, Minv, self.img_shape)
        # image = cv2.warpPerspective(image, Minv, self.img_shape)
        left_curvature, right_curvature = self.calculate_curvature(real=True)

        cv2.putText(image, "Curvature: " + str(int(left_curvature)) + '(m)',
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), thickness=3)
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