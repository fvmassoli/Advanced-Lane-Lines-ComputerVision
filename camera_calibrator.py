import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle


class CameraCalibrator(object):
    def __init__(self, chess_board_nx, chess_board_ny, images):
        self.chess_board_nx = chess_board_nx
        self.chess_board_ny = chess_board_ny
        self.images = images
        self.img_size = None
        self.mtx = None
        self.dist = None

    def _define_grid_points(self, ):
        obj = np.zeros((self.chess_board_ny * self.chess_board_nx, 3), np.float32)
        obj[:, :2] = np.mgrid[0:self.chess_board_nx, 0:self.chess_board_ny].T.reshape(-1, 2)
        return obj

    def _get_corners(self, obj, images):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        counter = 0
        obj_points = []
        img_points = []
        for image in images:
            img = mpimg.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.img_size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (self.chess_board_nx, self.chess_board_ny), None)
            if ret == True:
                counter += 1
                obj_points.append(obj)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)
        print('Camera ready!')
        return obj_points, img_points

    def calibrate_camera(self, ):
        obj = self._define_grid_points()
        obj_points, img_points = self._get_corners(obj, self.images)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, self.img_size, None, None)
        self.save_calibration_results(self.mtx, self.dist)

    def undistort_image(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_camera_calib_param(self, ):
        return self.mtx, self.dist

    def save_calibration_results(self, mtx, dist):
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open("calibration.p", "wb"))
        return None

    def plot_calibration_check(self, image):
        undistored_image = self.undistort_image(image)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        ax1.set_title('Orignal Image')
        ax1.imshow(image)
        ax2.set_title('Undistorted Image')
        ax2.imshow(undistored_image)
        return None