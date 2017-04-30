# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for the camera calibration is contained in the cells 2 and 3 of the python notebook. The cells 4 and 5 are used to plot some results from the calibration. In particular the cell number 4 shows the result on the calibration images while the cell number 5 shows the results on test images.

All the imageds are contained in the output_images/calibration_images folder.

The code used to calibrate the camera is located into the `camera_calibrator.py` file (lines 40-44).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world using the np.mgrid method. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, so that the object points are the same for each calibration image.  Thus, `obj` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Among the 20 given calibration images, only 17 of them share the same number of corners (9x6) and for that reason the camera has been calibrate only on 17 images.

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the calibration and test images using the `cv2.undistort()` function. In the case of a calibration image i obtained this result: 

![alt text](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/blob/master/output_images/calibration_images/chessboard_images/chessboard_calibration_output_1.jpg "Calibration image")

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/blob/master/output_images/calibration_images/test_images/tets_calibration_output_3.jpg "Test image")

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. 

The color mask is applied through the method apply_color_threshold() implemented in the cell 6 of the notebook. The color threshold is applied to an hsv image. The conversion from the RGB to the HSV color spaces has been implemented using the cv2.cvtColor() function.

A gradient threshold has also been applied by means of the sobel operators. The code for the gradient mask is located into the `gradient_threhold.py` file (lines 62-71). 

I then combined the color and gradient masks by calling method get_color_gradient_combined() implemented in the cell 6 of the notebook. Here is an example of gradient and color thresholds applied to a test image:

![alt text](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/blob/master/output_images/pipeline_result_images/color_gradient_threshold.jpg " ")

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in the cell 6 of the notebook.  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I used the following points for source and destination:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 720      | 205, 720        | 
| 1180, 720      | 1120, 720      |
| 780, 480     | 1120, 0      |
| 555, 480      | 205, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Before to warp the image I also applied a mask in order to select the region of the interest. The mask is applied by means of the method region_of_interest() implemented in the cell number 6 of the notebook. The result of the warp procedure on a test image is the following:

![alt text](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/blob/master/output_images/pipeline_result_images/warp_image.jpg "Warp image")

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to find the lane lines I used the code provided by udacity that I then modified. All the code relative to this goal is implemented in the `lane_lines.py` files (lines 39-134). I used the sliding window technique. 

For the first image the algorithm moves through the windows (lines 39-108). Next I use the previous fit in order to locate the lane lines (lines 110-134).

The final step is to convert back the image to the real world and highlight the lane line. The relative code is implemented in the `lane_lines.py` file (lines 160-180)

The result of previous procedure is shown in the figure below:

![alt text](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/blob/master/output_images/pipeline_result_images/pipeline_results.jpg "Pipeline result")

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In order to evaluate the radius of curvature and the position of the vehicle with respect to the center I basically used the code given by udacity plus some hints from various blogs such as Stack Overflow. I refactored it for my convenience and the final implementation is in the `lane_lines.py` file (lines 210-231)

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I tested the pipeline on the three given video. The results are in the folder output_video.

Here are the links to the videos:

[link to project_video result](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/tree/master/output_video/project_video_output.mp4)

[link to challenge_video result](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/tree/master/output_video/challenge_video_output.mp4)

[link to harder_challenge_video result](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/tree/master/output_video/harder_challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During the development of the code I faced several optimization problems. It is not hard to implement a working code but it is really hard to tune it and to find the proper way to set some model parameter. I spent a lot time trying different threshold values for gradient and color masks. It also took some time to figure out how to combine the different gradients (x, y, mag and dir) among them. 

Currently, I think the code can fail (and it actually does) on the harder_challenge video due to two main factors: the fix area in which the code looks for lane lines, the high variability in the brightness of the image. I tried to use image equalization but it wasn't enough.

Another crucial point regards the selection on the minimum number of pixels required to inside a search window in order to identify a lane candidate. Currently it is a fixed parameter but I think that it should vary somehow. That fact can be very fundamental in cases where the lane lines are not perfectly visible and such a situation indeed happen in the challenge video. I think that using camera images from different angles can help improving all these aspects. 

A possible improvement could be also to use statistical inference based on some template that could give the "likelihood" of a search result.

