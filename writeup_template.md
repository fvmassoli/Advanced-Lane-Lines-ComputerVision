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

The for the camera calibration is contained in the cells 2 and 3 of the python notebook. The cells 4 and 5 are usde to plot some resutls from the calibration. In particular the cell number3 shows the result on the calibration images while the cell number 4 shows the results on test images.

All the imageds are contained in the output_images/calibration_images folder.

The code used to calibrate the camera is located into the `camera_calibrator.py` (lines 40-44) file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world using the np.mgrid method. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

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

A gradient threshold has also been applied by means of the sobel operators. The code the graident mask is located into the `gradient_threhold.py` file (lines 62-71). 

I then combined the color and gradient masks by calling method get_color_gradient_combined() implemented in the cell 6 of the notebook. Here is an example of graient and color thresholds applied to a test image:

![alt text](https://github.com/fvmassoli/CarND-Advanced-Lane-Lines-P4/blob/master/output_images/pipeline_result_images/color_gradient_threshold.jpg " ")

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.



The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]


















####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

