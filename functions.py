import matplotlib.image as mpimg
import numpy as np
import cv2
import os

#debug mode
debug = False


def calibrateCam():

    #Get the images' name
    images = os.listdir("camera_cal/")

    objpoints = [] # original points
    imgpoints = [] # img points

    w = 9 # widht of chessboard corners
    h = 6 # height of chessboard corners

    #Get objpoints
    objp = np.zeros((w*h,3), np.float32)

    num = 0
    for h1 in range(h):
        for w1 in range(w):
            objp[num] = [w1, h1, 0]
            num += 1

    #Get calibrate parameters
    for fname in images:
        img = mpimg.imread('camera_cal/' + fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (w,h), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
   
    #return calibrate parameters 
    return ret, mtx, dist, rvecs, tvecs





def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh = (0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output




def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output




def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobel_x)
    abs_sobely = np.absolute(sobel_y)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.uint8(np.zeros_like(dir_sobel))
    binary_output[(dir_sobel > thresh[0]) & (dir_sobel < thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output




def hsv_filter(img):

    #Copy the img
    imgtemp = np.copy(img)

    #Blur the img
    blur = cv2.GaussianBlur(imgtemp,(5,5),5)

    #Get hsv values from the img
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]   

    #Create a binary mask
    mask = np.uint8(np.zeros_like(h)) 
    mask[(h > 11) & (h < 34) & (s > 100) & (s < 255) & (v > 150) & (v < 255)] = 1 #yellow
    mask[(h > 0) & (h < 180) & (s > 0) & (s < 30) & (v > 220) & (v <= 255)] = 1 #white

    #Crop the mask
    mask[:mask.shape[0] // 2, :] = 0
    mask[670:, :] = 0

    if debug:   
        mpimg.imsave('output_images/mask.jpg', mask, cmap = 'gray') 
        mpimg.imsave('output_images/hsv_h.jpg', h, cmap = 'gray') 
        mpimg.imsave('output_images/hsv_s.jpg', s, cmap = 'gray') 
        mpimg.imsave('output_images/hsv_v.jpg', v, cmap = 'gray') 

    return mask




def rgb_filter(img):

    #Copy the img
    imgtemp = np.copy(img)

    #Get rgb values from img
    r = imgtemp[:,:,0]
    g = imgtemp[:,:,1]
    b = imgtemp[:,:,2]

    #Create a binary mask
    mask = np.uint8(np.zeros_like(r))
    mask[(r >= 180) & (r <= 255) & (b >= 30) & (b <= 120) & ((g < 120) | (g > 180))] = 1 # yellow
    mask[(r >= 230) & (r <= 255) & (g >=230) & (g <= 255) & (b >= 230) & (b <= 255)] = 1  #white
    mask[:mask.shape[0] // 2, :] = 0
    mask[670:, :] = 0

    if debug:
        mpimg.imsave('output_images/rgb_r.jpg', r, cmap = 'gray')
        mpimg.imsave('output_images/rgb_g.jpg', g, cmap = 'gray')
        mpimg.imsave('output_images/rgb_b.jpg', b, cmap = 'gray')
        #mpimg.imsave('output_images/rgb_mask.jpg', mask, cmap = 'gray')

    return mask




def get_perspective(grayimg, src, dst):

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    img_size = (grayimg.shape[1], grayimg.shape[0])
    warped = cv2.warpPerspective(grayimg, M, img_size)

    return warped





#class Line():
#    def __init__(self):
#        # was the line detected in the last iteration?
#        self.detected = False  
#        # x values of the last n fits of the line
#        self.recent_xfitted = [] 
#        #average x values of the fitted line over the last n iterations
#        self.bestx = None     
#        #polynomial coefficients averaged over the last n iterations
#        self.best_fit = None  
#        #polynomial coefficients for the most recent fit
#        self.current_fit = [np.array([False])]  
#        #radius of curvature of the line in some units
#        self.radius_of_curvature = None 
#        #distance in meters of vehicle center from the line
#        self.line_base_pos = None 
#        #difference in fit coefficients between last and new fits
#        self.diffs = np.array([0,0,0], dtype='float') 
#        #x values for detected line pixels
#        self.allx = None  
#        #y values for detected line pixels
#        self.ally = None



