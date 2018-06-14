import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from functions import *

debug = False
#debug = True


#Initialize some variables
ret, mtx, dist, rvecs, tvecs = calibrateCam()

src = np.float32([(555, 480), (752, 480), (228, 700), (1072, 700)])
dst = np.float32([(228, 315), (1072, 315), (228, 700), (1072, 700)])

#left_line = Line()
#right_line = Line()




def find_line(img):

    #Copy the img
    img_temp = np.copy(img)

    #Undistort the img
    image = cv2.undistort(img_temp, mtx, dist, None, mtx)

    # Choose a Sobel kernel size
    ksize = 15 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(50, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_thresh(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hsv_mask = hsv_filter(image)
    rgb_mask = rgb_filter(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if debug:
        mpimg.imsave('output_images/gradx.jpg', gradx, cmap = 'gray') 
        mpimg.imsave('output_images/grady.jpg', grady, cmap = 'gray') 
        mpimg.imsave('output_images/mag_binary.jpg', mag_binary, cmap = 'gray') 
        mpimg.imsave('output_images/dir_binary.jpg', dir_binary, cmap = 'gray') 
        mpimg.imsave('output_images/hsv_mask.jpg', hsv_mask, cmap = 'gray') 
        mpimg.imsave('output_images/rgb_mask.jpg', rgb_mask, cmap = 'gray') 
        mpimg.imsave('output_images/gray_origial.jpg', gray, cmap = 'gray')   

 
    combined = np.uint8(np.zeros_like(gray))
    combined[((gradx == 1) & (grady == 1) ) | ((mag_binary == 1) & (dir_binary == 1) ) | (hsv_mask == 1) | (rgb_mask == 1)] = 1
    combined[:combined.shape[0] // 2 + 100, :] = 0
    combined[670:, :] = 0
    gray[(combined == 0)] = 0
 
    if debug:
        mpimg.imsave('output_images/gray.jpg', gray, cmap = 'gray') 
    
    binary_warped = get_perspective(gray, src, dst)
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
   
 
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
   

 
    if debug:
        mpimg.imsave('output_images/out_img01.jpg', out_img, cmap = 'gray')

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
   
    #Get radius of curvature of the lane 
    left_radius_deri1 = 2 * left_fit[0]*ploty + left_fit[1]
    left_radius_deri2 = 2 * left_fit[0]
    left_radius = np.mean(np.power( (1 + left_radius_deri1 ** 2), 3/2) / np.abs(2 * left_fit[0]))
    
    right_radius_deri1 = 2 * right_fit[0]*ploty + right_fit[1]
    right_radius_deri2 = 2 * right_fit[0]
    right_radius = np.mean(np.power( (1 + right_radius_deri1 ** 2), 3/2) / np.abs(2 * right_fit[0]))


    #fit_width = 400
    #n = 72
    #fit_distance = abs(left_fitx - right_fitx)
    #max_fit_distance = np.max(fit_distance)
    #min_fit_distance = np.min(fit_distance)
    #mean_fit_distance = np.mean(fit_distance)

    #if abs(mean_fit_distance - fit_width) < 50 and max_fit_distance - min_fit_distance < 10:
    #    left_line.detected = True
    #    right_line.detected = True
   
    #    left_line.recent_xfitted.append(left_fitx)
    #    left_line.bestx = np.mean(left_line.recent_xfitted, axis=0) 

    #    right_line.recent_xfitted.append(right_fitx)
    #    right_line.bestx = np.mean(right_line.recent_xfitted, axis=0) 


    #if len(left_line.recent_xfitted) >= n:
    #    left_line.recent_xfitted.pop(0)
    #    right_line.recent_xfitted.pop(0)
    #            

    #    if not (left_line.detected):
    #        left_fitx = left_line.bestx
    #        right_fitx = right_line.bestx
    

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +  left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Create an image to draw on and an image to show the selection window
    out_img_binary_warped = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img_binary_warped)
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    line_pts = np.hstack((left_line_window1, right_line_window2))
    
    middlex = (right_fitx - left_fitx) // 2 
    car_pos_left_window1 = np.array([np.transpose(np.vstack([left_fitx + middlex - 5, ploty]))])
    car_pos_right_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx - middlex + 5, ploty])))])
    car_pos_pts = np.hstack((car_pos_left_window1, car_pos_right_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([car_pos_pts]), (255,48, 48))
    
    if debug:
        mpimg.imsave('output_images/window_img.jpg', window_img)
    
    
    persimg = get_perspective(window_img, dst, src)
    
    #Calculate how far the car from center of lane
    car_pos_nonzero = persimg[window_img.shape[0] - 1].nonzero()
    car_pos = window_img.shape[1] // 2 - np.max(car_pos_nonzero) + middlex[middlex.shape[0] - 1]

    font=cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.addWeighted(image, 1, persimg, 0.3, 0)

    size = (213, 120) 
    gray_shrink = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)  
    gray_shrink_stack = np.dstack((gray_shrink, gray_shrink, gray_shrink))

    hsv_mask_shrink = cv2.resize(hsv_mask, size, interpolation=cv2.INTER_AREA)  
    hsv_mask_shrink_stack = np.dstack((hsv_mask_shrink, hsv_mask_shrink, hsv_mask_shrink)) * 255
    window_img_shrink = cv2.resize(window_img, size, interpolation=cv2.INTER_AREA) 
    out_img_shrink = cv2.resize(out_img, size, interpolation=cv2.INTER_AREA) 

    #Add some pictures in the process onto the image 
    result[0:120, 0:213] = gray_shrink_stack
    result[0:120, 214:427] = hsv_mask_shrink_stack
    result[121:241, 0:213] = window_img_shrink
    result[121:241, 214:427] = out_img_shrink

    #Add text onto the image
    text_start_pixel = 700
    result = cv2.putText(result,'left_radius:' + str(int(left_radius)),(text_start_pixel,40),font,1.2,(0,0,0),2)
    result = cv2.putText(result,'right_radius:' + str(int(right_radius)),(text_start_pixel,90),font,1.2,(0,0,0),2)
    result = cv2.putText(result,'mean_fit_distance:' + str(int(np.mean(abs(left_fitx - right_fitx)))),(text_start_pixel,140),font,1.2,(0,0,0),2)
    result = cv2.putText(result,'diff_radius:' + str(int(abs(left_radius - right_radius))),(text_start_pixel,190),font,1.2,(0,0,0),2)
    
    if car_pos > 0:
        result = cv2.putText(result,'Vehicle is ' + str(abs(int(car_pos))) + ' px rt of center',(text_start_pixel,240),font,1.2,(0,0,0),2)
    else:
        result = cv2.putText(result,'Vehicle is ' + str(abs(int(car_pos))) + ' px lt of center',(text_start_pixel,240),font,1.2,(0,0,0),2)
    
    if debug:
        mpimg.imsave('output_images/result02.jpg', result)
   
    return result





if not debug:
    #Test the images
    images = os.listdir("test_images/")
    for i in range(len(images)):
        image = mpimg.imread('test_images/' + images[i])
        img = find_line(image)
        mpimg.imsave('output_images/' + images[i], img)
    
    
    
    #Test the project_video
    from moviepy.editor import VideoFileClip
    white_output = 'output_videos/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(find_line) 
    white_clip.write_videofile(white_output, audio=False)

else:

    #Generate some images to writeup README.md

    img = mpimg.imread('test_images/test5.jpg')
    #img = mpimg.imread('test_images/test1.jpg')
    #img = mpimg.imread('temp/03/img_0000187_a.jpg')
    #img = mpimg.imread('test_images/img_0000569_a.jpg')
    result = find_line(img)
    
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    img_perspective = get_perspective(img, src, dst)
    img_unperspective = get_perspective(img_perspective, dst, src)
    
    mpimg.imsave('output_images/test5_distort.jpg', img)
    mpimg.imsave('output_images/test5_undistort.jpg', img_undist)
    mpimg.imsave('output_images/test5_perspective.jpg', img_perspective)
    mpimg.imsave('output_images/test5_unperspective.jpg', img_unperspective)



    img = mpimg.imread('camera_cal/calibration1.jpg')
    calibration_undist = cv2.undistort(img, mtx, dist, None, mtx)

    mpimg.imsave('output_images/calibration_distort.jpg', img)
    mpimg.imsave('output_images/calibration_undistort.jpg', calibration_undist)
