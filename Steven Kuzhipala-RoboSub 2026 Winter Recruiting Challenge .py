#Created by: STEVEN KUZHIPALA
#For ROBOSUB UCI - Recruitment 2026
#Program to detect objects colored in RED and its shade in an underwater enviroment using openCV.
#Handling noise and filtering using Gaussian Blurring, using HSV and LAB color spaces, and contours for appropriate bounding circles.

#Future possible improvements: Depending on the size of the object to be detected, we can implement a simple if condition filter to only detct object covering a certain
#                              pixel area in the image. This would help eliminate any small red objects that are act as noise in the image. 
#                              We coudl also further the tool's robutstness by allowign the usee to select their color of interest by providng RGB values and detecting 
#                              required color in the image. 
import cv2
import numpy as np

#Function to load image in a different window
def imshow(windowName,image):
    cv2.imshow(windowName,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#LOADING IMAGE FROM DATA SET
#Image inputted was stored in my local directory
image_path = "add directory with image of interest here"  
image = cv2.imread(image_path)
imshow("Original Image", image)

#APPLYING GAUSSIAN BLUR
#Reasoning: By Applying Gaussian blur and averaging pixel values over assigned kernel sizes of pixels, we are
#           able to effectively remove any small pixel-level noise, or tiny variations in color from the image.
#           We are also smoothening out the image as well.
blurred_image = cv2.GaussianBlur(image, (7, 7), 0)  #Gaussian Blur Kernal over the image.
imshow("Applying Gaussian Blur", blurred_image)

#LOADING IMAGE IN HSV COLOR SPACE TO DETECT HUES OF RED IN VARYING LIGHTING
#Reasoning: By loading the image in HSV lab space and using its mask, we can mitigate the effect of variable lighting underwater enviroments.
image_hsv=cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 80, 40])   #Bright red, low saturation, dim lighting.
upper_red1 = np.array([20, 255, 255]) #Bright orange-red, high saturation, bright lighting.
lower_red2 = np.array([160, 80, 40]) #Deep red, low saturation, dim lighting.
upper_red2 = np.array([180, 255, 255]) #Deep purplish-red, high saturation, bright lighting.
mask_hsv1 = cv2.inRange(image_hsv, lower_red1, upper_red1)  #HSV Mask for lower and upper bounds of bright reds.
mask_hsv2 = cv2.inRange(image_hsv, lower_red2, upper_red2)   #HSV Mask for lower and upper bounds of deeper reds.
mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)  #Object can be either in a bright red hue OR in a deep red hue and it will be detected.
imshow("HSV Mask", mask_hsv)    #Since we have already seperated the bright and deep reds, we could also detect them seperately if need be.


#LOADING IMAGE IN LAB COLOR SPACE TO FIND THE COLOR RED
#Reasoning: LAB Space solves this issue by separating into a(green-red) to map only green and red colors, ignoring the blue completely.
#           This is quite useful in our underwater image, which is filled with the color blue.
#           Since the colors yellow, red, and green are at opposite ends of the blue color axis, we can easily seperate them.
#           When we specify the detection to be solely for a-channel, we can clearly see any red objects to glow in white the converted image mask.
image_lab = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)
l,a,b = cv2.split(image_lab)   #split into LAB color space, with  b=(Blue-Yellow axis), a=(Green-Red axis), l channel only for brightness.
mask_lab = cv2.inRange(a, 128, 255)  #We have manually set the threshold for the color red, and any value above that threshold will be WHITE in the
                                                    #image, and everything else will be colored black. https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
imshow("Lab Mask", mask_lab)

#COMBING BOTH HSV AND LAB COLOR SPACES
#Reasoning:  HSV allows us to separate the effect of brightness on the image, since the underwater image is not necessarily always going to be in perfect lighting,
#            and using HSV helps us eliminate its possible effect, instead of relying on the LAB color space. (We cannot use LAB color space 'l' channel as its brightness ranges are developed for human perception and not general 'brightness' or an image)
#            The LAB color space helps confirm that the detected hue by the HSV space mask is truly Red even in its own red axis ranges,
#            giving us more confidence in our detection for the color of our interest.
combined_mask = cv2.bitwise_and(mask_hsv, mask_lab)
imshow("Combined HSV & Lab Mask", combined_mask)


#USING CONTOURS TO CREATE BOUNDING CIRCLES IN IMAGE FOR RED OBJECTS
#Reasoning: using cv2.inRange function(used for segmentation) on our LAB converted image, draws boundaries on the topmost point and the bottommost point of where it finds the color of interest.
#            This would mean that if there are two separate objects of the color of interest, a single large bounding circle is drawn on them instead of two separate ones.
#            cv2.findContours is used for shape detection and finds simultaneous curves of lined up pixels and gest their coordinates to help draw bouding shapes.
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #cv2.RETR_EXTERNAL finds the outline of the object of interested color. Anything inside that object will be ignored. (hence we can ignore the one of the return values which order in which any object with multiple shapes in it
                                                                                   #cv2.CHAIN_APPROX_SIMPLE only stores the corner points that covers the object of interested color .
                                                                                   #https://www.ccoderun.ca/programming/doxygen/opencv_3.2.0/tutorial_bounding_rects_circles.html
                                                                                   #Now len(contours) will even return the number of different objects of interested color is detected.

#DRAWING BOUNDING CIRCLES ON IMAGE
output_image = image.copy()     #Create a copy of the original image to draw the bounding shapes on.
# 'contours' is a list of all white(interested color)  shapes found
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    center_x = x + w // 2   #X-coordinate of circle's center
    center_y = y + h // 2   #Y-coordinate of circle's center
    radius = max(w, h) // 2  #Circle's radius is half of the maximum of width or height
    output_image = cv2.circle(output_image, (center_x, center_y), radius, (0, 255, 0), 2)  # drawing green outline circle using cv2 package
    #output_image = cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  #drawing rectangle using cv2 package

imshow("Detected Objects", output_image)

#Saving the image to upload to Submission form
cv2.imwrite("Detected Red output.jpeg", output_image)


