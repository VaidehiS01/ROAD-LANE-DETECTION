import cv2
import numpy as np

def make_coordinates(image,line_parameters): #used for placing the optimized lines at correct locations on the road image
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope) #from the realtion y=mx+b
    x2=int((y2-intercept)/slope) #from the realtion y=mx+b
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines): #used to optimize the lane detection in the final image
    left_fit=[] #lines of the right half
    right_fit=[] #lines of the left half
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1) #y = mx + b
        slope=parameters[0]  #m
        y_intercept=parameters[1] #b
        if slope < 0: #lines on the left of road have negative slop
            left_fit.append((slope,y_intercept)) #left_fit and right_fit are a list of tuples containing slope and y_intercept respectively
        else:
            right_fit.append((slope,y_intercept))
    left_fit_average=np.average(left_fit,axis=0) 
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordinates(image,left_fit_average)
    right_line=make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])

def canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  #Converts the Color image to grayscale
    blur=cv2.GaussianBlur(gray,(5,5),0) #applies Gaussian Blur to the grayscale image with a 5X5 kernel and deviation value of 0 to reduce noise and smoothen out the image
    #Appying Canny Edge detection
    canny=cv2.Canny(blur,50,150) #this basically shows the gradient image. If the gradient value is lesser than the lower_threshold argument then it is rejected as an edge, if the gradient value is greater than the upper_threshold then it is shown as an edge(white line).Any gradient value in between is shown as an edge only if it is connected to a main edge
    return canny

def display_lines(image,lines): #lines is a 3D array
    line_image=np.zeros_like(image) #black image
    if lines is not None:
        for line in lines: #line is a 2D array
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10) #To draw a line segment on line_image taking two points (x1,y1) and (x2,y2). After this comes the 'BGR' color code. Last argument is the value of line thickness
    return line_image

def region_of_interest(image):
    '''returns the region of interest as a triangle'''
    height=image.shape[0] #shape returns (m,n,l) of the numpy array. Here as we need the height of image = no of rows(m) we set the index as 0
    polygons=np.array([[(200,height),(1100,height),(550,250)]])  #the coordinates of the triangle(area of interest)
    # polygons=np.array([[(450,height),(1275,height),(950,550)]])
    mask=np.zeros_like(image) #creates an array of the same dimensions as the image except that all the pixels have 0 intensity which means it will be a black background
    cv2.fillPoly(mask,polygons,255) #fill the black background(mask) with triangle of 255 intensity(white color)
    masked_image=cv2.bitwise_and(image,mask) #perform bitwise and (&) on the canny image and the mask image to get the area of interest 
    return masked_image

#image lane detection
# image=cv2.imread('test_image.jpg') #RGB color image
# lane_image=np.copy(image)
# canny_image=canny(lane_image)
# cropped_image=region_of_interest(canny_image)
# lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) #the hough grids are used to determine the values of rho and thetha for the line that will pass through the given points. The grids have a resolution of 2 pixels and 1 degree(pi/180 radians). The fourth argument is the value of minimum threshold for the number of intersections
# averaged_lines=average_slope_intercept(lane_image,lines)
# line_image=display_lines(lane_image,averaged_lines)
# combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1) #combine the line_image(completely black with just the lines) with the original color image
# cv2.imshow("Result",combo_image)
# cv2.waitKey(0) #display the image infinitely until any key is pressed

# video lane detection
cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    #each frame of the video is similar to an image so we can apply the same image line detection algorithm here 
    canny_image=canny(frame)
    cropped_image=region_of_interest(canny_image)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) 
    averaged_lines=average_slope_intercept(frame,lines)
    line_image=display_lines(frame,averaged_lines)
    combo_image=cv2.addWeighted(frame,0.8,line_image,1,1) 
    cv2.imshow("Result",combo_image)
    if cv2.waitKey(1) & 0xFF ==ord("q"): #waitkey(1)waits 1ms between each frame. It also returns a 32 bit integer value. This property can be used to close the video on pressing a key(here 'q')
        break #break when the key 'q' is pressed
cap.release()
cv2.destroyAllWindows()