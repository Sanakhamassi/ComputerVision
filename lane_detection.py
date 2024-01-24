import numpy as np
import cv2
#video = several frames(images shown aftereach other)
video=cv2.VideoCapture('lane_detection_video.mp4')
def region_of_intrest(image,region_points):
    # we are going to replace pixels with 0(black)-the regions we are not intrested
    mask=np.zeros_like(image)
    #the region that we're intrested in is the lower triangle -255 white pixels
    cv2.fillPoly(mask,region_points,255)
    #we have to use the mask : we want to keep the regions of the original image
    #where the mask has white colored pixels
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image
def get_detected_lanes(image):
    #get dimenyion of a given frame
    (height,width)=(image.shape[0],image.shape[1])
    #we have to turn the image into grayscale
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #edge detection kernel (Canny's algorithm)
    canny_img=cv2.Canny(gray_image,100,120)
    #we are intresstredin the "Lower region" of the image (There are the driving lanes)
    region_of_itrests_vertices=[
        (0,height),
        (width/2,height*0.65),
        (width,height)
    ]
    #we can get rid of unrelevent parts of image we just keep the lower triangle region
    cropped_image=region_of_intrest(canny_img,np.array([region_of_itrests_vertices],np.int32))
    return cropped_image
while video.isOpened():
    #is_grapped will returns a boolean value wether the frame was returned successfully or not
    is_grabbed,frame=video.read()
    # because the end of the video
    if not is_grabbed:
        break
    frame=get_detected_lanes(frame)
    cv2.imshow("lane detection video",frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()