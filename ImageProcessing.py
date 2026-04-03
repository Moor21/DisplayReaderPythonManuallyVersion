import cv2
import numpy as np
class ImageProcessing:
    def __init__(self):
        pass
    def getCannyImage(self,image, blur_kernel,min_threshold,max_threshold, morph_kernel):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur= cv2.GaussianBlur(gray, blur_kernel, 0)
        canny= cv2.Canny(blur, min_threshold, max_threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
        morph = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        return canny
    def getContours(self,prepared_image):
        contours,_ = cv2.findContours(prepared_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours
    def getContoursImage(self, pure_image, contours, color):
        image = pure_image.copy()
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        location = []
        for contour in contours:
            print("Contour area: ", cv2.contourArea(contour))
            epsilon = 0.02*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour, epsilon, True)    
            if len(approx) == 4:
                location.append(approx)
                break;

        for loc in location:        
            current_rect = cv2.boundingRect(loc)
            print("Rect Area: ", current_rect[2]*current_rect[3])
            cv2.rectangle(image, (current_rect[0], current_rect[1]), (current_rect[0]+current_rect[2], current_rect[1]+current_rect[3]), (0,255,0), 3)
        return image
    def getThresholdImage(self,image, blur_kernel,max_threshold, morph_kernel):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur= cv2.GaussianBlur(gray, blur_kernel, 0)
        threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
        morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        return threshold
        
    def getRectanglesForAllContours(self, contours):
        rectangles = []
        for contour in contours:
            current_rect = cv2.boundingRect(contour)
            rectangles.append(current_rect)
        return np.array(rectangles)
    def _distance(self,ptr1,ptr2):
        return np.linalg.norm(ptr1-ptr2)
    def findDisplayContour(self,pure_image, contours):
        marked_display_image = pure_image.copy()
        location = None
        for contour in contours:
            if cv2.contourArea(contour) > 17000 and cv2.contourArea(contour) < 19000:
                epsilon  = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) ==4:
                    location = approx
        
        if location is not None:
            #----find all the required points-----#
            points = location.reshape(4,2)
            summs = np.sum(points, axis=1)
            diffs = np.diff(points, axis=1)

            top_left_point = points[np.argmin(summs)]
            top_right_point = points[np.argmin(diffs)]
            bottom_right_point = points[np.argmax(summs)]
            bottom_left_point = points[np.argmax(diffs)]

            #-----construct input_matrix------#
            input_matrix = np.float32([
                top_left_point,
                top_right_point,
                bottom_right_point, 
                bottom_left_point
            ])
            #------compute width and height------#
            width = int(max(self._distance(top_left_point,top_right_point), self._distance(bottom_left_point, bottom_right_point)))
            height = int(max(self._distance(top_left_point, bottom_left_point), self._distance(top_right_point, bottom_right_point)))
            #------construct output matrix------#
           
            output_matrix= np.float32([
                 [0,0],
                 [width, 0], 
                 [width, height],
                 [0,height]
                ])
            #-------get perspectiveTransform matrix-----#
            matrix = cv2.getPerspectiveTransform(input_matrix, output_matrix)
            #-----apply perspectiveTransform
            cv2.rectangle(marked_display_image, top_left_point, bottom_right_point, (255,0,0),2)
            marked_display_image = cv2.warpPerspective(marked_display_image, matrix, (width, height))
        else:
            print("Display is not found!")
            marked_display_image = None


        display_type = 1 if width/height < 1 else 0
        if display_type:
            marked_display_image = cv2.rotate(marked_display_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return marked_display_image

