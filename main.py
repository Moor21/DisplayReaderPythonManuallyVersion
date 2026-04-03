import cv2
from ImageProcessing import ImageProcessing
from DigitBoxes import DigitBoxes

def readFromFrame(frame,proc):
    whole_digit = None
    image = frame
    canny = proc.getCannyImage(image, (5,5),80,100,(7,7))
    contours = proc.getContours(canny)
    marked_display_image = proc.findDisplayContour(image, contours)
    if marked_display_image is not None:
        cv2.imshow("marked_display", marked_display_image)
        box = DigitBoxes(marked_display_image,proc)
        box.loadPureImage(marked_display_image)
        thresholdedDisplay = box.getDisplayThresholdImage()
        whole_digit = box.getWholeDigitString(thresholdedDisplay)
    else:
        print("Display is not found!")
    print("Whole_digit: ", whole_digit)
    return whole_digit
def writeInFile(whole_digit):
    if whole_digit == None:
        print("Whole_digit is None")
    else:
        real_number = int(whole_digit) / 100
        print("Real_number: ", real_number)
        file_path = "config.txt"
        with open(file_path, 'a') as file:
            file.write("\n")
            file.write(str(real_number))
            print(f"txt file {file_path} was created!")



proc = ImageProcessing()
#------Video from camera-----#
# cap = cv2.VideoCapture(0)
# i = 0
# while True:
#     ret, frame = cap.read()
#     i+=1
#     if i % 3 == 0:
#         cv2.imshow("frame", frame)
#         whole_digit = readFromFrame(frame,proc)
#         writeInFile(whole_digit)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


#-------Image-----#
image = cv2.imread("images/display.jpg")
image = cv2.GaussianBlur(image, (3,3),0)
cv2.imshow("Image", image)
whole_digit = readFromFrame(image,proc)
print("Whole_digit: ", whole_digit)
cv2.waitKey(0)


