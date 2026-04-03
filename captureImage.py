import cv2

cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame=cap.read()
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        i = i + 1
        filename = f"image{i}.png"
        print(filename)
        cv2.imwrite(filename, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    