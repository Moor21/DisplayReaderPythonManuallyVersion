cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    i+=1
    if i % 3 == 0:
        cv2.imshow("frame", frame)
        whole_digit = readFromFrame(frame,proc)
        writeInFile(whole_digit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
