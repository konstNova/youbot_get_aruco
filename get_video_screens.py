import cv2 as cv
import imutils

SCREEN_NAME = "aruco1"
# захват видео потока
cam = cv.VideoCapture(0)
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)

img_counter = 0

while True:
    # получение кадров
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    # изменить размер окна
    frame = imutils.resize(frame, width=860)
    # показать окно с видео
    cv.imshow("test", frame)

    k = cv.waitKey(1)
    # выход
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    # сохранить скрин
    elif k%256 == 32:
        # SPACE pressed
        folder_path = "/home/adminuser/Рабочий стол/video_screens/"
        img_name = SCREEN_NAME + "_{}.png".format(img_counter)
        cv.imwrite(folder_path+img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv.destroyAllWindows()
