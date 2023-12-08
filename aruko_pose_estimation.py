import cv2 
import imutils
import pickle
import numpy as np

marker_robot_len = 0.13 # cm
marker_plane_len = 0.22 # cm
# Загрузка параметров калибровки камеры
mtx, dist, rvecs, tvecs = np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
camera_param = [mtx, dist, rvecs, tvecs]
name_list = ['mtx', 'dist', 'rvecs', 'tvecs']
path = '/home/adminuser/Рабочий стол/camera_calibration_data/'
for i in range(len(name_list)):
    with open(path+f'{name_list[i]}.pickle', 'rb') as handle:
        camera_param[i] = pickle.load(handle)

def aruco_detection(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
        parameters=arucoParams)
    
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            # print("[INFO] ArUco marker ID: {}".format(markerID))

        # show the output image
        cv2.imshow("Aruco detedtion", image)
        # cv2.waitKey(0)

def draw_Aruco_marker_ID(image, markerCorner, markerID):
    # extract the marker corners (which are always returned in
    # top-left, top-right, bottom-right, and bottom-left order)
    corners = markerCorner.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners

    # convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    # draw the ArUco marker ID on the image
    cv2.putText(image, str(markerID),
        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 2)
    # print("[INFO] ArUco marker ID: {}".format(markerID))



def pose_estimation(frame, matrix_coefficients, distortion_coefficients, base_id, target_id):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)
    
    # Print marker's ids
    for i, id in zip(range(0, len(ids)),ids):
        draw_Aruco_marker_ID(frame, corners[i], id)

    path_vector = []
    is_target_detected = False
    # If markers are detected
    if len(corners) > 0:
        # Chacke the target presence
        for id in ids:
            if id == target_id[0]:
                is_target_detected = True
        # if no target stop robot
        if is_target_detected == False:
            path_vector = [0., 0., 0.]
            print("No target. Robot stopped")
        else:          
            coors0 = []
            coors2 = []
            rvec0 = []
            tvec0 = []   
            for i, id in zip(range(0, len(ids)),ids):
                
                if id == base_id:
                    # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_robot_len, matrix_coefficients,
                                                                        distortion_coefficients)
                    print("ID=0:")
                    coors0 = get_marker_coors(rvec, tvec)[0]
                    rvec0 = rvec
                    tvec0 = tvec
                    print(coors0)
                elif id == target_id[0]:
                    # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_plane_len, matrix_coefficients,
                                                                        distortion_coefficients)
                    print("ID=1:")
                    coors2 = get_marker_coors(rvec, tvec)[0]
                    print(coors2)               
                else:
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_plane_len, matrix_coefficients,
                                                                        distortion_coefficients)
                    # Draw a square around the markers
                    cv2.aruco.drawDetectedMarkers(frame, corners) 

                    # Draw Axis
                    cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1) 

            if len(rvec0)!=0 and len(tvec0)!=0 and len(coors2)!=0:
                path_vector = get_path_vector(rvec0, tvec0, coors2)
                print("Path_vector: \n", path_vector)
            if len(coors0)!=0 and len(coors2)!=0:
                dist = pow(pow((coors0[0]-coors2[0]),2)+pow((coors0[1]-coors2[1]),2), 0.5)
                print("DIST: ",round(dist[0], 3), "m")       
                if dist <= 0.25:
                    path_vector = [-1., 0., 0.] # change terget code
                    print("ROBOT FINISHED")

    return frame, path_vector

def get_path_vector(base_rvec, base_tvec, target_coors_camera_3d):
    path_vector = np.linalg.inv(make_trans_mat(base_rvec, base_tvec)).dot(target_coors_camera_3d)
    return path_vector

def make_trans_mat(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)  # 3x3 representation of rvec
    R = np.matrix(R).T  # transpose of 3x3 rotation matrix
    transformMatrix = np.zeros((4, 4), dtype=float)  # Transformation matrix
    # Transformation matrix fill operation, matrix should be [R|t,0|1]
    transformMatrix[0:3, 0:3] = R
    transformMatrix[0:3, 3] = tvec
    transformMatrix[3, 3] = 1

    return transformMatrix

def get_marker_coors(rvec, tvec):
    transformMatrix = make_trans_mat(rvec, tvec)
    # коррдинаты маркера в его системе координат
    world_coors = np.array([[0],[0],[0],[1]])
    # на изображении X - красная ось, У - зеленая ось, Z - синяя, 1
    # координаты маркера относительно камеры в 3д
    # координатные оси камеры совпадают с осями маркера, но
    # ось У направлена противоположно
    camera_3d_coors = transformMatrix.dot(world_coors)
    # print("RES:\n", camera_3d_coors)

    P_matr = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])

    # координаты (х,у) в плоскости изображения
    # (0,0) - верхний левый угол, х - влево, у - вниз
    camera_2d_coors = camera_param[0].dot(P_matr.dot(camera_3d_coors))
    # print("RES:\n", camera_2d_coors)
    return camera_3d_coors, camera_2d_coors



if __name__ == '__main__':

    # захват видео потока
    cam = cv2.VideoCapture(0)
 
    while True:
        # получение кадров
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # изменить размер окна
        frame = imutils.resize(frame, width=860)

        # ARUCO DETECTION
        # aruco_detection(frame)
        
        base_id = 0
        target_id = 140
        #POSE ESTIMATION      
        estim_frame, _ = pose_estimation(frame, camera_param[0], camera_param[1], base_id, target_id)
        cv2.imshow("aruko test", estim_frame)

        k = cv2.waitKey(1)
        # выход
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()



