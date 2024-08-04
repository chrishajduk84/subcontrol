import socket
import cv2 as cv


class Navigator:
    """ Responsible for managing various resources used for navigation and combining the resources together
    to execute the desired mission plan.
    The resources that are managed include:
        * Cameras/visuals (from onboard camera)
        * Position (depth/pitch/roll/yaw) estimates (from secondary PCB)
        * Mission Plan inputs from (communicator?)
    """

    def __init__(self):
        #################################
        # Initialize resource paths
        #################################
        cap = cv.VideoCapture("tcp://192.168.1.34:12345")
        print(cap.isOpened())
        # img = cv.imread("C:\\Users\\chris\\Pictures\\PassPhoto.png")
        # cv.imshow("D", img)
        # cv.waitey(0)
        # Camera
        while True:
            ret, frame = cap.read()
            cv.imshow('test', frame)
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        # self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.camera_socket.connect(("192.168.1.34", 12345))
        # while True:
        #     data = self.camera_socket.recv(4096)
        #     print(data)
        # s.close()



n = Navigator()