import csv
import socket
import time

from visualizer import Visualizer



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
        self.vis = Visualizer()
        # img = cv.imread("C:\\Users\\chris\\Pictures\\PassPhoto.png")
        # cv.imshow("D", img)
        # cv.waitey(0)

        # self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.camera_socket.connect(("192.168.1.34", 12345))
        # while True:
        #     data = self.camera_socket.recv(4096)
        #     print(data)
        # s.close()
        f = open("test.csv", 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=["time", "x", "y", "z"])
        writer.writeheader()
        while True:
            frame1 = self.vis.get_frame()
            frame2 = self.vis.get_frame()

            motion = self.vis.feature_match_frames(frame1, frame2)
            line_to_write = {"time":str(time.time()), "x":str(motion[0]), "y":str(motion[1]), "z":str(motion[2])}
            writer.writerow(line_to_write)
            # self.vis.detect_motion(frame1, frame2)


n = Navigator()