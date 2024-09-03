import math
from collections import OrderedDict

import cv2 as cv
import threading
import queue
import time

import numpy as np


class Visualizer:
    CAM_SERVER_ADDRESS = "192.168.1.34"
    CAM_SERVER_PORT = 12345
    def __init__(self):
        # Setup Camera Stream
        self.cap = cv.VideoCapture(f"tcp://{self.CAM_SERVER_ADDRESS}:{self.CAM_SERVER_PORT}")
        if not self.cap.isOpened():
            raise IOError(f"Failed to open camera stream at {self.CAM_SERVER_ADDRESS}:{self.CAM_SERVER_PORT}")

        self.stop_visualizer = threading.Event()
        self.visualizer_queue = queue.Queue(maxsize=1)  # We will make 1 frame available

        self.skipped_frames = 0
        self.frames_per_second = 0

        self.visualizer_thread = threading.Thread(target=self._run)
        self.visualizer_thread.start()

    def _run(self):
        # Camera loop
        last_time = time.time()
        while not self.stop_visualizer.is_set():
            ret, frame = self.cap.read()
            # cv.imshow('test', frame)
            # if cv.waitKey(1) == ord('q'):
            #     break

            if self.visualizer_queue.full():
                self.skipped_frames += 1
                # Skipping this frame will leave us with 1 frame
                self.visualizer_queue.get()
            else:
                self.skipped_frames = 0

            # Add or replace the frame
            self.visualizer_queue.put(frame)


            # Calculate metrics
            current_time = time.time()
            self.frames_per_second = current_time - last_time
            last_time = current_time

        self.cap.release()

    def __del__(self):
        # Does releasing twice cause a problem?
        self.stop_visualizer.set()

    def get_frame(self):
        return self.visualizer_queue.get()


    # def detect_motion(self, frame1, frame2):
    #     hsv = np.zeros_like(frame1)
    #     prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    #     #
    #     hsv[..., 1] = 255
    #     next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    #     flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #     hsv[..., 0] = ang * 180 / np.pi / 2
    #     hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    #     bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    #     cv.imshow('frame2', bgr)
    #     cv.waitKey(1)
    #     # prvs = next


    def feature_match_frames(self, frame1, frame2):
        """
        :param frame1:
        :param frame2:
        :return:
        # Two options:
        # * Canny edge detection - PROBABLY NOT THE BEST IDEA
        # * Feature detection (SIFT detection)
        #
        # Feature detection is probably better. We will implement it as follows:
        # - We will compare the features in two sequential frames to determine in which direction objects are
        # moving in the camera's frame of reference.
        # - Using this we can avoid direct collisions with walls and other objects
        # - Using this we can also feedforward the motion estimates into the kalman filter to be less dependent on the IMU

        """

        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(frame1, None)
        kp2, des2 = sift.detectAndCompute(frame2, None)
        # BFMatcher with default params
        bf = cv.BFMatcher()

        if des1 is None or des2 is None:
            return [0, 0, 0]

        #print(f"{len(des1)} - {des1}")
        #print(f" {len(des2)} - {des2}")
        matches = bf.knnMatch(des1, des2, k=2)
        # print(len(matches))
        # print(matches)
        # Apply ratio test
        good = []
        for m in matches:
            if len(m) == 2:
                i, j = m
                if i.distance < 0.3 * j.distance:
                    good.append([i])#([(m, n)])
        # The queryIdx will always be the same between m.queryIdx and n.queryIdx
        #     print(f"{m.trainIdx} - {n.trainIdx}")
        #     print(f"{m.queryIdx} - {n.queryIdx}")
        #     print(f"{m.imgIdx} - {n.imgIdx}")
        #     print("--------")

        temp = cv.drawMatchesKnn(frame1, kp1, frame2, kp2, good, None,
                                  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("temp", temp)
        cv.waitKey(1)

        #z_motion_vector_field = lambda x, y: (math.acos(y/math.sqrt(x**2 + y**2)), math.asin(x/math.sqrt(x**2 + y**2)))
        z_motion_vector_field = lambda x, y: (x / math.sqrt(x ** 2 + y ** 2), y / math.sqrt(x ** 2 + y ** 2))

        # Compare movement between all good sets to determine if there is net-motion
        average_motion = [0, 0, 0]
        divergence_data = {}
        for m in good:
            p1 = kp1[m[0].queryIdx].pt
            p2 = kp2[m[0].trainIdx].pt

            # We will look at the pixel delta between matching pairs
            dx = p1[0] - p2[0]  # Net motion in X (vectors move in a particular direction)
            dy = p1[1] - p2[1]  # Net motion in Y (vectors move in a particular direction)
            average_motion[0] += dx
            average_motion[1] += dy

            # Z-axis algorithm
            # 1) For each SIFT keypoint - track the change in position as a vector
            # 2) Take the projection of this displacement vector onto a vector field centered around (0,0) with unit vectors and positive divergence
            # 3) Since SIFT keypoints are not guaranteed to be evenly distributed across the image, segment the image like a pizza or pie into sections that take the average values of the adjacent displacement
            displacement_vector = (dx, dy) #/math.sqrt(dx**2 + dy**2)
            basis_vector = z_motion_vector_field(p1[0], p1[1])
            dot_product = displacement_vector[0]*basis_vector[0] + displacement_vector[1]*basis_vector[1]

            # Save dot_product result based on the key (angle of the SIFT point)
            rad_angle = math.atan2(p1[1], p1[0])
            divergence_data[rad_angle] = dot_product



            #dz =  # Net motion in z (vectors diverge)
            # print(f"{p1}, {p2} ")#= {p2 - p1}") # NO WORK

        if len(good) > 0:
            # X-Y Motion is average displacement in pixels
            average_motion[0] /= len(good)
            average_motion[1] /= len(good)

            # Z Motion is ...
            sorted_data = dict(sorted(divergence_data.items())).items()
            last_angle, last_data = next(iter(sorted_data))
            first_angle = last_angle
            first_data = last_data
            for angle, value in sorted_data:
                delta_angle = angle - last_angle
                average_motion[2] = (last_data + value)/2 * delta_angle/(2*math.pi)

                last_data = value
                last_angle = angle
            average_motion[2] += (last_data + first_data)/2 * (first_angle + 2*math.pi - last_angle)/(2*math.pi)
            average_motion[2] /= len(sorted_data)

            print(average_motion)
        return average_motion


        # # Calculate center of match and remove each keypoint/descriptor pair for another match attempt
        # if len(good) > 0:
        #     # centroidList = []
        #     centroid = np.array([0., 0.])  # x,y
        #     resRatio = 0
        #     resRatioList = []
        #     idPair = []
        #     for dmatch in good:
        #         id1 = dmatch[0].queryIdx  # Id of des1/kp1
        #         id2 = dmatch[0].trainIdx
        #         idPair.append((id1, id2))
        #         centroid += kp2[id2].pt
        #
        #     for i in range(len(idPair) - 1):
        #         # Compare distance between two different pairs (4 points)
        #         # Query points
        #         ptA = idPair[i][0]
        #         ptB = idPair[i + 1][0]
        #         queryDistance = cv.norm(kp1[ptA].pt, kp1[ptB].pt, cv.NORM_L2)
        #         ptA = idPair[i][1]
        #         ptB = idPair[i + 1][1]
        #         trainDistance = cv.norm(kp2[ptA].pt, kp2[ptB].pt, cv.NORM_L2)
        #         # Example: if target/train resolution is twice the resolution of query
        #         # Then this value will be 2. If there is a large discrepencancy between values
        #         # we can eliminate further outliers and determine the size of the character portrait
        #         # From that we can infer the size of other UI elements
        #         if (queryDistance != 0 or trainDistance != 0):
        #             resRatioList.append(trainDistance / queryDistance)
        #
        #     if (len(resRatioList) > 0):
        #         resRatio = sum(resRatioList) / len(resRatioList)
        #     else:
        #         resRatio = 1
        #     centroid /= len(good)
        #     centroid[0] = math.floor(centroid[0])
        #     centroid[1] = math.floor(centroid[1])
        #     # Determine size of Portrait based on query image (resolution independent)
        #     # size is the expected size of the character portrait within the frame
        #     size = (math.floor(resRatio * frame1.shape[1]), math.floor(resRatio * frame1.shape[0]))
        #
        #     topLeft = (math.floor(centroid[0] - size[0] / 2), math.floor(centroid[1] - size[1] / 2))
        #     bottomRight = (math.floor(centroid[0] + size[0] / 2), math.floor(centroid[1] + size[1] / 2))
        #     # Save position of character portrait to return later
        #     retArray.append((topLeft, bottomRight))
        #
        #     # DEBUG/Drawing
        #     color = (255, 0, 0)
        #     thick = 2
        #     temp = cv.rectangle(frame2, topLeft, bottomRight, color, thick)
        #     # cv2.imshow("temp", temp)
        #     # cv2.waitKey(0)
        #
        #     # Remove any keypoints within the bounding box
        #     newKeypoint = []
        #     newDescriptor = np.array([])
        #     i = 0
        #     while i < len(kp2):
        #         keypoint = kp2[i]
        #         descriptor = des2[i]
        #         if (keypoint.pt[0] > centroid[0] + size[0] / 2 or keypoint.pt[0] < centroid[0] - size[0] / 2 or
        #                 keypoint.pt[1] < centroid[1] - size[1] / 2 or keypoint.pt[1] > centroid[1] + size[1] / 2):
        #             # newKeypoint.append(keypoint)
        #             # newDescriptor = np.append(newDescriptor, descriptor)
        #             i += 1
        #             pass
        #         else:
        #             del kp2[i]
        #             des2 = np.delete(des2, i, 0)
        #             # Inefficient array element deletion
