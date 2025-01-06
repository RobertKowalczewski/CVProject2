import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_red_hough(board):
    img_hsv=cv2.cvtColor(board, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lb = 100
    lower_red = np.array([0,lb,lb])
    upper_red = np.array([4,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,lb,lb])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    k1 = np.ones((5,5))

    # join my masks
    redmask = mask0+mask1
    redmask = cv2.erode(redmask, k1)
    #redmask = cv2.erode(redmask, k1)
    #redmask = cv2.erode(redmask, k1)
    redmask = cv2.dilate(redmask, k1)
    redmask = cv2.dilate(redmask, k1)
    redmask = cv2.dilate(redmask, k1)

    #redmask = cv2.dilate(redmask, k2)
    #redmask = cv2.dilate(redmask, k2)

    redboard = np.array(board)
    redboard[redmask==0] = 0

    # redboard = cv2.cvtColor(redboard, cv2.COLOR_RGB2GRAY)

    minDist = 30
    param1 = 20 #canie threshold
    param2 = 5 #200 #smaller value-> more false circles
    minRadius = 30
    maxRadius = 40 #10

    circles = cv2.HoughCircles(redmask,cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    return circles,redboard

def detect_white_hough(board):
    epsilon = 190
    white_upper_range = np.array([255,255,255])
    white_lower_range = white_upper_range - epsilon

    k = np.ones((11,11))
    whitemask = cv2.inRange(board, white_lower_range, white_upper_range)
    whitemask = cv2.erode(whitemask, k)
    whitemask = cv2.erode(whitemask, k)
    #whitemask = cv2.dilate(whitemask ,k)
    #whitemask = cv2.dilate(whitemask ,k)
    whitemask = cv2.dilate(whitemask ,k)
    whitemask = cv2.dilate(whitemask ,k)

    whiteboard = np.array(board)
    whiteboard[whitemask==0] = 0
    whiteboard = cv2.cvtColor(whiteboard, cv2.COLOR_RGB2GRAY)

    minDist = 50
    param1 = 30 #500
    param2 = 30 #200 #smaller value-> more false circles
    minRadius = 30
    maxRadius = 40 #10

    circles = cv2.HoughCircles(whiteboard,cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    return circles

def orientation(p, q, r):
    """Return orientation of the triplet (p, q, r)."""
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

def on_segment(p, q, r):
    """Check if point r lies on segment pq."""
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])

def do_lines_intersect(A, B, C, D):
    """Return True if line segments AB and CD intersect."""
    o1 = orientation(A, B, C)
    o2 = orientation(A, B, D)
    o3 = orientation(C, D, A)
    o4 = orientation(C, D, B)

    # General case: Check if the segments straddle each other
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # Special cases: Check collinear cases where points lie on each other
    if o1 == 0 and on_segment(A, B, C):
        return True
    if o2 == 0 and on_segment(A, B, D):
        return True
    if o3 == 0 and on_segment(C, D, A):
        return True
    if o4 == 0 and on_segment(C, D, B):
        return True

    return False
