import math

def is_open(finger_tip, finger_mcp):
    return finger_tip[1] < finger_mcp[1]

def count_fingers(kp):
    # Keypoint indices MediaPipe-style:
    # Thumb: [1,2,3,4]
    # Index: [5,6,7,8]
    # Middle: [9,10,11,12]
    # Ring: [13,14,15,16]
    # Pinky: [17,18,19,20]

    thumb = kp[4][0] > kp[2][0]   # Pulgar abierto (horizontal)
    index = is_open(kp[8], kp[5])
    middle = is_open(kp[12], kp[9])
    ring = is_open(kp[16], kp[13])
    pinky = is_open(kp[20], kp[17])

    return sum([thumb, index, middle, ring, pinky])
