import numpy as np

def decision_engine(direction, objects, frame_shape):

    h, w = frame_shape[:2]

    center_zone_left = w // 3
    center_zone_right = 2 * w // 3

    obstacle_in_front = False

    # -----------------------------------
    # CHECK OBSTACLES IN CENTER ZONE
    # -----------------------------------
    for obj in objects:

        x, y, bw, bh = obj["box"]

        obj_center_x = x + bw // 2

        # if object is in front zone
        if center_zone_left < obj_center_x < center_zone_right:
            obstacle_in_front = True

    # -----------------------------------
    # DECISION RULES
    # -----------------------------------
    if obstacle_in_front:
        return "STOP"

    return direction