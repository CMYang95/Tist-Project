import numpy as np
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point
    c = np.array(c)  # Third point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_speed(distance, time_delta):
    if time_delta > 0:
        return distance / time_delta
    return 0

def calculate_acceleration(current_speed, previous_speed, time_delta):
    if time_delta > 0:
        return (current_speed - previous_speed) / time_delta
    return 0

def calculate_force(mass, acceleration):
    return mass * acceleration