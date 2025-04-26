import matplotlib.pyplot as plt

def plot_joint_angles(timestamps, angles):
    plt.figure()
    plt.plot(timestamps, angles[0], color='red', label="Elbow")
    plt.plot(timestamps, angles[1], color='green', label="Shoulder")
    plt.plot(timestamps, angles[2], color='blue', label="Knee")
    plt.plot(timestamps, angles[3], color='purple', label="Hip")
    plt.title('Angles of Joints')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend(loc='lower right')
    plt.show()

def plot_speed_force(data):
    timestamps = [item[0] for item in data]
    speeds = [item[1] for item in data]
    forces = [item[2] for item in data]
    accelerations = [item[3] for item in data]

    plt.figure()
    plt.plot(timestamps, speeds, color='b', label='Speed')
    plt.plot(timestamps, forces, color='r', label='Force')
    plt.plot(timestamps, accelerations, color='g', label='Acceleration')
    plt.title('Speed, Force, and Acceleration Over Time')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Value')
    plt.legend(loc='lower right')
    plt.show()