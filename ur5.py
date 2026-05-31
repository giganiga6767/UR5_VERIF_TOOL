import numpy as np
import time
import matplotlib.pyplot as plt

theta1 = None
theta2 = None
theta3 = None
theta4 = None
theta5 = None
theta6 = None
d1 = 0.089159
a2 = 0.425
a3 = 0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

def visualize_robot(j1, j2, j3, j4, j5, j6):
    t1 = transformmatrix(j1, d1, 0, np.pi/2)
    t2 = transformmatrix(j2, 0, a2, 0)
    t3 = transformmatrix(j3, 0, a3, 0)
    t4 = transformmatrix(j4, d4, 0, np.pi/2)
    t5 = transformmatrix(j5, d5, 0, -np.pi/2)
    t6 = transformmatrix(j6, d6, 0, 0)

    p0 = np.array([0, 0, 0, 1])
    p1 = t1 @ p0
    p2 = t1 @ t2 @ p0
    p3 = t1 @ t2 @ t3 @ p0
    p4 = t1 @ t2 @ t3 @ t4 @ p0
    p5 = t1 @ t2 @ t3 @ t4 @ t5 @ p0
    p6 = t1 @ t2 @ t3 @ t4 @ t5 @ t6 @ p0

    points = np.array([p0, p1, p2, p3, p4, p5, p6])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'o-', color='steelblue', linewidth=3, markersize=6, label='Links')
    ax.plot([x[0]], [y[0]], [z[0]], 'go', markersize=8, label='Base')
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'ro', markersize=8, label='End Effector')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UR5 Skeleton Plot')
    ax.legend()

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ---> THE LINUX OFFLINE FIX <---
    plt.savefig("ur5_skeleton.png", dpi=150)
    plt.close(fig)
    print("\n[SUCCESS] Plot saved! Open 'ur5_skeleton.png' in your file manager to see it.")

def transformmatrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])         

def forwardkinematics(theta1, theta2, theta3, theta4, theta5, theta6):
    t1 = transformmatrix(theta1, d1, 0 , np.pi/2)
    t2 = transformmatrix(theta2, 0, a2 , 0)
    t3 = transformmatrix(theta3, 0, a3, 0)
    t4 = transformmatrix(theta4, d4, 0 , np.pi/2)
    t5 = transformmatrix(theta5, d5, 0 , -np.pi/2)
    t6 = transformmatrix(theta6, d6, 0 , 0)
    t0_6 = t1@t2@t3@t4@t5@t6
    x = t0_6[0,3]
    y = t0_6[1,3]
    z = t0_6[2,3]
    rotation= t0_6[0:3, 0:3]
    return x, y, z, rotation

def rot(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx
    
def inversekinematics(angle1, angle2, angle3, px, py ,pz):
    R = rot(angle1, angle2, angle3)
    r13, r23, r33 = R[0,2], R[1,2], R[2,2]   
    r11, r12 = R[0,0], R[0,1]
    r21, r22 = R[1,0], R[1,1]
    r31, r32 = R[2,0], R[2,1]

    wc_x = px - d6 * r13
    wc_y = py - d6 * r23   
    wc_z = pz - d6 * r33   

    theta1 = np.arctan2(wc_y, wc_x)   

    c1, s1 = np.cos(theta1), np.sin(theta1)

    c5 = r13*s1 - r23*c1
    s5 = np.sqrt(1 - c5**2)
    theta5 = np.arctan2(s5, c5)

    s6 = (-r12*s1 + r22*c1) / s5
    c6 = ( r11*s1 - r21*c1) / s5
    theta6 = np.arctan2(s6, c6)

    c234 = r31 * (-1/s5) if s5 != 0 else 0
    s234 = r32 / s5 if s5 != 0 else 0

    Kc = wc_x*c1 + wc_y*s1    
    Ks = wc_z - d1             

    cos3 = (Kc**2 + Ks**2 - a2**2 - a3**2) / (2*a2*a3)
    cos3 = np.clip(cos3, -1, 1)
    theta3 = np.arctan2(np.sqrt(1 - cos3**2), cos3)
    
    theta2 = np.arctan2(
        Ks*(a2 + a3*np.cos(theta3)) - Kc*a3*np.sin(theta3),
        Kc*(a2 + a3*np.cos(theta3)) + Ks*a3*np.sin(theta3)
    )
    
    theta234 = np.arctan2(s234, c234)
    theta4 = theta234 - theta2 - theta3

    return theta1, theta2, theta3, theta4, theta5, theta6

while(True):
    print("\n" + "="*40)
    print(" UR5 Kinematics Verification Tool")
    print("="*40)
    choice = input("choose FK or IK verification(1 for FK, 2 for IK, 3 to exit): ")
    if choice == '1':
        try:
            theta1 = np.radians(float(input("enter 1st angle(degrees) ")))
            theta2 = np.radians(float(input("enter 2nd angle(degrees) ")))
            theta3 = np.radians(float(input("enter 3rd angle(degrees) ")))
            theta4 = np.radians(float(input("enter 4th angle(degrees) ")))
            theta5 = np.radians(float(input("enter 5th angle(degrees) ")))
            theta6 = np.radians(float(input("enter 6th angle(degrees) ")))
            print("\nthe final orientation is:")
            print("\n" + "="*40)
            answer = forwardkinematics(theta1, theta2, theta3, theta4, theta5, theta6)
            print(f"x:{answer[0]} m")
            print(f"y:{answer[1]} m")
            print(f"z:{answer[2]} m")
            print("\n ROTATION MATRIX\n")
            print(answer[3])
            visualize_robot(theta1, theta2, theta3, theta4, theta5, theta6)
        except Exception as e:
            print(f"error.... try again\n{e}")
    elif choice =='2' :
        try:
            px = float(input("enter target X (m): "))
            py = float(input("enter target Y (m): "))
            pz = float(input("entertarget Z (m): "))
            roll = np.radians(float(input("enter target roll (degrees): ")))
            pitch = np.radians(float(input("enter target pitch (degrees): ")))
            yaw = np.radians(float(input("enter target yaw (degrees): ")))
            
            print("\nCalculating IK...")
            angles = inversekinematics(roll, pitch, yaw, px, py, pz)
            
            print("\nCalculated angles (degrees):")
            print(f"[{np.degrees(angles[0])}, {np.degrees(angles[1])}, {np.degrees(angles[2])}, {np.degrees(angles[3])}, {np.degrees(angles[4])}, {np.degrees(angles[5])}]")
            
            print("\nVerifying with FK...")
            verif= forwardkinematics(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5])
            
            error_x = px - verif[0]
            error_y = py - verif[1]
            error_z = pz - verif[2]
            total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            print(f"Total 3D Positional Error: {total_error:.4e} meters")
            
            if total_error < 0.0001:
                print("\n[SUCCESS] IK Solution matches the target! Verification Passed.")
                visualize_robot(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5])
            else:
                print("\n[WARNING] High error detected. Target may be outside physical workspace.")
                
        except ValueError:
            print("[Error] Invalid input. Please enter numbers only.\n")

    elif choice == '3':
        print("Exiting Kinematics Tool...")
        break
        
    else:
        print("[Error] Invalid choice. Enter 1, 2, or 3.")