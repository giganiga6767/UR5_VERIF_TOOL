import numpy as np
import matplotlib.pyplot as plt

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

    plt.savefig("ur5_skeleton.png", dpi=150)
    plt.close(fig)
    print("\n[SUCCESS] Plot saved! Open 'ur5_skeleton.png' in your file manager.")

def transformmatrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forwardkinematics(theta1, theta2, theta3, theta4, theta5, theta6):
    t1 = transformmatrix(theta1, d1, 0, np.pi/2)
    t2 = transformmatrix(theta2, 0, a2, 0)
    t3 = transformmatrix(theta3, 0, a3, 0)
    t4 = transformmatrix(theta4, d4, 0, np.pi/2)
    t5 = transformmatrix(theta5, d5, 0, -np.pi/2)
    t6 = transformmatrix(theta6, d6, 0, 0)
    t0_6 = t1 @ t2 @ t3 @ t4 @ t5 @ t6

    x, y, z = t0_6[0, 3], t0_6[1, 3], t0_6[2, 3]
    rotation = t0_6[0:3, 0:3]

    return x, y, z, rotation


def rot(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def inversekinematics(roll, pitch, yaw, px, py, pz):
    target_pos = np.array([px, py, pz])
    target_R = rot(roll, pitch, yaw)

    # Initial guess (crooked to avoid shoulder/wrist singularities)
    angles = np.array([0.1, np.pi/4, np.pi/4, 0.1, 0.1, 0.1])

    delta = 0.0001
    max_iters = 150

    for _ in range(max_iters):
        x, y, z, curr_R = forwardkinematics(*angles)
        curr_pos = np.array([x, y, z])

        pos_error = target_pos - curr_pos
        R_err = target_R @ curr_R.T
        ori_error = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])

        error = np.concatenate((pos_error, ori_error))
        if np.linalg.norm(error) < 0.0001:
            break

        J = np.zeros((6, 6))
        for i in range(6):
            temp_angles = angles.copy()
            temp_angles[i] += delta
            nx, ny, nz, nR = forwardkinematics(*temp_angles)

            J[0, i] = (nx - x) / delta
            J[1, i] = (ny - y) / delta
            J[2, i] = (nz - z) / delta

            nR_err = nR @ curr_R.T
            J[3, i] = (0.5 * (nR_err[2, 1] - nR_err[1, 2])) / delta
            J[4, i] = (0.5 * (nR_err[0, 2] - nR_err[2, 0])) / delta
            J[5, i] = (0.5 * (nR_err[1, 0] - nR_err[0, 1])) / delta

        
        if abs(np.linalg.det(J)) < 1e-4:
            print("\n[WARNING] Reached singularity! Aborting IK solver.")
            break
            
        d_angles = np.linalg.pinv(J) @ error
        angles += d_angles
    
        
    # Wrap angles to be between -pi and +pi to prevent joint wind-up
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
        
       

    return angles[0], angles[1], angles[2], angles[3], angles[4], angles[5]

def compute_jacobian(th1, th2, th3, th4, th5, th6):
    curr_angles = np.array([th1, th2, th3, th4, th5, th6])
    delta = 0.0001
    x, y, z, R = forwardkinematics(th1, th2, th3, th4, th5, th6)
    J = np.zeros((6, 6))

    for i in range(6):
        temp_angles = np.copy(curr_angles)
        temp_angles[i] = temp_angles[i] + delta
        cx, cy, cz, cR = forwardkinematics(*temp_angles)

        J[0, i] = (cx - x) / delta
        J[1, i] = (cy - y) / delta
        J[2, i] = (cz - z) / delta

        nR_err = cR @ R.T
        J[3, i] = (0.5 * (nR_err[2, 1] - nR_err[1, 2])) / delta
        J[4, i] = (0.5 * (nR_err[0, 2] - nR_err[2, 0])) / delta
        J[5, i] = (0.5 * (nR_err[1, 0] - nR_err[0, 1])) / delta

    return J

while True:
    print("\n" + "="*45)
    print(" UR5 Kinematics Engine - Terminal Verifier")
    print("="*45)
    print(" 1. Forward Kinematics (Angles -> Position)")
    print(" 2. Inverse Kinematics (Position -> Angles)")
    print(" 3. Jacobian Singularity Check")
    print(" 4. Exit")
    print("="*45)

    choice = input("Select an option (1-4): ")

    if choice == '1':
        try:
            th1 = np.radians(float(input("Enter Joint 1 (deg): ")))
            th2 = np.radians(float(input("Enter Joint 2 (deg): ")))
            th3 = np.radians(float(input("Enter Joint 3 (deg): ")))
            th4 = np.radians(float(input("Enter Joint 4 (deg): ")))
            th5 = np.radians(float(input("Enter Joint 5 (deg): ")))
            th6 = np.radians(float(input("Enter Joint 6 (deg): ")))

            x, y, z, rot_mat = forwardkinematics(th1, th2, th3, th4, th5, th6)
            print("\n--- FK Result ---")
            print(f"X: {x:.4f} m | Y: {y:.4f} m | Z: {z:.4f} m")
            print("Rotation Matrix:\n", np.round(rot_mat, 4))

            visualize_robot(th1, th2, th3, th4, th5, th6)
            J_fk = compute_jacobian(th1, th2, th3, th4, th5, th6)
            if abs(np.linalg.det(J_fk)) < 1e-4:
                print("\n[WARNING] This posture is a SINGULARITY!")
        except Exception as e:
            print(f"[ERROR] Invalid input: {e}")

    elif choice == '2':
        try:
            px = float(input("Enter Target X (m): "))
            py = float(input("Enter Target Y (m): "))
            pz = float(input("Enter Target Z (m): "))
            r = np.radians(float(input("Enter Target Roll (deg): ")))
            p = np.radians(float(input("Enter Target Pitch (deg): ")))
            y = np.radians(float(input("Enter Target Yaw (deg): ")))

            print("\nCalculating IK (Numerical Method)...")
            angles = inversekinematics(r, p, y, px, py, pz)

            print(f"Calculated Angles (deg): {np.round(np.degrees(angles), 2)}")

            verif_x, verif_y, verif_z, _ = forwardkinematics(*angles)
            err = np.sqrt((px-verif_x)**2 + (py-verif_y)**2 + (pz-verif_z)**2)
            print(f"Total 3D Positional Error: {err:.4e} meters")

            J_final = compute_jacobian(*angles)
            det_final = np.linalg.det(J_final)
            print(f"Final Pose Jacobian Determinant: {det_final:.4e}")

            if err < 0.001:
                print("\n[SUCCESS] Solution is valid! Plotting...")
                visualize_robot(*angles)
            else:
                print("\n[WARNING] High error. Target likely outside workspace.")
        except Exception as e:
            print(f"[ERROR] Invalid input: {e}")

    elif choice == '3':
        try:
            th1 = np.radians(float(input("Enter Joint 1 (deg): ")))
            th2 = np.radians(float(input("Enter Joint 2 (deg): ")))
            th3 = np.radians(float(input("Enter Joint 3 (deg): ")))
            th4 = np.radians(float(input("Enter Joint 4 (deg): ")))
            th5 = np.radians(float(input("Enter Joint 5 (deg): ")))
            th6 = np.radians(float(input("Enter Joint 6 (deg): ")))

            J = compute_jacobian(th1, th2, th3, th4, th5, th6)
            det_J = np.linalg.det(J)

            print("\n--- 6x6 Jacobian Matrix ---")
            np.set_printoptions(precision=4, suppress=True)
            print(J)
            print(f"\nDeterminant: {det_J:.6e}")

            if abs(det_J) < 1e-4:
                print("[WARNING] Determinant near zero. Robot is in a SINGULARITY.")
            else:
                print("[SAFE] Robot is not singular. Full maneuverability available.")
        except Exception as e:
            print(f"[ERROR] Invalid input: {e}")

    elif choice == '4':
        print("Exiting Kinematics Engine. Goodbye!")
        break
    else:
        print("[ERROR] Please select a valid option (1-4).")
