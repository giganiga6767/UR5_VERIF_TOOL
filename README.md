# UR5 6-DOF Kinematics Engine

A complete kinematics solver for the Universal Robots UR5 arm, built entirely using numpy and matplotlib.

This project bypasses standard pre built robotics libraries to demonstrate an understanding of transformation matrices, DH parameters, and Jacobian based numerical solvers.

Key Features:

* **Custom Denavit-Hartenberg (DH) Modeling:** Implements a custom, forward-facing axis convention for the physical links (a2, a3), requiring manual derivation of the transformation matrices.
* **Forward Kinematics (FK):** Accurately computes the 3D end-effector position and rotation matrix based on 6 joint angles.
* **Numerical Inverse Kinematics (IK):** Uses a finite-difference Jacobian matrix and pseudo-inverse methodology to solve for joint angles. 
* **Singularity Detection:** Calculates the determinant of the 6x6 Jacobian matrix in real-time to warn when the robot loses a degree of freedom.
* **3D Skeleton Visualization:** Includes a matplotlib visualizer to plot the physical mathematical joints of the robot.(a static 3d plot to be precise)

## Tech Stack
* **Language:** Python 
* **Libraries:** numpy, matplotlib

## Usage

Run the main script to access the terminal-based Verification Tool:
```bash
python ur5.py
##Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/UR5_VERIF_TOOL.git](https://github.com/yourusername/UR5_VERIF_TOOL.git)
cd UR5_VERIF_TOOL
pip install -r requirements.txt
