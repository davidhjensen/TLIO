import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def verify_tlio_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ts = f['ts'][:]
        accel = f['accel_dcalibrated'][:] # Gravity should be GONE from this
        gyro = f['gyro_dcalibrated'][:]
        gt_p = f['vio_p'][:]
        gt_v = f['vio_v'][:]
        gt_q = f['vio_q_wxyz'][:] # [w, x, y, z]

    dt = np.mean(np.diff(ts))
    num_samples = len(ts)

    # --- 1. Simple Double Integration ---
    # We use the GT orientation to rotate local accel back to world frame
    # Since accel_dcalibrated has gravity removed, we don't add 9.81 back.
    est_v = np.zeros((num_samples, 3))
    est_p = np.zeros((num_samples, 3))
    
    # Initialize with GT starting values
    est_v[0] = gt_v[0]
    est_p[0] = gt_p[0]

    print("Integrating IMU paths...")
    for i in range(1, num_samples):
        # 1. Get orientation at this step (WXYZ -> XYZW for scipy)
        q_curr = R.from_quat([gt_q[i, 1], gt_q[i, 2], gt_q[i, 3], gt_q[i, 0]])
        
        # 2. Rotate local calibrated accel to World Frame
        accel_world = q_curr.apply(accel[i])
        
        # 3. Integrate Velocity: v = v + a*dt
        est_v[i] = est_v[i-1] + accel_world * dt
        
        # 4. Integrate Position: p = p + v*dt
        est_p[i] = est_p[i-1] + est_v[i] * dt

    # --- 2. Visualization ---
    fig = plt.figure(figsize=(12, 8))
    
    # Plot 3D Trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gt_p[:, 0], gt_p[:, 1], gt_p[:, 2], label='Ground Truth (VIO)', color='blue')
    ax1.plot(est_p[:, 0], est_p[:, 1], est_p[:, 2], label='IMU Integrated', color='red', linestyle='--')
    ax1.set_title("3D Path Comparison")
    ax1.legend()

    # Plot Acceleration to check gravity removal
    ax2 = fig.add_subplot(222)
    ax2.plot(ts, accel[:, 0], label='Acc X', alpha=0.7)
    ax2.plot(ts, accel[:, 1], label='Acc Y', alpha=0.7)
    ax2.plot(ts, accel[:, 2], label='Acc Z', alpha=0.7)
    ax2.set_title("Calibrated Accel (Should be ~0 when static)")
    ax2.legend()

    # Plot Velocity comparison
    ax3 = fig.add_subplot(224)
    ax3.plot(ts, gt_v[:, 0], label='GT Vel X', color='black')
    ax3.plot(ts, est_v[:, 0], label='Est Vel X', color='red', linestyle='--')
    ax3.set_title("Velocity X-Axis")
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_tlio_data("C:\\Users\\dhjen\\Desktop\\ETH Zurich\\school\\3d_vision\\TLIO\\data\\Dataset\\seq10\\data.hdf5")