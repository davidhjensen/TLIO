import numpy as np
import h5py
import os
import glob
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

def generate_tlio_hdf5(imu_path, gt_path, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 1. Load Raw Data
    # IMU format: [ts_ns, gx, gy, gz, ax, ay, az]
    print("Reading IMU...")
    imu_raw_data = np.genfromtxt(imu_path, delimiter=',', skip_header=1)
    # GT format: [ts_s, px, py, pz, qx, qy, qz, qw]
    print("Reading GT...")
    gt_raw_data = np.genfromtxt(gt_path, delimiter=' ', skip_header=1)

    # Convert timestamps (Assuming 19-digit nanoseconds from your example)
    imu_ts = imu_raw_data[:, 0] / 1e9 
    gt_ts = gt_raw_data[:, 0]

    # Sync time window to 1000Hz grid
    t_start = max(imu_ts[0], gt_ts[0])
    t_end = min(imu_ts[-1], gt_ts[-1])
    ts = np.arange(t_start, t_end, 0.001)

    # 2. Interpolation
    def interp(old_t, old_val):
        return interp1d(old_t, old_val, axis=0, fill_value="extrapolate")(ts)

    accel_raw = interp(imu_ts, imu_raw_data[:, 4:7])
    gyro_raw = interp(imu_ts, imu_raw_data[:, 1:4])
    vio_p = interp(gt_ts, gt_raw_data[:, 1:4])
    q_xyzw = interp(gt_ts, gt_raw_data[:, 4:8])

    # 3. Calculations
    print("Processing physics (Gravity & Velocity)...")
    q_objs = R.from_quat(q_xyzw)
    g_world = np.array([0, 0, -9.81])
    
    # Gravity compensation for 'accel_dcalibrated'
    accel = accel_raw - q_objs.inv().apply(g_world)
    
    # Gyro calibration (assumed raw for now, or subtract bias if known)
    gyro = gyro_raw 

    # Convert to WXYZ for HDF5
    vio_q_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0:3]])
    
    # Numerical velocity (v = dp/dt)
    vio_v = np.diff(vio_p, axis=0, prepend=[vio_p[0]]) / 0.001

    # Placeholder for filter/calibration states (Required by TLIO schema)
    # integration_q and filter_q are usually the same as ground truth for training data
    integration_q_wxyz = vio_q_wxyz
    filter_q_wxyz_interp = vio_q_wxyz
    
    # Calibration States [t, acc_scale_inv (9), gyr_scale_inv (9), gyro_g_sense (9), b_acc (3), b_gyr (3)]
    # Total columns: 1 (time) + 9 + 9 + 9 + 3 + 3 = 34 columns
    offline_calib = np.zeros((len(ts), 34))
    offline_calib[:, 0] = ts  # Timestamp

    # Set scale invariants to Identity (1.0 on diagonals of the 3x3s)
    # Accel scale inv (cols 1, 5, 9) and Gyro scale inv (cols 10, 14, 18)
    offline_calib[:, [1, 5, 9, 10, 14, 18]] = 1.0

    # If you have static bias estimates, you can set them here:
    # b_acc starts at index 28 (cols 28, 29, 30)
    # b_gyr starts at index 31 (cols 31, 32, 33)
    offline_calib[:, 28:31] = 0.0 # Accel bias
    offline_calib[:, 31:34] = 0.0 # Gyro bias

    # 4. HDF5 Writing
    h5_path = os.path.join(outdir, "data.hdf5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ts", data=ts)
        f.create_dataset("accel_dcalibrated", data=accel)
        f.create_dataset("gyro_dcalibrated", data=gyro)
        f.create_dataset("accel_raw", data=accel_raw)
        f.create_dataset("gyro_raw", data=gyro_raw)
        f.create_dataset("vio_q_wxyz", data=vio_q_wxyz)
        f.create_dataset("vio_p", data=vio_p)
        f.create_dataset("vio_v", data=vio_v)
        f.create_dataset("integration_q_wxyz", data=integration_q_wxyz)
        f.create_dataset("filter_q_wxyz", data=filter_q_wxyz_interp)
        f.create_dataset("offline_calib", data=offline_calib)
        
    print(f"File data.hdf5 written to {outdir}")

if __name__ == "__main__":

    # Define the root directory containing your raw data folders
    root_dir = r"C:\Users\dhjen\Desktop\ETH Zurich\school\3d_vision\TLIO\EDS"
    output_base = r"C:\Users\dhjen\Desktop\ETH Zurich\school\3d_vision\TLIO\data\Dataset"

    # Use glob to find all subdirectories in the EDS folder
    # This looks for any folder inside root_dir
    folders = glob.glob(os.path.join(root_dir, "*/"))

    for folder_path in sorted(folders):
        # Get the folder name (e.g., '01_peanuts_light')
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
        # Extract the sequence number for the output (e.g., 'seq01')
        # This assumes the folder starts with a 2-digit number
        seq_num = folder_name.split('_')[0]
        output_name = f"seq{seq_num}"
        
        # Construct the full paths for the required files
        imu_path = os.path.join(folder_path, "imu.csv")
        gt_path = os.path.join(folder_path, "stamped_groundtruth.txt")
        output_path = os.path.join(output_base, output_name)
        
        # Ensure the output directory exists
        os.makedirs(output_base, exist_ok=True)

        print(f"Processing: {folder_name} -> {output_name}")
        
        # Call your function
        try:
            generate_tlio_hdf5(imu_path, gt_path, output_path)
        except Exception as e:
            print(f"Failed to process {folder_name}: {e}")