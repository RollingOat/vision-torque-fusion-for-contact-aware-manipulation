import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
# ------------------------
# Utilities
# ------------------------

def compute_drot(q_curr, q_next):
    if np.dot(q_curr, q_next) < 0:
        q_next = -q_next

    r_curr = R.from_quat(q_curr)
    r_next = R.from_quat(q_next)

    r_diff = r_next * r_curr.inv()
    d_rot = r_diff.as_rotvec()
    return d_rot

def skew(v):
    """Skew-symmetric matrix of a 3D vector."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])


# ------------------------
# Main estimation function
# ------------------------

def estimate_payload_with_bias(data, g=9.81):
    """
    Estimate payload mass, CoM, force bias, and torque bias.

    Parameters
    ----------
    data : list of dict
        Each element must contain:
        {
          'R_ws': 3x3 rotation matrix (world -> sensor),
          'force':  length-3 ndarray (sensor frame),
          'torque': length-3 ndarray (sensor frame)
        }

    g : float
        Gravity magnitude (default 9.81)

    Returns
    -------
    result : dict
        {
          'mass': float,
          'com': ndarray (3,),
          'force_bias': ndarray (3,),
          'torque_bias': ndarray (3,),
          'condition_number': float
        }
    """

    H_blocks = []
    Y_blocks = []

    g_world = np.array([0.0, 0.0, -g])

    for sample in data:
        R_sw = sample['R_sw']
        f = sample['force']
        tau = sample['torque']

        # gravity expressed in sensor frame
        g_s = R_sw.T @ g_world   # 3x1

        # Per-sample regressor (6 x 10)
        Hi = np.block([
            # force equation
            [g_s.reshape(3, 1),      np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
            # torque equation
            [np.zeros((3, 1)), -skew(g_s),             np.zeros((3, 3)), np.eye(3)]
        ])

        Yi = np.hstack([f, tau])

        H_blocks.append(Hi)
        Y_blocks.append(Yi)

    H = np.vstack(H_blocks)
    Y = np.hstack(Y_blocks)

    # Solve least squares
    theta, residuals, rank, s = np.linalg.lstsq(H, Y, rcond=None)
    print("the residuals of least squares:", residuals)
    # Extract parameters
    m = theta[0]
    ell = theta[1:4]            # first moment = m * r
    bf = theta[4:7]
    bt = theta[7:10]

    if m <= 0:
        raise ValueError("Estimated mass is non-positive. Check data quality.")

    r = ell / m

    cond_number = np.linalg.cond(H)

    return {
        'mass': m,
        'com': r,
        'force_bias': bf,
        'torque_bias': bt,
        'condition_number': cond_number
    }


def optimize_payload_with_bias(data, initial_guess, g=9.81):
    """
    Refine payload estimation using non-linear optimization.

    Parameters
    ----------
    data : list of dict
        Each element must contain:
        {
          'R_sw': 3x3 rotation matrix (sensor -> world),
          'force':  length-3 ndarray (sensor frame),
          'torque': length-3 ndarray (sensor frame)
        }

    initial_guess : dict
        Output from estimate_payload_with_bias() to initialize optimization.

    g : float
        Gravity magnitude (default 9.81)

    Returns
    -------
    result : dict
        Same format as estimate_payload_with_bias(), but refined.
    """
    

    def residuals(params):
        m = params[0]
        r = params[1:4]
        bf = params[4:7]
        bt = params[7:10]

        res = []
        for sample in data:
            R_sw = sample['R_sw'] # 3 by 3 rotation from sensor to world
            f_meas = sample['force'] # n by 3
            tau_meas = sample['torque'] # n by 3

            g_s = R_sw.T @ np.array([0.0, 0.0, -g])
            f_pred = m * g_s + bf
            tau_pred = np.cross(r, m * g_s) + bt

            res.extend((f_meas - f_pred).flatten())
            res.extend((tau_meas - tau_pred).flatten())

        return np.array(res)

    x0 = np.hstack([
        initial_guess['mass'],
        initial_guess['com'],
        initial_guess['force_bias'],
        initial_guess['torque_bias']
    ])

    result = least_squares(residuals, x0, verbose=2)
    m_opt = result.x[0]
    r_opt = result.x[1:4]
    bf_opt = result.x[4:7]
    bt_opt = result.x[7:10]

    if m_opt <= 0:
        raise ValueError("Optimized mass is non-positive. Check data quality.")

    return {
        'mass': m_opt,
        'com': r_opt,
        'force_bias': bf_opt,
        'torque_bias': bt_opt,
        'optimization_success': result.success,
        'optimization_message': result.message
    }

def optimize_payload(data, initial_guess, g=9.81):
    """
    Refine payload estimation using non-linear optimization.

    Parameters
    ----------
    data : list of dict
        Each element must contain:
        {
          'R_sw': 3x3 rotation matrix (sensor -> world),
          'force':  length-3 ndarray (sensor frame),
          'torque': length-3 ndarray (sensor frame)
        }

    initial_guess : dict
        Output from estimate_payload_with_bias() to initialize optimization.

    g : float
        Gravity magnitude (default 9.81)

    Returns
    -------
    result : dict
        Same format as estimate_payload_with_bias(), but refined.
    """
    

    def residuals(params):
        m = params[0]
        r = params[1:4]

        res = []
        for sample in data:
            R_sw = sample['R_sw'] # 3 by 3 rotation from sensor to world
            f_meas = sample['force'] # n by 3
            tau_meas = sample['torque'] # n by 3

            g_s = R_sw.T @ np.array([0.0, 0.0, -g])
            f_pred = m * g_s
            tau_pred = np.cross(r, m * g_s)

            res.extend((f_meas - f_pred).flatten())
            res.extend((tau_meas - tau_pred).flatten())

        return np.array(res)

    x0 = np.hstack([
        initial_guess['mass'],
        initial_guess['com'],
        initial_guess['force_bias'],
        initial_guess['torque_bias']
    ])

    result = least_squares(residuals, x0, verbose=2)
    m_opt = result.x[0]
    r_opt = result.x[1:4]

    if m_opt <= 0:
        raise ValueError("Optimized mass is non-positive. Check data quality.")

    return {
        'mass': m_opt,
        'com': r_opt,
        'optimization_success': result.success,
        'optimization_message': result.message
    }

def compensate_weight(measurement, mass, com, force_bias, torque_bias, pose, g=9.81):
    '''
    Docstring for compensate_weight
    
    :param measurement: (n, 6) array of raw force/torque measurement (fx, fy, fz, tx, ty, tz)
     in sensor frame, assumed to be affected by gravity and biases.
     The function will return a compensated measurement with gravity effect removed and biases subtracted.
     The compensation is based on the estimated mass, CoM, and biases.

     The gravity compensation is computed as:
       f_gravity = m * R_ws^T @ [0, 0, -g]^T
       tau_gravity = r x f_gravity
    :param mass: Estimated payload mass (scalar)
    :param com: Estimated payload center of mass (3D vector in sensor frame)
     The CoM is used to compute the torque due to gravity as tau = r x f, where r is the CoM position vector.
     The function assumes the CoM is expressed in the same frame as the force/torque measurement.
     If the CoM is given in a different frame, it should be transformed to the sensor frame before calling this function.
     The accuracy of compensation depends on the accuracy of the CoM estimate, especially for larger payloads or longer lever arms.
     If the CoM is close to the sensor origin, the torque due to gravity will be small and compensation may be less sensitive to CoM errors.

     Note: This function does not handle dynamic effects such as acceleration or external disturbances. It only compensates for static gravity and biases.
     For dynamic compensation, additional terms would need to be included based on the robot's motion and dynamics.

     The output compensated measurement can be used for control or analysis as if the payload were not present, allowing for better performance when handling unknown or varying payloads.

     Example usage:
       compensated = compensate_weight(raw_measurement, estimated_mass, estimated_com, force_bias, torque_bias)
       # Use compensated for control or analysis
    :param force_bias: Estimated force bias (3D vector)
    :param torque_bias: Estimated torque bias (3D vector)
    :param pose: (4,4) Affine matrix of EE pose in world frame, used to compute the rotation from world to sensor frame.
    :param g: Gravity magnitude (default 9.81)
    '''
    # Compute gravity force in sensor frame
    g_world = mass * np.array([0.0, 0.0, -g])
    # Assuming pose is EE in world, and sensor frame is aligned with EE frame
    g_sensor = pose[:3, :3] @ g_world
    # Compute gravity torque in sensor frame
    tau_gravity = np.cross(com, g_sensor)
    # Compensate measurement
    external_force = measurement[:, :3] - g_sensor - force_bias
    external_torque = measurement[:, 3:] - tau_gravity - torque_bias
    return np.hstack([external_force, external_torque])