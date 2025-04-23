import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from robosuite.controllers.interpolators.base_interpolator import Interpolator
from robosuite.controllers.interpolators.linear_interpolator import LinearInterpolator
from robosuite.controllers.interpolators.utils import compute_3d_error

def test_linear_interpolation():
    # Create interpolator with more steps for smoother visualization
    interpolator = LinearInterpolator(
        ndim=3,
        controller_freq=100,
        policy_freq=20,
        ramp_ratio=2.0,  # More steps for smoother interpolation
        ori_interpolate=None
    )
    
    # Create a dense non-linear reference trajectory (sine wave pattern)
    dense_steps = 100  # For smooth reference curve
    t = np.linspace(0, 2*np.pi, dense_steps)
    dense_ref = np.column_stack([
        t,
        np.sin(t),
        np.cos(t)
    ])
    
    # Sparse waypoints for the interpolator (few points along the same curve)
    num_waypoints = 5
    waypoint_indices = np.linspace(0, dense_steps-1, num_waypoints).astype(int)
    waypoints = dense_ref[waypoint_indices]
    
    # Test interpolation between waypoints
    all_interp_points = []
    
    for i in range(num_waypoints - 1):
        start = waypoints[i]
        goal = waypoints[i+1]
        
        interpolator.set_goal(goal, start)
        
        for _ in range(int(interpolator.total_steps)):
            all_interp_points.append(interpolator.get_interpolated_goal())
    
    interp_points = np.array(all_interp_points)
    
    # After interpolation:
    error_metrics = compute_3d_error(dense_ref[:len(interp_points)], interp_points)
    
    print("\n=== 3D Trajectory Error Metrics ===")
    print(f"Max Error: {error_metrics['max_error']:.6f}")
    print(f"Mean Error: {error_metrics['mean_error']:.6f}")
    print(f"RMSE: {error_metrics['rmse']:.6f}")
    print(f"Total Cumulative Error: {error_metrics['total_error']:.6f}")

    # Plot results
    fig = plt.figure(figsize=(18, 8))
    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot dense reference (true non-linear path)
    ax1.plot(dense_ref[:, 0], dense_ref[:, 1], dense_ref[:, 2], 
            'm-', label='True Non-linear Path', alpha=0.4, linewidth=4)
    
    # Plot waypoints
    ax1.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
            'ro', label='Waypoints', markersize=8)
    
    # Plot interpolated path
    ax1.plot(interp_points[:, 0], interp_points[:, 1], interp_points[:, 2],
            'b-', label='Linear Interpolation', alpha=0.8, linewidth=2)
    
    ax1.set_title('Linear Interpolation of Non-linear Path')
    ax1.legend()
    
    # Projection Plots
    ax2 = fig.add_subplot(322)
    ax2.plot(dense_ref[:, 0], dense_ref[:, 1], 'm-', alpha=0.4, linewidth=4)
    ax2.plot(waypoints[:, 0], waypoints[:, 1], 'ro')
    ax2.plot(interp_points[:, 0], interp_points[:, 1], 'b-')
    ax2.set_title('XY Projection')
    ax2.grid(True)
    
    ax3 = fig.add_subplot(324)
    ax3.plot(dense_ref[:, 0], dense_ref[:, 2], 'm-', alpha=0.4, linewidth=4)
    ax3.plot(waypoints[:, 0], waypoints[:, 2], 'ro')
    ax3.plot(interp_points[:, 0], interp_points[:, 2], 'b-')
    ax3.set_title('XZ Projection')
    ax3.grid(True)
    
    ax4 = fig.add_subplot(326)
    ax4.plot(dense_ref[:, 1], dense_ref[:, 2], 'm-', alpha=0.4, linewidth=4)
    ax4.plot(waypoints[:, 1], waypoints[:, 2], 'ro')
    ax4.plot(interp_points[:, 1], interp_points[:, 2], 'b-')
    ax4.set_title('YZ Projection')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

test_linear_interpolation()