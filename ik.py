import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from typing import List, Optional, Tuple
from matplotlib.backend_bases import MouseEvent

class RoboticArm2D:
    def __init__(
        self,
        link_lengths: List[float],
        max_speeds: List[float],
        max_accelerations: List[float],
        dt: float = 0.05,
    ) -> None:
        self.link_lengths: np.ndarray = np.array(link_lengths)
        self.num_links: int = len(link_lengths)
        self.joints: np.ndarray = np.zeros(self.num_links)  # Current joint angles
        self.max_speeds: np.ndarray = np.array(max_speeds)
        self.max_accelerations: np.ndarray = np.array(max_accelerations)
        self.target_joints: np.ndarray = np.zeros(self.num_links)
        self.target: Optional[np.ndarray] = None
        self.current_time: float = 0.0
        self.movement_time: float = 0.0
        self.is_lin_movement: bool = False
        self.lin_start_pos: Optional[np.ndarray] = None
        self.start_joints: np.ndarray = np.zeros(self.num_links)
        self.dt: float = dt
        # Choose the inverse kinematics function to use
        self.inverse_kinematics_function = self.inverse_kinematics_scipy

    def forward_kinematics(self, joints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the forward kinematics for the given joint angles.
        Returns the positions of each joint and the end-effector.
        """
        if joints is None:
            joints = self.joints
        cumulative_angles = np.cumsum(joints)
        x_positions = np.cumsum(self.link_lengths * np.cos(cumulative_angles))
        y_positions = np.cumsum(self.link_lengths * np.sin(cumulative_angles))
        x_positions = np.concatenate(([0.0], x_positions))
        y_positions = np.concatenate(([0.0], y_positions))
        positions = np.column_stack((x_positions, y_positions))
        return positions

    def inverse_kinematics_scipy(self, target: np.ndarray) -> np.ndarray:
        """
        Perform inverse kinematics using the SciPy optimization library.
        """

        def objective(joints: np.ndarray) -> float:
            end_effector_pos = self.forward_kinematics(joints)[-1]
            return np.linalg.norm(end_effector_pos - target)

        def jacobian_objective(joints: np.ndarray) -> np.ndarray:
            end_effector_pos = self.forward_kinematics(joints)[-1]
            error = end_effector_pos - target
            norm_error = np.linalg.norm(error)
            if norm_error < 1e-8:
                # Gradient is zero when the error is negligible
                return np.zeros_like(joints)
            J = self.jacobian(joints)
            grad = J.T @ (error / norm_error)
            return grad

        result = minimize(
            objective,
            self.joints,
            method="BFGS",
            jac=jacobian_objective,
            options={"disp": False},
        )
        return result.x

    def jacobian(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for the given joint angles.
        """
        cumulative_angles = np.cumsum(joints)
        sines = np.sin(cumulative_angles)
        cosines = np.cos(cumulative_angles)
        n = self.num_links
        J = np.zeros((2, n))
        for i in range(n):
            J[0, i] = -np.sum(self.link_lengths[i:] * sines[i:])
            J[1, i] = np.sum(self.link_lengths[i:] * cosines[i:])
        return J

    def normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Normalize the joint angles to be within the range [-pi, pi].
        """
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def set_target(self, target: np.ndarray, is_lin_movement: bool = False) -> None:
        """
        Set the target position for the end-effector and plan the trajectory.
        """
        self.target = target
        self.is_lin_movement = is_lin_movement
        self.start_joints = self.joints.copy()
        self.current_time = 0.0

        if is_lin_movement:
            self.lin_start_pos = self.forward_kinematics(self.start_joints)[-1]
            self.plan_trajectory_lin()
        else:
            self.target_joints = self.normalize_angles(
                self.inverse_kinematics_function(target)
            )
            self.plan_trajectory_ptp()

    def plan_trajectory_ptp(self) -> None:
        """
        Plan a point-to-point (PTP) trajectory to the target joint angles.
        """
        diff = self.normalize_angles(self.target_joints - self.start_joints)
        max_times = np.sqrt(2 * np.abs(diff) / self.max_accelerations)
        max_time = np.max(max_times)
        if max_time == 0:
            max_time = self.dt  # Prevent division by zero
        self.movement_time = max_time

    def plan_trajectory_lin(self) -> None:
        """
        Plan a linear trajectory to the target position.
        """
        if self.lin_start_pos is None or self.target is None:
            return
        diff = self.target - self.lin_start_pos
        max_acc = np.max(self.max_accelerations)
        max_time = np.sqrt(2 * np.linalg.norm(diff) / max_acc)
        if max_time == 0:
            max_time = self.dt  # Prevent division by zero
        self.movement_time = max_time

    def move_to_target(self) -> bool:
        """
        Move the joints towards the target position or joint angles.
        Returns True if the movement is complete, False otherwise.
        """
        if self.target is None or self.current_time >= self.movement_time:
            if not self.is_lin_movement:
                self.joints = self.target_joints.copy()
            return True

        t = self.current_time
        T = self.movement_time
        s = t / T  # Progress ratio (0 to 1)

        if self.is_lin_movement:
            if self.lin_start_pos is None or self.target is None:
                return True
            # Compute the current target along the straight line path
            current_target = self.lin_start_pos + s * (self.target - self.lin_start_pos)
            # Solve inverse kinematics for the current target position
            new_joints = self.normalize_angles(
                self.inverse_kinematics_function(current_target)
            )
            # Update the joints directly to the new position
            self.joints = new_joints
        else:
            diff = self.normalize_angles(self.target_joints - self.start_joints)
            self.joints = self.normalize_angles(self.start_joints + s * diff)

        self.current_time += self.dt
        return False

# Robot configuration
robot_configuration = {
    "link_lengths": [2.0, 1.5, 1.0],
    "max_speeds": [np.deg2rad(360)] * 3,  # Max speed in radians per second
    "max_accelerations": [np.deg2rad(100)] * 3,  # Max acceleration in radians per second^2
    "dt": 0.016,  # Time step in seconds (0.016 seconds per frame = 60 FPS)
}

# Set up the robotic arm
arm = RoboticArm2D(**robot_configuration)

# Set up the plot
fig, ax = plt.subplots()
axis_limits = [-5.0, 5.0]
ax.set_xlim(axis_limits[0], axis_limits[1])
ax.set_ylim(axis_limits[0], axis_limits[1])
ax.set_aspect("equal")
line, = ax.plot([], [], "o-", lw=2)
target_point, = ax.plot([], [], "rx")

# Click event handler
def on_click(event: MouseEvent) -> None:
    if event.inaxes == ax:
        target = np.array([event.xdata, event.ydata])
        if event.button == 1:  # Left click for PTP movement
            arm.set_target(target, is_lin_movement=False)
        elif event.button == 3:  # Right click for LIN movement
            arm.set_target(target, is_lin_movement=True)

# Connect the click event to the figure
fig.canvas.mpl_connect("button_press_event", on_click)

# Animation function
def animate(frame: int) -> Tuple[plt.Line2D, plt.Line2D]:
    arm.move_to_target()
    positions = arm.forward_kinematics()
    line.set_data(positions[:, 0], positions[:, 1])
    if arm.target is not None:
        target_point.set_data([arm.target[0]], [arm.target[1]])
    else:
        target_point.set_data([], [])
    return line, target_point

# Create the animation
anim = FuncAnimation(fig, animate, interval=int(arm.dt * 1000), blit=True)
plt.title("2D Robotic Arm PTP and LIN Movement (Left Click: PTP, Right Click: LIN)")
plt.grid(True)
plt.show()