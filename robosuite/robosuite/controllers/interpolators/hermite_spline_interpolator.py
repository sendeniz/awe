import numpy as np
from scipy.interpolate import CubicHermiteSpline
from robosuite.controllers.interpolators.linear_interpolator import LinearInterpolator

class HermiteSplineInterpolator(LinearInterpolator):
        
    def set_goal(self, goal, start=None, start_vel=None, goal_vel=None):
        # don't rebuild spline if goal hasn't changed
        if hasattr(self, '_spline') and self._spline is not None and hasattr(self, 'goal'):
            if self.goal.ndim == 1 and np.allclose(np.array(goal), self.goal, atol=1e-6):
                return
        if start is not None:
            self.start = np.array(start)
        else:
            self.start = np.array(self.goal)
        self.goal      = np.array(goal)
        #self.start_vel = np.array(start_vel) if start_vel is not None else np.zeros(self.dim)
        
        if start_vel is not None:
            self.start_vel = np.array(start_vel)
        elif hasattr(self, 'goal_vel') and self.goal.ndim == 1:
            self.start_vel = np.array(self.goal_vel)
        else:
            self.start_vel = np.zeros(self.dim)
            
        self.goal_vel  = np.array(goal_vel)  if goal_vel  is not None else np.zeros(self.dim)
        self.step      = 0
        
        # debug prints
        #if self.goal.ndim == 1:  # only print for position, not orientation
        #    print(f"start_vel zero: {np.allclose(self.start_vel, 0)} | {self.start_vel}")
        #    print(f"goal_vel  zero: {np.allclose(self.goal_vel,  0)} | {self.goal_vel}")
        #    print("---")

        # orientation goal is a (3,3) matrix — fall back to linear
        if self.goal.ndim > 1:
            self._spline = None
            return

        self._spline = CubicHermiteSpline(
            x    = np.array([0, self.total_steps]),
            y    = np.array([self.start, self.goal]),
            dydx = np.array([self.start_vel, self.goal_vel]),
        )

    def get_interpolated_goal(self):
        if self._spline is None:
            # linear fallback for orientation
            dx = (self.goal - self.start) * (self.step + 1) / self.total_steps
            point = self.start + dx
        else:
            point = self._spline(self.step + 1)

        if self.step < self.total_steps - 1:
            self.step += 1

        return point