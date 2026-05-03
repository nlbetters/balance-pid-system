import math


class RobotKinematics:

    def __init__(self, lp=7.125, l1=6.20, l2=4.50, lb=4.00, invert=False):

        self.lp = lp    # Radius of Top
        self.l1 = l1    # Top Arm
        self.l2 = l2    # Bottom Arm
        self.lb = lb    # Radius of Bottom
        self.invert = invert    # False selects outward elbows; True selects inward elbows.

        self.maxh = self.compute_maxh() - 0.2
        self.minh = self.compute_minh() + 0.45
        self.p = [0.0, 0.0, self.maxh]
        self.h = (self.maxh + self.minh) / 2
        self.maxtheta = 10

        # Top Nodes
        self.A1 = [0, 0, 0]
        self.A2 = [0, 0, 0]
        self.A3 = [0, 0, 0]
        self.A4 = [0, 0, 0]

        # Bottom Nodes
        self.B1 = [0, 0, 0]
        self.B2 = [0, 0, 0]
        self.B3 = [0, 0, 0]
        self.B4 = [0, 0, 0]

        # Middle Nodes
        self.C1 = [0.0, 0.0, 0.0]
        self.C2 = [0.0, 0.0, 0.0]
        self.C3 = [0.0, 0.0, 0.0]
        self.C4 = [0.0, 0.0, 0.0]

        self.max_theta(self.h)

        self.theta1 = 0
        self.theta2 = 0
        self.theta3 = 0
        self.theta4 = 0

    def compute_maxh(self):
        return math.sqrt(((self.l1 + self.l2) ** 2) - ((self.lp - self.lb) ** 2))

    def compute_minh(self):
        if self.l1 > self.l2:
            return math.sqrt((self.l1 ** 2) - ((self.lb + self.l2 - self.lp) ** 2))
        elif self.l2 > self.l1:
            return math.sqrt(((self.l2 - self.l1) ** 2) - ((self.lp - self.lb) ** 2))
        else:
            return 0

    def solve_top(self, a, b, c, h):

        def top_point(phi):
            proj = a * math.cos(phi) + b * math.sin(phi)
            denom = math.sqrt(c * c + proj * proj)
            x = self.lp * c * math.cos(phi) / denom
            y = self.lp * c * math.sin(phi) / denom
            z = h - self.lp * proj / denom
            return [x, y, z]

        self.A1 = top_point(0.0)
        self.A2 = top_point(math.pi / 2)
        self.A3 = top_point(math.pi)
        self.A4 = top_point(3 * math.pi / 2)

    def _circle_intersection_2d(self, A, B, r0, r1):
        ax, az = float(A[0]), float(A[1])
        bx, bz = float(B[0]), float(B[1])
        dx = bx - ax
        dz = bz - az
        d = math.hypot(dx, dz)
        if d < 1e-9:
            raise ValueError("A and B cannot coincide")
        if d > r0 + r1 or d < abs(r0 - r1):
            raise ValueError("No valid leg solution for given geometry")

        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = math.sqrt(max(0.0, r0**2 - a**2))
        p2x = ax + a * dx / d
        p2z = az + a * dz / d
        perp_x = -dz / d
        perp_z = dx / d

        return [p2x + h * perp_x, p2z + h * perp_z], [p2x - h * perp_x, p2z - h * perp_z]

    def _solve_leg(self, A, B, axis):
        if axis == "x":
            A2d = [A[0], A[2]]
            B2d = [B[0], B[2]]
            sol1, sol2 = self._circle_intersection_2d(A2d, B2d, self.l1, self.l2)
            candidates = [[sol1[0], 0.0, sol1[1]], [sol2[0], 0.0, sol2[1]]]
        elif axis == "y":
            A2d = [A[1], A[2]]
            B2d = [B[1], B[2]]
            sol1, sol2 = self._circle_intersection_2d(A2d, B2d, self.l1, self.l2)
            candidates = [[0.0, sol1[0], sol1[1]], [0.0, sol2[0], sol2[1]]]
        else:
            raise ValueError("Unsupported leg axis")

        bx, by = B[0], B[1]
        base_radius = math.hypot(bx, by)
        if base_radius < 1e-9:
            raise ValueError("Base point cannot be at platform center")

        def radial_position(candidate):
            return (candidate[0] * bx + candidate[1] * by) / base_radius

        if self.invert:
            return min(candidates, key=radial_position)
        return max(candidates, key=radial_position)

    def _compute_theta(self, C):
        horizontal = math.sqrt(C[0]**2 + C[1]**2)
        return math.pi / 2 - math.atan2(horizontal - self.lb, C[2])

    def solve_middle(self):
        self.C1 = self._solve_leg(self.A1, self.B1, axis="x")
        self.C2 = self._solve_leg(self.A2, self.B2, axis="y")
        self.C3 = self._solve_leg(self.A3, self.B3, axis="x")
        self.C4 = self._solve_leg(self.A4, self.B4, axis="y")

    def solve_inverse_kinematics_vector(self, a, b, c, h):
        self.B1 = [self.lb, 0, 0]
        self.B2 = [0, self.lb, 0]
        self.B3 = [-self.lb, 0, 0]
        self.B4 = [0, -self.lb, 0]

        self.solve_top(a, b, c, h)
        self.solve_middle()

        self.theta1 = self._compute_theta(self.C1)
        self.theta2 = self._compute_theta(self.C2)
        self.theta3 = self._compute_theta(self.C3)
        self.theta4 = self._compute_theta(self.C4)

    def solve_inverse_kinematics_spherical(self, theta, phi, h):
        self.h = h
        self.max_theta(h)

        # Do not clamp theta to self.maxtheta here while tuning the real robot.
        # main.py already applies the software tilt limit, and this extra clamp was
        # preventing the physical robot from matching the larger motion seen in app.py.
        requested_theta = theta

        a = math.sin(math.radians(requested_theta)) * math.cos(math.radians(phi))
        b = math.sin(math.radians(requested_theta)) * math.sin(math.radians(phi))
        c = math.cos(math.radians(requested_theta))

        try:
            self.solve_inverse_kinematics_vector(a, b, c, h)
        except Exception as e:
            print(a, b, c, h, requested_theta, phi)
            pass

    def max_theta(self, h, tol=1e-3):
        theta_low, theta_high = 0.0, math.radians(20)

        def valid(theta):
            c = math.cos(theta)
            for s in (1, -1):
                a21 = self.lp * c
                a23 = h - self.lp * (s * math.sin(theta))
                try:
                    p2 = (self.lb - a21) / a23
                    q2 = (a21**2 + a23**2 - self.lb**2 + self.l2**2 - self.l1**2) / (2 * a23)
                    r2 = p2**2 + 1
                    s2 = 2 * (p2 * q2 - self.lb)
                    t2 = q2**2 - self.l2**2 + self.lb**2
                    disc = s2**2 - 4 * r2 * t2
                    if disc < 0:
                        return False
                    c21 = (-s2 + math.sqrt(disc)) / (2 * r2)
                    delta = self.l2**2 - (c21 - self.lb)**2
                    if delta < 0:
                        return False
                    c23 = math.sqrt(delta)
                    if abs(math.sqrt((a21 - c21)**2 + (a23 - c23)**2) - self.l1) > 1e-3:
                        return False
                    if abs(math.sqrt((self.lb - c21)**2 + c23**2) - self.l2) > 1e-3:
                        return False
                except Exception:
                    return False
            return True

        while theta_high - theta_low > tol:
            theta_mid = (theta_low + theta_high) / 2
            if valid(theta_mid):
                theta_low = theta_mid
            else:
                theta_high = theta_mid

        self.maxtheta = max(0, math.degrees(round(theta_low, 4)) - 0.5)
