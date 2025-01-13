from math import tan, atan2, acos, pi
import numpy as np
from dataclasses import dataclass
from math import cos, sin, sqrt
from numpy.typing import NDArray
from typing import Optional, List

def transform(x:float, y:float, w:float, l:float, theta:float, id:int):
    """ Coordinates transform. """
    x_ = y_ = 0

    if id == 1:
        x_ = x + w*cos(theta) - l*sin(theta)
        y_ = y + w*sin(theta) + l*cos(theta)
    if id == 2:
        x_ = x + w*cos(theta) + l*sin(theta)
        y_ = y + w*sin(theta) - l*cos(theta)
    if id == 3:
        x_ = x - w*cos(theta) - l*sin(theta)
        y_ = y - w*sin(theta) + l*cos(theta)
    if id == 4:
        x_ = x - w*cos(theta) + l*sin(theta)
        y_ = y - w*sin(theta) - l*cos(theta)

    return np.array([x_, y_])


def directional_theta(vec1:NDArray[np.float_], vec2:NDArray[np.float_], d):
    """ Calculate the directional theta change. """

    theta = atan2(vec2[1], vec2[0]) - atan2(vec1[1], vec1[0])

    if theta < 0 and d == 1:
        theta += 2*pi
    elif theta > 0 and d == -1:
        theta -= 2*pi

    return theta


def distance(pt1:NDArray[np.float_], pt2:NDArray[np.float_]):
    """ Distance of two points. """

    d = sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

    return d


@dataclass
class Params:
    """ Store parameters for different dubins paths. """
    d : List[int]
    t1 : Optional[NDArray[np.float_]] = None
    t2 : Optional[NDArray[np.float_]] = None
    c1 : Optional[NDArray[np.float_]] = None
    c2 : Optional[NDArray[np.float_]] = None
    len : Optional[float] = None


class DubinsPath:
    """
    Consider four dubins paths
    - LSL
    - LSR
    - RSL
    - RSR
    and find the shortest obstacle-free one.
    """

    def __init__(self, car):

        self.car = car
        self.r = self.car.l / tan(self.car.max_phi)

        # turn left: 1, turn right: -1
        self.direction = {
            'LSL': [1, 1],
            'LSR': [1, -1],
            'RSL': [-1, 1],
            'RSR': [-1, -1]
        }

    def find_tangents(self, start_pos:NDArray[np.float_], end_pos:NDArray[np.float_]):
        """ Find the tangents of four dubins paths. """

        self.start_pos = start_pos
        self.end_pos = end_pos

        x1, y1, theta1 = start_pos
        x2, y2, theta2 = end_pos

        self.s = np.array(start_pos[:2])
        self.e = np.array(end_pos[:2])

        self.lc1 = transform(x1, y1, 0, self.r, theta1, 1)
        self.rc1 = transform(x1, y1, 0, self.r, theta1, 2)
        self.lc2 = transform(x2, y2, 0, self.r, theta2, 1)
        self.rc2 = transform(x2, y2, 0, self.r, theta2, 2)

        solutions = [self._LSL(), self._LSR(), self._RSL(), self._RSR()]
        solutions = [s for s in solutions if s is not None]
        solutions.sort(key=lambda x: x.len, reverse=False)

        return solutions

    def get_params(self, dub:Params, c1:NDArray[np.float_], c2:NDArray[np.float_], t1:NDArray[np.float_], t2:NDArray[np.float_]):
        """ Calculate the dubins path length. """

        v1 = self.s - c1
        v2 = t1     - c1
        v3 = t2     - t1
        v4 = t2     - c2
        v5 = self.e - c2

        delta_theta1 = directional_theta(v1, v2, dub.d[0])
        delta_theta2 = directional_theta(v4, v5, dub.d[1])

        arc1    = abs(delta_theta1*self.r)
        tangent = np.linalg.norm(v3)
        arc2    = abs(delta_theta2*self.r)

        theta = self.start_pos[2] + delta_theta1

        dub.t1 = t1.tolist() + [theta]
        dub.t2 = t2.tolist() + [theta]
        dub.c1 = c1
        dub.c2 = c2
        dub.len = arc1 + tangent + arc2

        return dub

    def _LSL(self):

        lsl = Params(self.direction['LSL'])

        cline = self.lc2 - self.lc1
        R = np.linalg.norm(cline) / 2
        theta = atan2(cline[1], cline[0]) - acos(0)

        t1 = transform(self.lc1[0], self.lc1[1], self.r, 0, theta, 1)
        t2 = transform(self.lc2[0], self.lc2[1], self.r, 0, theta, 1)

        lsl = self.get_params(lsl, self.lc1, self.lc2, t1, t2)

        return lsl

    def _LSR(self):

        lsr = Params(self.direction['LSR'])

        cline = self.rc2 - self.lc1
        R = np.linalg.norm(cline) / 2

        if R < self.r:
            return None

        theta = atan2(cline[1], cline[0]) - acos(self.r/R)

        t1 = transform(self.lc1[0], self.lc1[1], self.r, 0, theta, 1)
        t2 = transform(self.rc2[0], self.rc2[1], self.r, 0, theta+pi, 1)

        lsr = self.get_params(lsr, self.lc1, self.rc2, t1, t2)

        return lsr

    def _RSL(self):

        rsl = Params(self.direction['RSL'])

        cline = self.lc2 - self.rc1
        R = np.linalg.norm(cline) / 2

        if R < self.r:
            return None

        theta = atan2(cline[1], cline[0]) + acos(self.r/R)

        t1 = transform(self.rc1[0], self.rc1[1], self.r, 0, theta, 1)
        t2 = transform(self.lc2[0], self.lc2[1], self.r, 0, theta+pi, 1)

        rsl = self.get_params(rsl, self.rc1, self.lc2, t1, t2)

        return rsl

    def _RSR(self):

        rsr = Params(self.direction['RSR'])

        cline = self.rc2 - self.rc1
        R = np.linalg.norm(cline) / 2
        theta = atan2(cline[1], cline[0]) + acos(0)

        t1 = transform(self.rc1[0], self.rc1[1], self.r, 0, theta, 1)
        t2 = transform(self.rc2[0], self.rc2[1], self.r, 0, theta, 1)

        rsr = self.get_params(rsr, self.rc1, self.rc2, t1, t2)

        return rsr


    def construct_ringsectors(self, start_pos, end_pos, d, c, r):
        """ Construct inner and outer ringsectors of a turning route. """

        x, y, theta = start_pos

        delta_theta = end_pos[2] - theta

        p_inner = start_pos[:2]
        id = 1 if d == -1 else 2
        p_outer = transform(x, y, 1.3*self.car.l, 0.4*self.car.l, theta, id)

        r_inner = r - self.car.carw / 2
        r_outer = distance(p_outer, c)

        v_inner = [p_inner[0]-c[0], p_inner[1]-c[1]]
        v_outer = [p_outer[0]-c[0], p_outer[1]-c[1]]

        if d == -1:
            end_inner = atan2(v_inner[1], v_inner[0]) % (2*pi)
            start_inner = (end_inner + delta_theta) % (2*pi)

            end_outer = atan2(v_outer[1], v_outer[0]) % (2*pi)
            start_outer = (end_outer + delta_theta) % (2*pi)

        if d == 1:
            start_inner = atan2(v_inner[1], v_inner[0]) % (2*pi)
            end_inner = (start_inner + delta_theta) % (2*pi)

            start_outer = atan2(v_outer[1], v_outer[0]) % (2*pi)
            end_outer = (start_outer + delta_theta) % (2*pi)

        rs_inner = [c[0], c[1], r_inner, r, start_inner, end_inner]
        rs_outer = [c[0], c[1], r, r_outer, start_outer, end_outer]

        return rs_inner, rs_outer

    def get_route(self, s):
        """ Get the route of dubins path. """

        phi1 = self.car.max_phi if s.d[0] == 1 else -self.car.max_phi
        phi2 = self.car.max_phi if s.d[1] == 1 else -self.car.max_phi

        phil = [phi1, 0, phi2]
        goal = [s.t1, s.t2, self.end_pos]
        ml = [1, 1, 1]

        return list(zip(goal, phil, ml))
