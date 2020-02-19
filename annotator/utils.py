import numpy as np
import cv2


class PointList(object):
    def __init__(self, npoints):
        self.npoints = npoints
        self.ptlist = []
        self.stlist = []
        self.pos = 0

    def add(self, x, y, status):
        if self.pos < self.npoints:
            self.ptlist.append((x, y))
            self.stlist.append(status)
            self.pos += 1
            return True
        else:
            print('max points....')
            return False

    def sub(self):
        if self.pos > 0:
            self.pos -= 1
            del self.ptlist[-1]
            del self.stlist[-1]
        else:
            print('No key-points.')

    def len_ptlist(self):
        return len(self.ptlist)


def draw_points(img, points, color=(255, 0, 0)):
    """

    Args:
        img (np.ndarray): Array of an image.
        points (list or PointList): An instance contains key-points.
        color (tuple): BGR color.

    """
    if isinstance(points, PointList):
        points = points.ptlist

    for p in points:
        if is_occlusion_point(p):
            continue
        x, y = p
        cv2.circle(img, (x, y), 2, color, 2)


def draw_edge(img, points, edge_table, color=(255, 0, 0)):
    if isinstance(points, PointList):
        points = points.ptlist

    for i, p in enumerate(points):
        prev_idx = edge_table[i]
        if prev_idx is None:
            continue
        prev_p = points[prev_idx]

        if is_occlusion_point(p) or is_occlusion_point(prev_p):
            continue
        cv2.line(img, tuple(p), tuple(prev_p), color)


def make_edge_table(palm_indices, lst_finger_indices):
    edge_table = dict()
    for idx in palm_indices:
        if idx == 0:
            edge_table[idx] = None
        else:
            edge_table[idx] = 0

    for finger_indices in lst_finger_indices:
        for i, idx in enumerate(finger_indices):
            if i == 0:
                edge_table[idx] = 0
            else:
                edge_table[idx] = idx - 1
    return edge_table


def is_occlusion_point(pt):
    if pt[0] < 0 or pt[1] < 0:
        return True
    else:
        return False
