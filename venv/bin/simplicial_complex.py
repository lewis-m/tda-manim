#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import pdist
import gudhi as g

from manimlib.imports import *
from manimlib.mobject.geometry import Line


class SimplicialComplex(VGroup):

    CONFIG = {
        'simplex_color_0': RED,
        'simplex_width_0': 0.000001,
        'simplex_opacity_0': 1,
        'simplex_color_1': WHITE,
        'simplex_width_1': 1,
        'simplex_opacity_1': 1,
        'simplex_color_2': WHITE,
        'simplex_opacity_2': 0.2
    }

    def __init__(self, simp_comp: g.SimplexTree, point_cloud: np.ndarray, add_simp=True, **kwargs):
        super(VGroup, self).__init__(**kwargs)
        digest_config(self, kwargs)

        if point_cloud.shape[1] == 2:
            point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), axis=1)
        assert point_cloud.shape[1] == 3
        assert point_cloud.shape[0] == simp_comp.num_vertices()

        if simp_comp.dimension() > 2:
            simp_comp.set_dimension(2)

        self.simp_comp = simp_comp
        self.point_cloud = point_cloud
        self.mobject_dict = dict({})
        if add_simp:
            self.init_sc()

    def init_sc(self):
        self.mobject_dict = dict({})
        for i in [3, 2, 1]:
            for s, _ in self.simp_comp.get_simplices():
                if len(s) == 1 == i:
                    d = Dot(self.point_cloud[s[0]], radius=self.CONFIG['simplex_width_0'])
                    d.set_stroke(self.CONFIG['simplex_color_0'], 1,
                                 self.CONFIG['simplex_opacity_0'])
                    d.set_fill(self.CONFIG['simplex_color_0'],
                               self.CONFIG['simplex_opacity_0'])
                    self.add(d)
                    self.mobject_dict[str(s)] = d
                elif len(s) == 2 == i:
                    start, end = self.point_cloud[s[0]], self.point_cloud[s[1]]
                    l = Line(start, end, color=self.CONFIG['simplex_color_1'],
                             stroke_width=self.CONFIG['simplex_width_1'], fill_opacity=self.CONFIG['simplex_opacity_1'])
                    self.add(l)
                    self.mobject_dict[str(s)] = l
                elif len(s) == 3 == i:
                    p0, p1, p2 = self.point_cloud[s[0]], self.point_cloud[s[1]], self.point_cloud[s[2]]
                    t = Polygon(p0, p1, p2, fill_opacity=self.CONFIG['simplex_opacity_2'],
                                fill_color=self.CONFIG['simplex_color_2'])
                    self.add(t)
                    self.mobject_dict[str(s)] = t

        return self

    def change_simplex_color(self, *simplices, color=WHITE):
        for s in simplices:
            self.mobject_dict[str(s)].set_fill(color, self.mobject_dict[str(s)].CONFIG['opacity'])
        return self

    def change_simplex_size(self, *simplices, size=1):
        for s in simplices:
            if len(s) == 1:
                self.mobject_dict[str(s)].CONFIG['radius'] = size
            elif len(s) == 2:
                self.mobject_dict[str(s)].CONFIG['stroke_width'] = size
        return self

    def change_simplex_opacity(self, *simplices, opacity=1):
        for s in simplices:
            self.mobject_dict[str(s)].set_fill(self.mobject_dict[str(s)].CONFIG['fill_color'], opacity)
        return self

    @property
    def size(self):
        return np.max(pdist(self.point_cloud, metric='chebyshev'))
