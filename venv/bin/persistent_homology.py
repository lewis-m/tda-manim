#!/usr/bin/env python3

import numpy as np
import gudhi as g

from filtrations import *
from barcodes import *
from ect import *

from manimlib.imports import *
from manimlib.mobject.geometry import Line


class ECT(VGroup):

    def __init__(self, simp_comp, points, direction, extra_filt_mobjects=[], extra_vis_mobjects=[], **kwargs):
        VGroup.__init__(self, **kwargs)

        direction = direction / np.linalg.norm(direction)
        self.filtration = SweepingPlaneFiltration(simp_comp, points, direction, extra_mobjects=extra_filt_mobjects)

        if points.shape[1] == 2:
            points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)

        self.ect, self.euler_critical_points = euler_curve_callable([points[s] for s, _ in simp_comp.get_filtration()],
                                                                    direction,
                                                                    (self.min_fv, self.max_fv))

        ect = [self.ect(x) for x in self.euler_critical_points]
        y_min, y_max = min(ect), max(max(ect), 1)

        self.ect_vis = ECTFunction(self.min_fv, self.max_fv, y_min, y_max,
                                   self.euler_critical_points, self.ect)
        self.ect_vis.add(*extra_vis_mobjects)

        self.ect_vis.scale(0.6)  # need automated scaling
        self.add(self.filtration)
        self.add(self.ect_vis)
        self.arrange(3 * RIGHT, aligned_edge=DOWN)  # need better arrangement

    @property
    def min_fv(self):
        return self.filtration.min_fv - self.filtration.offset

    @property
    def max_fv(self):
        return self.filtration.max_fv + self.filtration.offset

    def animate_filtration(self, to_fv=None, graphs=None):
        if graphs is None:
            graphs = ['plane', 'discrete']
        if to_fv is None:
            to_fv = self.max_fv

        anims, fvs = [], []
        if 'plane' in graphs and self.filtration.current_fv <= to_fv <= self.max_fv:
            fvs.append(self.filtration.current_fv)
            anims.append(self.filtration.animate_filtration(to_fv))

        if 'discrete' in graphs and self.ect_vis.current_fv_d <= to_fv <= self.max_fv:
            fvs.append(self.ect_vis.current_fv_d)
            anims.append(self.ect_vis.animate_discrete_graph(to_fv))

        if 'smooth' in graphs and self.ect_vis.current_fv_s <= to_fv <= self.max_fv:
            fvs.append(self.ect_vis.current_fv_s)
            anims.append(self.ect_vis.animate_smooth_graph(to_fv))

        fvs = np.array(fvs)
        anim_grp = AnimationGroup(*anims, lag_ratio=0)
        anim_grp.anims_with_timings = []
        for a, v in zip(anims, fvs - np.min(fvs)):
            end_time = v + a.get_run_time()
            anim_grp.anims_with_timings.append(
                (a, v, end_time)
            )

        if anim_grp.anims_with_timings:
            anim_grp.max_end_time = np.max([
                awt[2] for awt in anim_grp.anims_with_timings
            ])
        else:
            anim_grp.max_end_time = 0
        if anim_grp.run_time is None:
            anim_grp.run_time = anim_grp.max_end_time

        anim_grp.max_end_time = to_fv - np.min(fvs) if fvs.size > 0 else 0
        anim_grp.run_time = to_fv - np.min(fvs) if fvs.size > 0 else 0
        return anim_grp

    def animate_reverse_filtration(self, to_fv=None, graphs=None):
        if graphs is None:
            graphs = ['plane', 'discrete']
        if to_fv is None:
            to_fv = self.min

        anims, fvs = [], []
        if 'plane' in graphs and self.filtration.current_fv >= to_fv >= self.min_fv:
            fvs.append(self.filtration.current_fv)
            anims.append(self.filtration.animate_reverse_filtration(to_fv))

        if 'discrete' in graphs and self.ect_vis.current_fv_d >= to_fv >= self.min_fv:
            fvs.append(self.ect_vis.current_fv_d)
            anims.append(self.ect_vis.animate_reverse_discrete_graph(to_fv))

        if 'smooth' in graphs and self.ect_vis.current_fv_s >= to_fv >= self.min_fv:
            fvs.append(self.ect_vis.current_fv_s)
            anims.append(self.ect_vis.animate_reverse_smooth_graph(to_fv))

        fvs = np.array(fvs)
        anim_grp = AnimationGroup(*anims, lag_ratio=0)
        anim_grp.anims_with_timings = []
        if fvs.size > 0:
            for a, v in zip(anims, np.max(fvs) - fvs):
                end_time = v + a.get_run_time()
                anim_grp.anims_with_timings.append(
                    (a, v, end_time)
                )

            if anim_grp.anims_with_timings:
                anim_grp.max_end_time = np.max([
                    awt[2] for awt in anim_grp.anims_with_timings
                ])
            else:
                anim_grp.max_end_time = 0
            if anim_grp.run_time is None:
                anim_grp.run_time = anim_grp.max_end_time

        anim_grp.max_end_time = -(to_fv - np.max(fvs)) if fvs.size > 0 else 0
        anim_grp.run_time = -(to_fv - np.max(fvs)) if fvs.size > 0 else 0
        return anim_grp

    def update_direction(self, new_direction):
        self.ect_vis.drop_current_graphs()
        return self.filtration.update_direction(new_direction)

    def animate_discrete_mean_shift(self):
        return self.ect_vis.animate_discrete_mean_shift()
