#!/usr/bin/env python3

import numpy as np
import gudhi as g

from filtrations import *
from barcodes import *
from ect import *

from manimlib.imports import *
from manimlib.mobject.geometry import Line


# TODO: add automated scaling (including time)
# TODO: add automated arrangement


class ECT(VGroup):

    def __init__(self, simp_comp, point_cloud, direction, extra_filt_mobjects=[], extra_vis_mobjects=[],
                 previous_vis_mobjects=[],
                 **kwargs):
        VGroup.__init__(self, **kwargs)

        direction = direction / np.linalg.norm(direction)
        self.filtration = SweepingPlaneFiltration(simp_comp, point_cloud, direction, extra_mobjects=extra_filt_mobjects)

        if point_cloud.shape[1] == 2:
            point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), axis=1)

        self.ect, self.euler_critical_points = euler_curve_callable([point_cloud[s] for s, _ in simp_comp.get_filtration()],
                                                                    direction,
                                                                    (self.min_fv, self.max_fv))

        ect = [self.ect(x) for x in self.euler_critical_points]
        y_min, y_max = min(ect), max(max(ect), 1)

        self.ect_vis = ECTFunction(self.min_fv, self.max_fv, y_min, y_max,
                                   self.euler_critical_points, self.ect, extra_vis_mobjects, previous_vis_mobjects)
        self.ect_vis.add(*extra_vis_mobjects)

        self.ect_vis.scale(3 / self.ect_vis.size)
        self.filtration.scale(1 / self.filtration.size)
        self.add(self.filtration)
        self.add(self.ect_vis)
        self.ect_vis.next_to(self.filtration, 5*RIGHT)

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


class PersistentHomology(VGroup):

    def __init__(self, simp_comp, point_cloud, hom_dim=0):
        super().__init__()
        self.filtration = Filtration(simp_comp, point_cloud, 'appearing')
        self.hom_dim = hom_dim
        self.barcode = Barcode(self.extract_barcode_in_dim(self.filtration.simp_comp.persistence(), self.hom_dim),
                               self.min_fv, self.max_fv)
        self.add(self.filtration)
        self.barcode.scale(0.3)
        self.add(self.barcode)
        #self.arrange(5 * RIGHT, aligned_edge=DOWN)
        self.barcode.next_to(self.filtration, 5*RIGHT)
        #self.shift(5 * LEFT)

    @staticmethod
    def extract_barcode_in_dim(barcode, dim):
        new_barcode = []
        for bar in barcode:
            if bar[0] == dim:
                new_barcode.append(bar[1])
        return new_barcode

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv

        anims, fvs = [], []
        if self.filtration.current_fv <= to_fv <= self.max_fv:
            fvs.append(self.filtration.current_fv)
            anims.append(self.filtration.animate_filtration(to_fv))

        if self.barcode.current_fv <= to_fv <= self.max_fv:
            fvs.append(self.barcode.current_fv)
            anims.append(self.barcode.animate_barcode(to_fv))

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

    def animate_reverse_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.min

        anims, fvs = [], []
        if self.filtration.current_fv >= to_fv >= self.min_fv:
            fvs.append(self.filtration.current_fv)
            anims.append(self.filtration.animate_reverse_filtration(to_fv))

        if self.barcode.current_fv >= to_fv >= self.min_fv:
            fvs.append(self.barcode.current_fv)
            anims.append(self.barcode.animate_reverse_barcode(to_fv))

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

    @property
    def min_fv(self):
        return self.filtration.min_fv - self.filtration.offset

    @property
    def max_fv(self):
        return self.filtration.max_fv + self.filtration.offset


class CechPersistence(PersistentHomology):

    def __init__(self, point_cloud, max_radius, hom_dim=0):
        point_cloud += 3 * LEFT
        filtration = CechFiltration(point_cloud, max_radius)
        super().__init__(filtration.simp_comp, point_cloud, hom_dim)
        self.remove(self.filtration)
        self.filtration = filtration
        self.add(self.filtration)


class RipsPersistence(PersistentHomology):

    def __init__(self, point_cloud, max_radius, hom_dim=0):
        filtration = RipsFiltration(point_cloud, max_radius)
        super().__init__(filtration.simp_comp, point_cloud, hom_dim)
        self.remove(self.filtration)
        self.filtration = filtration
        self.add(self.filtration)
