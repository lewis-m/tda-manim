#!/usr/bin/env python3

import simplicial_complex

import numpy as np
import gudhi as g
from manimlib.imports import *
from manimlib.mobject.geometry import Line


class Filtration(SimplicialComplex):
    CONFIG = {
        'simplex_color_0': RED,
        'simplex_width_0': 0.5,
        'simplex_opacity_0': 1,
        'simplex_color_1': WHITE,
        'simplex_width_1': 2,
        'simplex_opacity_1': 1,
        'simplex_color_2': WHITE,
        'simplex_opacity_2': 0.5,
        'pre_filter_color': GRAY,
        'pre_filter_opacity': lambda x: x / 2
    }

    def __init__(self, simp_comp: g.SimplexTree, points: np.ndarray, display_type='color_change', **kwargs):
        super().__init__(simp_comp, points, True, **kwargs)

        assert display_type in ['color_change', 'appearing']
        self.display_type = display_type
        self.filtered = False

        if display_type == 'color_change':
            for s, _ in self.simp_comp.get_simplices():
                if len(s) == 3:
                    self.mobject_dict[str(s)].set_fill(color=self.CONFIG['pre_filter_color'],
                                                       opacity=self.CONFIG['pre_filter_opacity'](
                                                           self.CONFIG['simplex_opacity_2']))
                    self.mobject_dict[str(s)].set_stroke(color=self.CONFIG['pre_filter_color'],
                                                         opacity=self.CONFIG['pre_filter_opacity'](
                                                             self.CONFIG['simplex_opacity_2']),
                                                         width=0)
                else:
                    self.mobject_dict[str(s)].set_fill(color=self.CONFIG['pre_filter_color'],
                                                       opacity=0.5)
                    self.mobject_dict[str(s)].set_stroke(color=self.CONFIG['pre_filter_color'],
                                                         width=self.mobject_dict[str(s)].stroke_width / 100,
                                                         opacity=0.5)
        else:
            for s, _ in self.simp_comp.get_simplices():
                if len(s) > 1:
                    self.mobject_dict[str(s)].set_fill(self.mobject_dict.fill_color, 0)
                    self.mobject_dict[str(s)].set_stroke(self.mobject_dict.fill_color, 0)

    def change_simplex(self, s):
        if isinstance(s, Dot):
            s.set_fill(color=self.CONFIG['simplex_color_0'], opacity=self.CONFIG['simplex_opacity_0'])
            s.set_stroke(color=self.CONFIG['simplex_color_0'], width=self.CONFIG['simplex_width_0'],
                         opacity=self.CONFIG['simplex_opacity_0'])
        elif isinstance(s, Line):
            s.set_fill(color=self.CONFIG['simplex_color_1'], opacity=self.CONFIG['simplex_opacity_1'])
            s.set_stroke(color=self.CONFIG['simplex_color_1'], width=self.CONFIG['simplex_width_1'],
                         opacity=self.CONFIG['simplex_opacity_1'])
        elif isinstance(s, Polygon):
            s.set_fill(color=self.CONFIG['simplex_color_2'], opacity=self.CONFIG['simplex_opacity_2'])

        return s

    def animate_filtration(self, offset=0.5):
        if not self.filtered:
            anims, fvs = [], []
            for s, v in self.simp_comp.get_filtration():
                anims.append(ApplyFunction(self.change_simplex, self.mobject_dict[str(s)], run_time=0.01))
                fvs.append(v)

            fvs = np.array(fvs)
            fvs -= np.min(fvs)
            fvs /= np.max(fvs)
            fvs += offset

            anim_grp = AnimationGroup(*anims)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
                end_time = v + a.get_run_time()
                anim_grp.anims_with_timings.append(
                    (a, v, end_time)
                )
                print((v, end_time))
            if anim_grp.anims_with_timings:
                anim_grp.max_end_time = np.max([
                    awt[2] for awt in anim_grp.anims_with_timings
                ])
            else:
                anim_grp.max_end_time = 0
            if anim_grp.run_time is None:
                anim_grp.run_time = anim_grp.max_end_time

            anim_grp.max_end_time += offset
            anim_grp.run_time += offset
            self.filtered = True
            return anim_grp
        else:
            return AnimationGroup([])


class SweepingPlaneFiltration(Filtration):

    def __init__(self, simp_comp: g.SimplexTree, points: np.ndarray, normal_vector, plane_computed=False,
                 plane_color=YELLOW, plane_width=1, plane_offset=0.5, **kwargs):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.CONFIG['plane_color'] = plane_color
        self.CONFIG['plane_width'] = plane_width
        self.CONFIG['plane_offset'] = plane_offset

        if not plane_computed:
            for s, _ in simp_comp.get_simplices():
                simp_comp.assign_filtration(s, max([points[vertex].dot(self.normal_vector) for vertex in s]))
        fvs = [v for _, v in simp_comp.get_simplices()]
        self.min_fv, self.max_fv = min(fvs), max(fvs)

        Filtration.__init__(self, simp_comp, points, display_type='color_change', **kwargs)

        max_length = np.max(np.linalg.norm(self.points, axis=1)) * 1.2
        line_vector = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else \
            np.array([-self.normal_vector[1] / self.normal_vector[0], 1, 0])
        line_vector /= np.linalg.norm(line_vector)

        p1 = -max_length * line_vector + (self.min_fv - self.CONFIG['plane_offset']) * self.normal_vector
        p2 = max_length * line_vector + (self.min_fv - self.CONFIG['plane_offset']) * self.normal_vector
        self.line = Line(p1, p2, color=self.CONFIG['plane_color'], width=self.CONFIG['plane_width'])

    def animate_filtration(self, offset=None):
        if offset is None:
            offset = self.CONFIG['plane_offset']

        if not self.filtered:
            anims, fvs = [], []
            for s, v in self.simp_comp.get_filtration():
                anims.append(ApplyFunction(self.change_simplex, self.mobject_dict[str(s)], run_time=0.01))
                fvs.append(v)

            fvs = np.array(fvs)
            fvs -= np.min(fvs)
            fvs /= np.max(fvs)
            fvs += offset

            self.add(self.line)
            self.line.generate_target()
            self.line.target.shift((self.max_fv - self.min_fv + 2 * offset) * self.normal_vector)

            anims.append(MoveToTarget(self.line, rate_func=linear, run_time=1 + 2 * offset))

            anim_grp = AnimationGroup(*anims, lag_ratio=0)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
                end_time = v + a.get_run_time()
                anim_grp.anims_with_timings.append(
                    (a, v, end_time)
                )
                print((v, end_time))
            anim_grp.anims_with_timings.append((anims[-1], 0, 1 + 2 * offset))

            if anim_grp.anims_with_timings:
                anim_grp.max_end_time = np.max([
                    awt[2] for awt in anim_grp.anims_with_timings
                ])
            else:
                anim_grp.max_end_time = 0
            if anim_grp.run_time is None:
                anim_grp.run_time = anim_grp.max_end_time

            anim_grp.max_end_time += offset
            anim_grp.run_time += offset
            self.filtered = True
            return anim_grp
        else:
            return AnimationGroup([])

    @property
    def sweeping_plane(self):
        return self.line

    def add_sweeping_plane(self):
        self.add(self.line)
        return self

    def remove_sweeping_plane(self):
        self.remove(self.line)
        return self
