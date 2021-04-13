#!/usr/bin/env python3

from simplicial_complex import *

import numpy as np
import gudhi as g
from manimlib.imports import *
from manimlib.mobject.geometry import Line


# TODO: Fix opacity adaption in filtering
# TODO: Keep 0-simplices in foreground
# TODO: Use Ripser for VR with 2 simplices

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

    def __init__(self, simp_comp: g.SimplexTree, points: np.ndarray, display_type='color_change', offset=0.5, **kwargs):
        super().__init__(simp_comp, points, True, **kwargs)

        assert display_type in ['color_change', 'appearing']
        self.display_type = display_type
        self.offset = offset

        fvs = [v for _, v in simp_comp.get_simplices()]
        self.min_fv, self.max_fv = min(fvs), max(fvs)
        self.current_fv = self.min_fv - self.offset

        if self.display_type == 'color_change':
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
                    self.mobject_dict[str(s)].set_fill(self.mobject_dict[str(s)].fill_color, 0)
                    self.mobject_dict[str(s)].set_stroke(self.mobject_dict[str(s)].fill_color, 0)

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

    def revert_simplex(self, s):
        if self.display_type == 'color_change':
            if isinstance(s, Polygon):
                s.set_fill(color=self.CONFIG['pre_filter_color'], opacity=self.CONFIG['pre_filter_opacity'](
                                                         self.CONFIG['simplex_opacity_2']))
                s.set_stroke(color=self.CONFIG['pre_filter_color'], opacity=self.CONFIG['pre_filter_opacity'](
                                                         self.CONFIG['simplex_opacity_2']), width=0)
            else:
                s.set_fill(color=self.CONFIG['pre_filter_color'], opacity=0.5)
                s.set_stroke(color=self.CONFIG['pre_filter_color'], width=s.stroke_width, opacity=0.5)
        else:
            if not isinstance(s, Dot):
                s.set_fill(WHITE, 0)
                s.set_stroke(WHITE, 0, 0)

        return s

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv + self.offset

        if to_fv >= self.current_fv:
            anims, fvs = [], []
            for s, v in self.simp_comp.get_filtration():
                if to_fv >= v >= self.current_fv:
                    anims.append(ApplyFunction(self.change_simplex, self.mobject_dict[str(s)], run_time=0.01))
                    fvs.append(v)

            fvs = np.array(fvs)
            fvs -= self.current_fv

            anim_grp = AnimationGroup(*anims)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
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

            anim_grp.max_end_time = to_fv - self.current_fv
            anim_grp.run_time = to_fv - self.current_fv
            self.current_fv = to_fv
            return anim_grp
        else:
            return AnimationGroup([])

    def animate_reverse_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.min_fv - self.offset

        if to_fv <= self.current_fv:
            anims, fvs = [], []
            rev_filt = [t for t in self.simp_comp.get_filtration()]
            rev_filt.sort(key=lambda x: x[1], reverse=True)
            for s, v in rev_filt:
                if to_fv <= v <= self.current_fv:
                    anims.append(ApplyFunction(self.revert_simplex, self.mobject_dict[str(s)], run_time=0.01))
                    fvs.append(v)

            fvs = -np.array(fvs)
            fvs += self.current_fv

            anim_grp = AnimationGroup(*anims)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
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

            anim_grp.max_end_time = -(to_fv - self.current_fv)
            anim_grp.run_time = -(to_fv - self.current_fv)
            self.current_fv = to_fv
            return anim_grp
        else:
            return AnimationGroup([])


class SweepingPlaneFiltration(Filtration):

    def __init__(self, simp_comp: g.SimplexTree, points: np.ndarray, normal_vector, plane_computed=False,
                 plane_color=YELLOW, plane_width=1, plane_expansion=1.2, **kwargs):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.CONFIG['plane_color'] = plane_color
        self.CONFIG['plane_width'] = plane_width
        self.CONFIG['plane_expansion'] = plane_expansion

        if not plane_computed:
            for s, _ in simp_comp.get_simplices():
                simp_comp.assign_filtration(s, max([points[vertex].dot(self.normal_vector) for vertex in s]))

        Filtration.__init__(self, simp_comp, points, display_type='color_change', **kwargs)

        max_length = np.max(np.linalg.norm(self.points, axis=1)) * self.CONFIG['plane_expansion']
        line_vector = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else \
            np.array([-self.normal_vector[1] / self.normal_vector[0], 1, 0])
        line_vector /= np.linalg.norm(line_vector)

        p1 = -max_length * line_vector + self.current_fv * self.normal_vector
        p2 = max_length * line_vector + self.current_fv * self.normal_vector
        self.line = Line(p1, p2, color=self.CONFIG['plane_color'], width=self.CONFIG['plane_width'])

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv + self.offset

        if to_fv >= self.current_fv:
            anims, fvs = [], []
            for s, v in self.simp_comp.get_filtration():
                if self.current_fv <= v <= to_fv:
                    anims.append(ApplyFunction(self.change_simplex, self.mobject_dict[str(s)], run_time=0.01))
                    fvs.append(v)

            fvs = np.array(fvs)
            fvs -= self.current_fv

            self.add(self.line)
            self.line.generate_target()
            self.line.target.shift((to_fv - self.current_fv) * self.normal_vector)

            anims.append(MoveToTarget(self.line, rate_func=linear, run_time=to_fv - self.current_fv))

            anim_grp = AnimationGroup(*anims, lag_ratio=0)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
                end_time = v + a.get_run_time()
                anim_grp.anims_with_timings.append(
                    (a, v, end_time)
                )
            anim_grp.anims_with_timings.append((anims[-1], 0, to_fv - self.current_fv))

            if anim_grp.anims_with_timings:
                anim_grp.max_end_time = np.max([
                    awt[2] for awt in anim_grp.anims_with_timings
                ])
            else:
                anim_grp.max_end_time = 0
            if anim_grp.run_time is None:
                anim_grp.run_time = anim_grp.max_end_time

            anim_grp.max_end_time = to_fv - self.current_fv
            anim_grp.run_time = to_fv - self.current_fv
            self.current_fv = to_fv
            return anim_grp
        else:
            return AnimationGroup([])

    def animate_reverse_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.min_fv - self.offset

        if to_fv <= self.current_fv:
            anims, fvs = [], []
            rev_filt = [t for t in self.simp_comp.get_filtration()]
            rev_filt.sort(key=lambda x: x[1], reverse=True)
            for s, v in rev_filt:
                if self.current_fv >= v >= to_fv:
                    anims.append(ApplyFunction(self.revert_simplex, self.mobject_dict[str(s)], run_time=0.01))
                    fvs.append(v)

            fvs = -np.array(fvs)
            fvs += self.current_fv

            self.add(self.line)
            self.line.generate_target()
            self.line.target.shift((to_fv - self.current_fv) * self.normal_vector)

            anims.append(MoveToTarget(self.line, rate_func=linear, run_time=self.current_fv - to_fv))

            anim_grp = AnimationGroup(*anims, lag_ratio=0)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
                end_time = v + a.get_run_time()
                anim_grp.anims_with_timings.append(
                    (a, v, end_time)
                )
            anim_grp.anims_with_timings.append((anims[-1], 0, self.current_fv - to_fv))

            if anim_grp.anims_with_timings:
                anim_grp.max_end_time = np.max([
                    awt[2] for awt in anim_grp.anims_with_timings
                ])
            else:
                anim_grp.max_end_time = 0
            if anim_grp.run_time is None:
                anim_grp.run_time = anim_grp.max_end_time

            anim_grp.max_end_time = -(to_fv - self.current_fv)
            anim_grp.run_time = -(to_fv - self.current_fv)
            self.current_fv = to_fv
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


class ExpandingBallFiltration(Filtration):

    def __init__(self, simp_comp: g.SimplexTree, points: np.ndarray, expansion_func=None, ball_color=ORANGE,
                 ball_opacity=0.2, ball_stroke_width=0.1, **kwargs):
        Filtration.__init__(self, simp_comp, points, 'appearing', **kwargs)
        if expansion_func is None:
            expansion_func = lambda x: x if x >= 0 else 0
        self.expansion_func = expansion_func

        self.CONFIG['ball_color'] = ball_color
        self.CONFIG['ball_opacity'] = ball_opacity
        self.CONFIG['ball_stroke_width'] = ball_stroke_width

        self.circles = []

        for p in self.points:
            c = Circle(arc_center=p, radius=self.expansion_func(self.current_fv))
            c.set_fill(self.CONFIG['ball_color'], self.CONFIG['ball_opacity'])
            c.set_stroke(self.CONFIG['ball_color'], self.CONFIG['ball_stroke_width'], self.CONFIG['ball_opacity'])
            self.add(c)
            self.circles.append(c)

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv + self.offset

        if to_fv >= self.current_fv:
            anims, fvs = [], []
            for s, v in self.simp_comp.get_filtration():
                if to_fv >= v >= self.current_fv:
                    anims.append(ApplyFunction(self.change_simplex, self.mobject_dict[str(s)], run_time=0.01))
                    fvs.append(v)
            for c, p in zip(self.circles, self.points):
                anims.append(Transform(c, target_mobject=Circle(arc_center=p, radius=self.expansion_func(to_fv)),
                                       run_time=to_fv - max(self.current_fv, 0), rate_func=linear))
                fvs.append(max(self.current_fv, 0))

            fvs = np.array(fvs)
            fvs -= self.current_fv

            anim_grp = AnimationGroup(*anims)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
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

            anim_grp.max_end_time = to_fv - self.current_fv
            anim_grp.run_time = to_fv - self.current_fv
            self.current_fv = to_fv
            return anim_grp
        else:
            return AnimationGroup([])

    def animate_reverse_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = 0
        to_fv = max(to_fv, 0)

        if 0 <= to_fv <= self.current_fv:
            anims, fvs = [], []
            rev_filt = [t for t in self.simp_comp.get_filtration()]
            rev_filt.sort(key=lambda x: x[1], reverse=True)
            for s, v in rev_filt:
                if self.current_fv >= v >= to_fv:
                    anims.append(ApplyFunction(self.revert_simplex, self.mobject_dict[str(s)], run_time=0.01))
                    fvs.append(v)
            for c, p in zip(self.circles, self.points):
                anims.append(Transform(c, target_mobject=Circle(arc_center=p, radius=self.expansion_func(to_fv)),
                                       run_time=-(max(to_fv, 0) - max(self.current_fv, 0)), rate_func=linear))
                fvs.append(max(self.current_fv, 0))

            fvs = -np.array(fvs)
            fvs += self.current_fv

            anim_grp = AnimationGroup(*anims, lag_ratio=0)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
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

            anim_grp.max_end_time = -(to_fv - self.current_fv)
            anim_grp.run_time = -(to_fv - self.current_fv)
            self.current_fv = to_fv
            return anim_grp
        else:
            return AnimationGroup([])


class RipsFiltration(ExpandingBallFiltration):

    def __init__(self, points: np.ndarray, max_radius, **kwargs):
        self.max_radius = max_radius
        vr = g.RipsComplex(points=points, max_edge_length=self.max_radius).create_simplex_tree()

        def expansion_func(x):
            if x <= 0:
                return 0
            elif x <= max_radius:
                return x
            else:
                return max_radius

        ExpandingBallFiltration.__init__(self, vr, points, expansion_func, **kwargs)

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv + self.offset
        to_fv = min(to_fv, self.max_radius)
        return ExpandingBallFiltration.animate_filtration(self, to_fv)


class CechFiltration(ExpandingBallFiltration):

    def __init__(self, points: np.ndarray, max_radius, **kwargs):
        self.max_radius = max_radius
        alpha = g.AlphaComplex(points=points).create_simplex_tree(max_alpha_square=self.max_radius**2)

        def expansion_func(x):
            if x <= 0:
                return 0
            elif x <= max_radius:
                return x
            else:
                return max_radius

        ExpandingBallFiltration.__init__(self, alpha, points, expansion_func, **kwargs)

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv + self.offset
        to_fv = min(to_fv, self.max_radius)
        return ExpandingBallFiltration.animate_filtration(self, to_fv)

