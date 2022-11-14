#!/usr/bin/env python3

from simplicial_complex import *

import numpy as np
import gudhi as g
from manimlib.imports import *
from manimlib.mobject.geometry import Line


# TODO: Fix opacity adaption in filtering
# TODO: Keep 0-simplices in foreground

class Filtration(SimplicialComplex):

    CONFIG = {
        'simplex_color_0': RED,
        'simplex_width_0': 0.05,
        'simplex_opacity_0': 1,
        'simplex_color_1': WHITE,
        'simplex_width_1': 2,
        'simplex_opacity_1': 1,
        'simplex_color_2': WHITE,
        'simplex_opacity_2': 0.5,
        'pre_filter_color': GRAY,
        'pre_filter_opacity': lambda x: x / 2
    }

    def __init__(self, simp_comp: g.SimplexTree, point_cloud: np.ndarray, display_type='color_change', offset=0.5, **kwargs):
        """
        Most basic instance of animating a Filtration. Simplices change colour at their birth time and bars grow
        linearly with time.
        :param simp_comp: Filtration in form of Gudhi SimplexTree
        :param point_cloud: 2d coordinates of points in the filtration as numpy array
        :param display_type: Whether simplices should change their colour or appear at their birth time
        :param offset: The subtracts/adds offset to min/max values in filtration and starts visualisation at those offset values
        :param **kwargs: Further arguments on visual characteristics of simplices
        """
        SimplicialComplex.__init__(self, simp_comp, point_cloud, True, **kwargs)

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
                                                         width=self.mobject_dict[str(s)].stroke_width,
                                                         opacity=0.5)
        else:
            for s, _ in self.simp_comp.get_simplices():
                if len(s) > 1:
                    self.mobject_dict[str(s)].set_fill(self.mobject_dict[str(s)].fill_color, 0)
                    self.mobject_dict[str(s)].set_stroke(self.mobject_dict[str(s)].fill_color, 0)

    def change_simplex(self, s):  # In 'color_change' mode, this function turns a simplex into filtered state
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

    def revert_simplex(self, s):  # In 'color_change' mode, this function turns a simplex into unfiltered state
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
        # Animates a filtration in 'positive direction' from current_fv to to_fv
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
        # Animates a filtration in 'negative/backward direction' from current_fv to to_fv
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

    def __init__(self, simp_comp: g.SimplexTree, point_cloud: np.ndarray, normal_vector, plane_computed=False,
                 plane_color=YELLOW, plane_width=1, plane_expansion=1.2, extra_mobjects=[], **kwargs):
        """
        Class animating the persistent homology of a sweeping plane filtration of a simplicial comples
        :param simp_comp: A simplicial complex in form of a Gudhi SimplexTree
        :param point_cloud: A numpy array corresponding to the 2d positions of the indices in simp_comp
        :param normal_vector: The normal vector of the sweeping plane
        :param plane_computed: whether the filtration wrt to the plane is already computed and stored as
                                filtration values in simp_comp
        :param plane_color: What color to animate the plane in
        :param plane_width: Width of the plane in the animation in pt
        :param plane_expansion: Width of the plane
        :param extra_mobjects: any extra manim mobjects to add to the scene
        :param kwargs: additional arguments to pass on to the Filtration class
        """
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.CONFIG['plane_color'] = plane_color
        self.CONFIG['plane_width'] = plane_width
        self.CONFIG['plane_expansion'] = plane_expansion

        if point_cloud.shape[1] == 2:
            point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), axis=1)

        if not plane_computed:
            for s, _ in simp_comp.get_simplices():
                simp_comp.assign_filtration(s, max([point_cloud[vertex].dot(self.normal_vector) for vertex in s]))

        Filtration.__init__(self, simp_comp, point_cloud, display_type='color_change', **kwargs)

        max_length = np.max(np.linalg.norm(self.point_cloud, axis=1)) * self.CONFIG['plane_expansion']
        line_vector = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else \
            np.array([-self.normal_vector[1] / self.normal_vector[0], 1, 0])
        line_vector = line_vector / np.linalg.norm(line_vector)

        p1 = -max_length * line_vector + self.current_fv * self.normal_vector
        p2 = max_length * line_vector + self.current_fv * self.normal_vector
        self.line = Line(p1, p2, color=self.CONFIG['plane_color'], width=self.CONFIG['plane_width'])
        self.add(self.line)
        self.add(*extra_mobjects)

    def scale(self, scale_factor, **kwargs):
        super(SweepingPlaneFiltration, self).scale(scale_factor, **kwargs)
        self.normal_vector *= scale_factor
        return self

    def animate_filtration(self, to_fv=None):
        # Modification of inherited method to move sweeping plane in addition to changing simplices
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

            #self.add(self.line)
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
            anim_grp.anims_with_timings.append((anims[-1], 0, max(to_fv - self.current_fv, 2**-10)))

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
        # Modification of inherited method to move sweeping plane in addition to changing simplices
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

            anims.append(MoveToTarget(self.line, rate_func=linear, run_time=max(self.current_fv - to_fv, 2**-10)))

            anim_grp = AnimationGroup(*anims, lag_ratio=0)
            anim_grp.anims_with_timings = []
            for a, v in zip(anims, fvs):
                end_time = v + a.get_run_time()
                anim_grp.anims_with_timings.append(
                    (a, v, end_time)
                )
            anim_grp.anims_with_timings.append((anims[-1], 0, max(self.current_fv - to_fv, 2**-10)))

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

    def update_direction(self, new_direction):
        # changes the direction of filtration
        a = self.animate_reverse_filtration()
        self.normal_vector = new_direction / np.linalg.norm(new_direction)
        self.current_fv = self.min_fv - self.offset

        for s, _ in self.simp_comp.get_simplices():
            self.simp_comp.assign_filtration(s, max([self.point_cloud[vertex].dot(self.normal_vector) for vertex in s]))

        max_length = np.max(np.linalg.norm(self.point_cloud, axis=1)) * self.CONFIG['plane_expansion']
        line_vector = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else \
            np.array([-self.normal_vector[1] / self.normal_vector[0], 1, 0])
        line_vector = line_vector / np.linalg.norm(line_vector)

        p1 = -max_length * line_vector + self.current_fv * self.normal_vector
        p2 = max_length * line_vector + self.current_fv * self.normal_vector
        line = Line(p1+self.get_center(), p2+self.get_center(), color=self.CONFIG['plane_color'], width=self.CONFIG['plane_width'])

        return AnimationGroup(a, Transform(self.line, line), lag_ratio=1, run_time=2**-10)

    @property
    def sweeping_plane(self):
        return self.line

    def add_sweeping_plane(self):
        # makes the sweeping plane visible
        self.add(self.line)
        return self

    def remove_sweeping_plane(self):
        # makes the sweeping plane invisible
        self.remove(self.line)
        return self

    @property
    def size(self):
        return super(SweepingPlaneFiltration, self).size * self.plane_expansion


class ExpandingBallFiltration(Filtration):

    def __init__(self, simp_comp: g.SimplexTree, point_cloud: np.ndarray, expansion_func=None, ball_color=ORANGE,
                 ball_opacity=0.2, ball_stroke_width=0.1, **kwargs):
        Filtration.__init__(self, simp_comp, point_cloud, 'appearing', **kwargs)
        if expansion_func is None:
            expansion_func = lambda x: x if x >= 0 else 0
        self.expansion_func = expansion_func

        self.CONFIG['ball_color'] = ball_color
        self.CONFIG['ball_opacity'] = ball_opacity
        self.CONFIG['ball_stroke_width'] = ball_stroke_width

        self.circles = []

        for p in self.point_cloud:
            c = Circle(arc_center=p, radius=self.expansion_func(self.current_fv))
            c.set_fill(self.CONFIG['ball_color'], self.CONFIG['ball_opacity'])
            c.set_stroke(self.CONFIG['ball_color'], self.CONFIG['ball_stroke_width'], self.CONFIG['ball_opacity'])
            self.add(c)
            self.circles.append(c)

    def animate_filtration(self, to_fv=None):
        # Modification of inherited method to expand balls in addition to changing simplices
        if to_fv is None:
            to_fv = self.max_fv + self.offset

        if to_fv >= self.current_fv:
            anims, fvs = [], []
            for s, v in self.simp_comp.get_filtration():
                if to_fv >= v >= self.current_fv:
                    anims.append(ApplyFunction(self.change_simplex, self.mobject_dict[str(s)], run_time=0.01))
                    fvs.append(v)
            for c, p in zip(self.circles, self.point_cloud):
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
        # Modification of inherited method to un-expand balls in addition to changing simplices
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
            for c, p in zip(self.circles, self.point_cloud):
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

    @property
    def size(self):
        return super(ExpandingBallFiltration, self).size + self.max_fv + self.offset


class RipsFiltration(ExpandingBallFiltration):

    def __init__(self, point_cloud: np.ndarray, max_radius, max_dimension=2, **kwargs):
        """
        Animates a 2d Vietoris Rips Filtration (points with expanding balls on the)
        :param point_cloud: Points used for VR filtration (2d numpy array)
        :param max_radius: Maximum radius to compute VR filtration for
        :param max_dimension: Maximum dimension in which to compute persistent homology
        :param kwargs: Further parameters to pass to ExpandingBallFiltration and Filtration classes
        """
        self.max_radius = max_radius
        self.max_dimension = max_dimension
        vr = g.RipsComplex(points=point_cloud, max_edge_length=self.max_radius)
        vr = vr.create_simplex_tree(max_dimension=self.max_dimension)

        def expansion_func(x):
            if x <= 0:
                return 0
            elif x <= max_radius:
                return x
            else:
                return max_radius

        ExpandingBallFiltration.__init__(self, vr, point_cloud, expansion_func, **kwargs)

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv + self.offset
        to_fv = min(to_fv, self.max_radius)
        return ExpandingBallFiltration.animate_filtration(self, to_fv)


class CechFiltration(ExpandingBallFiltration):

    def __init__(self, point_cloud: np.ndarray, max_radius, **kwargs):
        """
        Animates a 2d Cech Filtration (points with expanding balls)
        :param point_cloud: Points used for Cech filtration (2d numpy array)
        :param max_radius: Maximum radius to compute Chech filtration for
        :param kwargs: Further parameters to pass to ExpandingBallFiltration and Filtration classes
        """
        self.max_radius = max_radius
        alpha = g.AlphaComplex(points=point_cloud).create_simplex_tree(max_alpha_square=self.max_radius**2)

        for simplex, fv in alpha.get_filtration():
            alpha.assign_filtration(simplex, np.sqrt(fv))

        def expansion_func(x):
            if x <= 0:
                return 0
            elif x <= max_radius:
                return x
            else:
                return max_radius

        ExpandingBallFiltration.__init__(self, alpha, point_cloud, expansion_func, **kwargs)

    def animate_filtration(self, to_fv=None):
        if to_fv is None:
            to_fv = self.max_fv + self.offset
        to_fv = min(to_fv, self.max_radius)
        return ExpandingBallFiltration.animate_filtration(self, to_fv)

