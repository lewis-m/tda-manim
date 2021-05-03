#!/usr/bin/env python3

import numpy as np
from scipy.integrate import quad
import itertools as it

from manimlib.imports import *
from manimlib.mobject.geometry import Line


# TODO: fix axes
# TODO: fix multiple bar issue
# TODO: fix moving ECTFunction as group issue


class ECTFunction(VGroup):

    CONFIG = {
        "x_axis_width": 9,
        "x_tick_frequency": 1,
        "x_leftmost_tick": None,  # Change if different from x_min
        "x_labeled_nums": None,
        "x_axis_label": "",
        "y_axis_height": 6,
        "y_tick_frequency": 1,
        "y_bottom_tick": None,  # Change if different from y_min
        "y_labeled_nums": None,
        "y_axis_label": "ECT",
        "discrete_graph_color": RED,
        "continuous_graph_color": BLUE,
        "axes_color": GREY,
        "graph_origin": 2.5 * DOWN + 4 * LEFT,
        "exclude_zero_label": True,
        "area_opacity": 0.8,
        "num_rects": 50,
    }

    def __init__(self, x_min, x_max, y_min, y_max, x_crit, ect_discr, extra_mobjects=[],
                 previous_mobjects=[], **kwargs):
        super().__init__(**kwargs)
        self.ect_discr = ect_discr
        self.x_min, self.x_max = x_min, x_max
        self.x_crit = x_crit.tolist()
        self.x_crit.append(self.x_max)
        self.x_crit = np.array(self.x_crit)
        self.ect_smooth, self.mean = self.construct_smooth_ect()
        self.y_min, self.y_max = y_min - self.mean, y_max
        self.setup_axes()

        self.current_fv_d, self.current_fv_s = x_min, x_min
        self.discrete_ect_mobjects = []
        p = np.array([*self.coords_to_point(self.x_min, self.ect_discr(self.x_crit[0]))])
        l = Line(p, p, color=self.CONFIG['discrete_graph_color'])
        self.discrete_ect_mobjects.append(l)
        self.add(l)

        def parameterized_function(alpha):
            x = interpolate(self.x_min, self.current_fv_s, alpha)
            y = self.ect_smooth(x)
            if not np.isfinite(y):
                y = self.y_max
            return self.coords_to_point(x, y)

        graph = ParametricFunction(
            parameterized_function,
            color=self.CONFIG['continuous_graph_color']
        )
        self.add(graph)
        self.smooth_ect_mobjects = [graph]

        self.discrete_dump, self.smooth_dump = [], []

        self.add(*extra_mobjects)
        self.add(*previous_mobjects)

    def construct_smooth_ect(self):
        mean = sum([self.ect_discr(self.x_crit[i]) * (self.x_crit[i + 1] - self.x_crit[i])
                    for i in range(len(self.x_crit) - 1)]) / (self.x_max - self.x_min)

        def smooth_ect(x):
            return sum([self.ect_discr(self.x_crit[i]) * max(min(self.x_crit[i + 1], x) - self.x_crit[i], 0)
                        for i in range(len(self.x_crit) - 1)]) - (x - self.x_min) * mean

        return smooth_ect, mean

    def setup_axes(self):
        """
        This method sets up the axes of the graph.
        """
        """
                This method sets up the axes of the graph.

                Parameters
                ----------
                animate (bool=False)
                    Whether or not to animate the setting up of the Axes.
                """
        # TODO, once eoc is done, refactor this to be less redundant.
        x_num_range = float(self.x_max - self.x_min)
        self.space_unit_to_x = self.x_axis_width / x_num_range
        if self.x_labeled_nums is None:
            self.x_labeled_nums = []
        if self.x_leftmost_tick is None:
            self.x_leftmost_tick = self.x_min
        x_axis = NumberLine(
            x_min=self.x_min,
            x_max=self.x_max + 0.2,
            unit_size=self.space_unit_to_x,
            tick_frequency=self.x_tick_frequency,
            leftmost_tick=self.x_leftmost_tick,
            numbers_with_elongated_ticks=self.x_labeled_nums,
            color=self.axes_color,
            stroke_width=1,
            include_tip=True
        )
        x_axis.shift(self.graph_origin - x_axis.number_to_point(0))
        if len(self.x_labeled_nums) > 0:
            if self.exclude_zero_label:
                self.x_labeled_nums = [x for x in self.x_labeled_nums if x != 0]
            x_axis.add_numbers(*self.x_labeled_nums)
        if self.x_axis_label:
            x_label = TextMobject(self.x_axis_label)
            x_label.next_to(
                x_axis.get_tick_marks(), UP + RIGHT,
                buff=SMALL_BUFF
            )
            x_label.shift_onto_screen()
            x_axis.add(x_label)
            self.x_axis_label_mob = x_label

        y_num_range = float(self.y_max - self.y_min)
        self.space_unit_to_y = self.y_axis_height / y_num_range

        if self.y_labeled_nums is None:
            self.y_labeled_nums = []
        if self.y_bottom_tick is None:
            self.y_bottom_tick = self.y_min
        y_axis = NumberLine(
            x_min=self.y_min - 0.1,
            x_max=self.y_max + 0.1,
            unit_size=self.space_unit_to_y,
            tick_frequency=self.y_tick_frequency,
            leftmost_tick=self.y_bottom_tick,
            numbers_with_elongated_ticks=self.y_labeled_nums,
            color=self.axes_color,
            line_to_number_vect=LEFT,
            label_direction=UP,
            stroke_width=1,
            include_tip=True
        )
        y_axis.shift(self.graph_origin + y_axis.number_to_point(self.x_min + 0.1)[0] * RIGHT)
        y_axis.rotate(np.pi / 2, about_point=y_axis.number_to_point(0))
        if len(self.y_labeled_nums) > 0:
            if self.exclude_zero_label:
                self.y_labeled_nums = [y for y in self.y_labeled_nums if y != 0]
            y_axis.add_numbers(*self.y_labeled_nums)
        if self.y_axis_label:
            y_label = TextMobject(self.y_axis_label)
            y_label.next_to(
                y_axis.get_corner(UP + LEFT), UP + LEFT,
                buff=SMALL_BUFF
            )
            #y_label.shift_onto_screen()
            y_axis.add(y_label)
            self.y_axis_label_mob = y_label

            self.add(x_axis, y_axis)
        self.x_axis, self.y_axis = self.axes = VGroup(x_axis, y_axis)

        return self

    def coords_to_point(self, x, y):
        """
        The graph is smaller than the scene.
        Because of this, coordinates in the scene don't map
        to coordinates on the graph.
        This method returns a scaled coordinate for the graph,
        given cartesian coordinates that correspond to the scene..

        Parameters
        ----------
        x : (int,float)
            The x value

        y : (int,float)
            The y value

        Returns
        -------
        np.ndarray
            The array of the coordinates.
        """
        assert (hasattr(self, "x_axis") and hasattr(self, "y_axis"))
        result = self.x_axis.number_to_point(x)[0] * RIGHT
        result += self.y_axis.number_to_point(y)[1] * UP
        return result

    def animate_discrete_graph(self, to_fv):
        if self.current_fv_d <= to_fv <= self.x_max:
            anims, fvs, cps = [], [], []
            for i, c in enumerate(self.x_crit):
                if self.current_fv_d < c <= to_fv:
                    cps.append((c, i))

            if cps[-1][0] != to_fv:
                cps.append((to_fv, cps[-1][1]+1))
            anims.append(Transform(self.discrete_ect_mobjects[-1],
                                   Line([*self.coords_to_point(self.x_crit[cps[0][1]-1], self.ect_discr(self.x_crit[cps[0][1]-1]))],
                                        [*self.coords_to_point(cps[0][0], self.ect_discr(self.x_crit[cps[0][1]-1]))],
                                        color=self.CONFIG['discrete_graph_color']),
                                   run_time=cps[0][0]-self.x_crit[cps[0][1]-1], rate_func=linear))
            fvs.append(self.current_fv_d)

            for i in range(len(cps) - 1):
                l = Line([*self.coords_to_point(cps[i][0], self.ect_discr(cps[i][0]))],
                                               [*self.coords_to_point(cps[i+1][0], self.ect_discr(cps[i][0]))],
                                               color=self.CONFIG['discrete_graph_color'])
                anims.append(ShowCreation(l, run_time=cps[i+1][0]-cps[i][0], rate_func=linear))
                self.discrete_ect_mobjects.append(l)
                fvs.append(cps[i][0])

            fvs = np.array(fvs)
            fvs -= self.current_fv_d

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

            anim_grp.max_end_time = to_fv - self.current_fv_d
            anim_grp.run_time = to_fv - self.current_fv_d
            self.current_fv_d = to_fv
            return anim_grp
        else:
            return AnimationGroup([])

    def animate_reverse_discrete_graph(self, to_fv):
        if self.x_min <= to_fv <= self.current_fv_d:
            anims, fvs, cps = [], [], []
            iterlist = [n for n in enumerate(self.x_crit)]
            iterlist.reverse()
            for i, c in iterlist:
                if self.current_fv_d > c > to_fv:
                    cps.append((c, i))
            if len(cps) == 0:
                i = min([c for c in range(len(self.x_crit)) if self.x_crit[c] > self.current_fv_d])
                cps.append((self.current_fv_d, i))
            if cps[0][0] != self.current_fv_d:
                cps.insert(0, (self.current_fv_d, cps[0][1]))
            for i in range(1, len(cps)):
                anims.append(Uncreate(self.discrete_ect_mobjects[-1], run_time=abs(cps[i-1][0]-cps[i][0]),
                                      rate_func=lambda t: 1 - t))
                fvs.append(cps[i-1][0])
                self.discrete_ect_mobjects.pop(-1)

            anims.append(Transform(self.discrete_ect_mobjects[-1],
                                   Line([*self.coords_to_point(self.x_crit[cps[-1][1]-1], self.ect_discr(self.x_crit[cps[-1][1]-1]))],
                                        [*self.coords_to_point(to_fv, self.ect_discr(self.x_crit[cps[-1][1]-1]))],
                                        color=self.CONFIG['discrete_graph_color']),
                                   run_time=cps[-1][0]-to_fv, rate_func=linear))
            fvs.append(cps[-1][0])

            fvs = -np.array(fvs)
            fvs += self.current_fv_d

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

            anim_grp.max_end_time = -(to_fv - self.current_fv_d)
            anim_grp.run_time = -(to_fv - self.current_fv_d)
            self.current_fv_d = to_fv
            return anim_grp
        else:
            return AnimationGroup([])

    def animate_smooth_graph(self, to_fv):
        if self.current_fv_s <= to_fv <= self.x_max:
            def parameterized_function(alpha):
                x = interpolate(self.current_fv_s, to_fv, alpha)
                y = self.ect_smooth(x)
                if not np.isfinite(y):
                    y = self.y_max
                return self.coords_to_point(x, y)

            new_graph = ParametricFunction(
                parameterized_function,
                color=self.CONFIG['continuous_graph_color']
            )
            self.smooth_ect_mobjects.append(new_graph)
            anims = [ShowCreation(new_graph, run_time=to_fv-self.current_fv_s, rate_func=linear)]
            self.current_fv_s = to_fv
            return AnimationGroup(*anims)
        else:
            return AnimationGroup([])

    def animate_reverse_smooth_graph(self, to_fv):
        if self.x_min <= to_fv <= self.current_fv_s:
            def parameterized_function_rem(alpha):
                x = interpolate(to_fv, self.current_fv_s, alpha)
                y = self.ect_smooth(x)
                if not np.isfinite(y):
                    y = self.y_max
                return self.coords_to_point(x, y)

            new_graph_rem = ParametricFunction(
                parameterized_function_rem,
                color=self.CONFIG['continuous_graph_color']
            )

            def parameterized_function_perm(alpha):
                x = interpolate(self.x_min, to_fv, alpha)
                y = self.ect_smooth(x)
                if not np.isfinite(y):
                    y = self.y_max
                return self.coords_to_point(x, y)

            new_graph_perm = ParametricFunction(
                parameterized_function_perm,
                color=self.CONFIG['continuous_graph_color']
            )
            self.add(new_graph_rem, new_graph_perm)
            self.remove(*self.smooth_ect_mobjects)

            anims = [Uncreate(new_graph_rem, run_time=self.current_fv_s-to_fv, rate_func=lambda t: 1-t)]
            anims += [ShowCreation(new_graph_perm, run_time=2**-10)]
            anims += [Uncreate(m, run_time=2**-10) for m in self.smooth_ect_mobjects]
            self.smooth_ect_mobjects = [new_graph_perm]
            self.current_fv_s = to_fv
            return AnimationGroup(*anims, lag_ratio=0)
        else:
            return AnimationGroup([])

    def animate_discrete_mean_shift(self):
        if self.ect_discr(self.x_min) == 0:
            def shift_down(line):
                line.shift(self.mean * DOWN)
                return line

            anims = [ApplyFunction(shift_down, line) for line in self.discrete_ect_mobjects]
            #self.ect_discr = lambda x: self.ect_discr(x) - self.mean
            return AnimationGroup(*anims)
        else:
            def shift_up(line):
                line.shift(self.mean * UP)
                return line

            anims = [ApplyFunction(shift_up, line) for line in self.discrete_ect_mobjects]
            #self.ect_discr = lambda x: self.ect_discr(x) + self.ect_discr(self.x_min)
            return AnimationGroup(*anims)

    def drop_current_graphs(self):
        self.discrete_dump += self.discrete_ect_mobjects
        self.smooth_dump += self.smooth_ect_mobjects
        self.current_fv_d, self.current_fv_s = self.x_min, self.x_min

        p = np.array([*self.coords_to_point(self.x_min, self.ect_discr(self.x_crit[0]))])
        l = Line(p, p, color=self.CONFIG['discrete_graph_color'])
        self.discrete_ect_mobjects = [l]

        def parameterized_function(alpha):
            x = interpolate(self.x_min, self.current_fv_s, alpha)
            y = self.ect_smooth(x)
            if not np.isfinite(y):
                y = self.y_max
            return self.coords_to_point(x, y)

        graph = ParametricFunction(
            parameterized_function,
            color=self.CONFIG['continuous_graph_color']
        )
        self.add(graph)
        self.smooth_ect_mobjects = [graph]

        return self

    @property
    def size(self):
        return max(self.x_max - self.x_min, self.y_max - self.y_min + 0.2)


class Barcode(VGroup):

    CONFIG = {
        'bar_color': BLUE,
        'homological_dimension': 0,
        "x_axis_width": 9,
        "x_tick_frequency": 1,
        "x_leftmost_tick": None,  # Change if different from x_min
        "x_labeled_nums": None,
        "x_axis_label": "",
        "y_axis_height": 6,
        "y_tick_frequency": 1,
        "y_bottom_tick": None,  # Change if different from y_min
        "y_labeled_nums": None,
        "y_axis_label": "birth / death time",
        "axes_color": GREY,
        "graph_origin": 2.5 * DOWN + 4 * LEFT,
        "exclude_zero_label": True,
        "area_opacity": 0.8,
        "num_rects": 50,
    }

    def __init__(self, bars, min_fv, max_fv):
        super().__init__()
        self.bars = bars
        self.min_fv, self.max_fv = min_fv, max_fv
        self.current_fv = self.min_fv
        assert self.min_fv < self.max_fv

        for i, bar in enumerate(self.bars):
            if bar[1] > self.max_fv:
                self.bars[i] = (bar[0], self.max_fv)
            if bar[0] < self.min_fv:
                self.bars[i] = (self.min_fv, bar[1])

        self.delta_y = 3 / (len(bars) + 2)
        self.animated_lines = dict({})
        self.setup_axes()

    def setup_axes(self):
        x_num_range = float(self.max_fv - self.min_fv)
        self.space_unit_to_x = self.x_axis_width / x_num_range
        if self.x_labeled_nums is None:
            self.x_labeled_nums = []
        if self.x_leftmost_tick is None:
            self.x_leftmost_tick = self.min_fv
        x_axis = NumberLine(
            x_min=self.min_fv,
            x_max=self.max_fv,
            unit_size=self.space_unit_to_x,
            tick_frequency=self.x_tick_frequency,
            leftmost_tick=self.x_leftmost_tick,
            numbers_with_elongated_ticks=self.x_labeled_nums,
            color=self.axes_color,
            stroke_width=1,
            include_tip=False
        )
        x_axis.shift(self.graph_origin - x_axis.number_to_point(0))
        if len(self.x_labeled_nums) > 0:
            if self.exclude_zero_label:
                self.x_labeled_nums = [x for x in self.x_labeled_nums if x != 0]
            x_axis.add_numbers(*self.x_labeled_nums)
        if self.x_axis_label:
            x_label = TextMobject(self.x_axis_label)
            x_label.next_to(
                x_axis.get_tick_marks(), UP + RIGHT,
                buff=SMALL_BUFF
            )
            x_label.shift_onto_screen()
            x_axis.add(x_label)
            self.x_axis_label_mob = x_label

        y_num_range = 3
        self.space_unit_to_y = self.y_axis_height / y_num_range

        if self.y_labeled_nums is None:
            self.y_labeled_nums = []
        if self.y_bottom_tick is None:
            self.y_bottom_tick = 0
        y_axis = NumberLine(
            x_min=0,
            x_max=3,
            unit_size=self.space_unit_to_y,
            tick_frequency=self.y_tick_frequency,
            leftmost_tick=self.y_bottom_tick,
            numbers_with_elongated_ticks=self.y_labeled_nums,
            color=self.axes_color,
            line_to_number_vect=LEFT,
            label_direction=UP,
            stroke_width=1,
            include_tip=False,
            include_ticks=False
        )
        y_axis.shift(self.graph_origin + y_axis.number_to_point(self.min_fv)[0] * RIGHT)
        y_axis.rotate(np.pi / 2, about_point=y_axis.number_to_point(0))
        if len(self.y_labeled_nums) > 0:
            if self.exclude_zero_label:
                self.y_labeled_nums = [y for y in self.y_labeled_nums if y != 0]
            y_axis.add_numbers(*self.y_labeled_nums)
        if self.y_axis_label:
            y_label = TextMobject(self.y_axis_label)
            y_label.next_to(
                y_axis.get_corner(UP + LEFT), UP + LEFT,
                buff=SMALL_BUFF
            )
            #y_label.shift_onto_screen()
            y_axis.add(y_label)
            self.y_axis_label_mob = y_label

            self.add(x_axis, y_axis)
        self.x_axis, self.y_axis = self.axes = VGroup(x_axis, y_axis)

        return self

    def coords_to_point(self, x, y):
        assert (hasattr(self, "x_axis") and hasattr(self, "y_axis"))
        result = self.x_axis.number_to_point(x)[0] * RIGHT
        result += self.y_axis.number_to_point(y)[1] * UP
        return result

    def animate_barcode(self, to_fv):
        if self.current_fv <= to_fv <= self.max_fv:
            anims, fvs = [], []
            for i, bar in enumerate(self.bars):
                if self.current_fv < bar[0] <= to_fv:
                    line = Line(self.coords_to_point(bar[0], self.delta_y * (i + 1)),
                                self.coords_to_point(min(bar[1], to_fv), self.delta_y * (i + 1)),
                                color=self.CONFIG['bar_color'])
                    anims.append(ShowCreation(line, run_time=min(bar[1], to_fv)-bar[0], rate_func=linear))
                    self.axes.add(line)
                    fvs.append(bar[0])
                    self.animated_lines[bar] = line
                elif bar[0] < self.current_fv:
                    line1 = self.animated_lines[bar]
                    line2 = Line(self.coords_to_point(bar[0], self.delta_y * (i + 1)),
                                self.coords_to_point(min(bar[1], to_fv), self.delta_y * (i + 1)),
                                color=self.CONFIG['bar_color'])
                    anims.append(Transform(line1, line2, run_time=min(bar[1], to_fv)-to_fv, rate_func=linear))
                    fvs.append(self.current_fv)

            fvs = np.array(fvs)
            fvs -= self.current_fv

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

            anim_grp.max_end_time = to_fv - self.current_fv
            anim_grp.run_time = to_fv - self.current_fv
            self.current_fv = to_fv
            return anim_grp
        else:
            return AnimationGroup()

    def animate_reverse_barcode(self, to_fv):
        if self.current_fv >= to_fv >= self.min_fv:
            anims, fvs = [], []
            for i, bar in enumerate(self.bars):
                if self.current_fv >= bar[0] >= to_fv:  # complete uncreation
                    line = self.animated_lines[bar]
                    anims.append(Uncreate(line, run_time=min(bar[1], self.current_fv)-bar[0], rate_func=lambda t: 1-t))
                    self.axes.remove(line)
                    fvs.append(min(bar[1], self.current_fv))
                    del self.animated_lines[bar]
                elif bar[0] <= to_fv <= bar[1] <= self.current_fv:  # partial uncreation
                    line1 = self.animated_lines[bar]
                    line2 = Line(self.coords_to_point(bar[0], self.delta_y * (i + 1)),
                                self.coords_to_point(to_fv, self.delta_y * (i + 1)),
                                color=self.CONFIG['bar_color'])
                    anims.append(Transform(line1, line2, run_time=bar[1]-to_fv, rate_func=linear))
                    fvs.append(bar[1])

            fvs = -np.array(fvs)
            fvs -= np.min(fvs)

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
