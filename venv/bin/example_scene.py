#!/usr/bin/env python3

from simplicial_complex import *
from filtrations import *
from persistent_homology import *

import numpy as np
import gudhi as g
from manimlib.imports import *


class ExampleScene(Scene):

    def construct(self):
        sc = [[0], [1], [2], [0, 1],
              [0, 2], [1, 2]]
        filtration = [0, 0, 0.5, 1, 1, 1]
        points = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        st = g.SimplexTree()
        for s, v in zip(sc, filtration):
            st.insert(s, v)

        simp_comp = ECT(st, points, np.array([1, 2, 0]))
        # self.play(ShowCreation(simp_comp))
        self.add(simp_comp)
        self.wait(2)
        self.play(simp_comp.animate_filtration())
        self.wait(2)
        self.play(simp_comp.animate_reverse_filtration(to_fv=0.2))
        self.wait(5)
