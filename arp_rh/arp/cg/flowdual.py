"""Duals of the flow-model LP (arp.model) mapped onto the path-master rows.

The flow model contains the same copy-linking rows (y_{f,dep} = sum_k x_{f,dep,k}) and end-position rows as the
path master, so their LP duals are valid pricing duals for the path formulation.  The model is solved *unchanged*:
``_Builder.solve`` is wrapped from outside to keep the row duals of the LP call.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, Optional

import highspy
import numpy as np

from .. import model as M
from ..costs import itinerary_info
from ..instance import Instance
from ..model import Params, initial_state


def flow_lp_duals(I: Instance, p: Params, warm: dict, time_limit: float = 3600.0, verbose: bool = False):
    """Solve the flow-model LP of the full-horizon window; return dict(obj, link (list over y in creation order),
    pos (list over position rows in model order), time)."""
    cap = {}
    orig_solve = M._Builder.solve

    def solve(self, time_limit, gap, threads, verbose, heur=0.05, integrality=True, fixed_ub=None, lp_method="choose"):
        if integrality:
            return orig_solve(self, time_limit, gap, threads, verbose, heur, integrality, fixed_ub, lp_method)
        h = highspy.Highs()
        h.setOptionValue("solver", lp_method)
        h.setOptionValue("output_flag", bool(verbose))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("threads", int(threads))
        lp = highspy.HighsLp()
        n = len(self.cost)
        lp.num_col_, lp.num_row_ = n, len(self.rows)
        lp.col_cost_ = np.array(self.cost, dtype=float)
        lp.col_lower_ = np.array(self.lb, dtype=float)
        lp.col_upper_ = np.array(self.ub, dtype=float)
        lp.row_lower_ = np.array(self.rlb, dtype=float)
        lp.row_upper_ = np.array(self.rub, dtype=float)
        cols = {}
        for r, terms in enumerate(self.rows):
            for c, v in terms:
                cols.setdefault(c, []).append((r, v))
        start, index, value = [0], [], []
        for c in range(n):
            for r, v in cols.get(c, ()):
                index.append(r)
                value.append(v)
            start.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(start, dtype=np.int32)
        lp.a_matrix_.index_ = np.array(index, dtype=np.int32)
        lp.a_matrix_.value_ = np.array(value, dtype=float)
        h.passModel(lp)
        t0 = time.time()
        h.run()
        st = time.time() - t0
        info = h.getInfo()
        sol = np.array(h.getSolution().col_value) if info.primal_solution_status > 0 else None
        rd = np.array(h.getSolution().col_dual) if sol is not None else None
        cap.update(rows=self.rows, rlb=list(self.rlb), rub=list(self.rub), row_dual=np.array(h.getSolution().row_dual),
                   ydef=list(self.ydef.keys()), cost=list(self.cost), obj=info.objective_function_value, time=st,
                   pen=set(self.pen))
        return sol, info.objective_function_value, rd, h.modelStatusToString(h.getModelStatus()), st

    M._Builder.solve = solve
    try:
        S = initial_state(I, known_all=True)
        itins = itinerary_info(I)
        q = replace(p, time_limit=time_limit, lp_restrict=True, rc_keep=50.0, mip_gap=1.0)
        M.solve_window(I, S, I.modifiable, itins, q, final=True, warm=warm)
    finally:
        M._Builder.solve = orig_solve
    rdl = cap["row_dual"]
    rows, rlb, rub, cost = cap["rows"], cap["rlb"], cap["rub"], cap["cost"]
    ydefs = set(cap["ydef"])
    link = []
    row_of_y = {}
    for r, terms in enumerate(rows):
        if rlb[r] == 0.0 and rub[r] == 0.0 and len(terms) > 1 and terms[0][0] in ydefs and terms[0][1] == 1.0 \
                and all(v == -1.0 for _, v in terms[1:]):
            row_of_y[terms[0][0]] = r
    link = [float(rdl[row_of_y[y]]) for y in cap["ydef"]]
    pos = []
    for r, terms in enumerate(rows):
        if (rub[r] == np.inf and rlb[r] >= 1.0 and terms and terms[0][1] == 1.0 and terms[0][0] in cap["pen"]
                and cost[terms[0][0]] < 1e5 - 1 and all(v > 0 for _, v in terms)):
            pos.append(float(rdl[r]))
    return dict(obj=cap["obj"], link=link, pos=pos, time=cap["time"])
