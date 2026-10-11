"""Compact restricted master of the path formulation (HiGHS, incremental column addition).

Exact projection of the flight-copy model of ``arp.model`` onto route variables: the copy-selection variables y_c and
their linking rows are substituted out (y_c = sum_p a_pc lam_p), so the LP value is identical to the earlier
y-based master and to the flow-model LP (valid lower bound for the original model).

Columns   lam_p (aircraft route), L_f (flight cancelled), z_i (multi-leg itinerary i broken), u (end-position shortfall)
Static rows (small)
  cover[f]  L_f + sum_p n_pf lam_p = 1             (coverage or cancellation)
  cap[...]  sum_p (copies of p in the hour) lam_p <= remaining departure / arrival quota   (only rows that can bind)
  pos[...]  u + sum_p [end(p) matches] lam_p >= n    (end positions, three levels)
  conv[k]   sum_{p in class k} lam_p = n_k           (one convexity row per aircraft class)
Lazy rows (row generation, exact at convergence; added only when violated by the current solution)
  z_i >= L_f                                         for every open leg f of a multi-leg itinerary i
  z_i >= 1 - ...                                     connection rows of multi-leg itineraries (model.py thresholds)
  v_pc + z_i - y_c >= 0  or  w_i + D z_i - sum dc y_c >= 0   delay of multi-leg itinerary i only if intact (R-1)
Separable passenger costs (single-leg itineraries: delay cost, cancellation cost; seat spill) are arc costs / L costs.
Pricing duals per copy:  pi_c = - pi_cover[f] - pi_cap[d] - pi_cap[a] + sum_{active lazy rows containing c} pi_row
(= the maximal dual-feasible link dual, so arc reduced cost = kind cost + delay cost + pi_c).
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import highspy
import numpy as np

from ..costs import ItinInfo
from ..model import MIN_CONNECT, Params, executed_departures, fixed_broken, quota_rows
from .network import AircraftClass, Network

INF = np.inf


class Master:
    def __init__(self, net: Network, itins: List[ItinInfo], p: Params, verbose: bool = False):
        self.net, self.p, self.itins = net, p, itins
        I, S = net.I, net.S
        alpha, beta, gamma = I.weights
        nc = net.ncopies
        fset = set(net.flights)
        self.cost: List[float] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        rows: List[Tuple[float, float]] = []          # static row bounds (rows are filled by columns)

        def var(c, lo=0.0, hi=INF):
            self.cost.append(c)
            self.lb.append(lo)
            self.ub.append(hi)
            return len(self.cost) - 1

        # ---- separable copy costs: passenger delay cost of single-leg and last-leg arrival, per copy ----------
        ycost = np.zeros(nc)
        self.L: Dict[int, int] = {f: var(0.0, 1.0 if f in net.forced_cancel else 0.0, 1.0) for f in net.flights}
        n_single = n_multi = 0
        pax_single = pax_multi = 0
        multi = []
        for inf in itins:
            legs = [f for f, _ in inf.it.legs]
            open_legs = [f for f in legs if f in fset]
            if not open_legs or fixed_broken(I, S, legs):         # already broken: constant (R-2, as arp.model)
                continue
            started = legs[0] in S.executed
            ccost = inf.it.count * (inf.cancel_cost_in if (inf.it.kind == "R" or started) else inf.cancel_cost_out)
            last = legs[-1]
            if last in fset and len(legs) == 1:      # single-leg delay is separable (audit Stage 2 R-1)
                for c in net.copies_of_flight.get(last, []):
                    dc = inf.delay_cost(net.carr[c] - inf.sched_arr)
                    if dc:
                        ycost[c] += inf.it.count * dc
            if len(legs) == 1:
                self.cost[self.L[last]] += ccost
                n_single += 1
                pax_single += inf.it.count
            else:
                n_multi += 1
                pax_multi += inf.it.count
                multi.append((inf, legs, open_legs, ccost))
        self.stats = dict(itins_single=n_single, itins_multi=n_multi, pax_single=pax_single, pax_multi=pax_multi)
        self.ycost = ycost
        net.set_copy_cost(ycost)
        # ---- cover rows --------------------------------------------------------------------------------------
        self.cover_row: Dict[int, int] = {}
        self.static_terms: List[List[Tuple[int, float]]] = []

        def row(lo, hi, terms=()):
            rows.append((lo, hi))
            self.static_terms.append(list(terms))
            return len(rows) - 1

        for f in net.flights:
            self.cover_row[f] = row(1.0, 1.0, [(self.L[f], 1.0)])
        self._cover_of_copy = np.array([self.cover_row[net.cflt[c]] for c in range(nc)], dtype=np.int64)
        # ---- quotas (same selection rule as model.py) -----------------------------------------------------------
        use = defaultdict(list)
        for c in range(nc):
            fo = I.flights[net.cflt[c]]
            use[(fo.orig, net.cdep[c] // 60 * 60, "d")].append(c)
            use[(fo.dest, net.carr[c] // 60 * 60, "a")].append(c)
        self.cap_row: Dict[tuple, int] = {}
        self.cap_val: Dict[tuple, int] = {}
        self.copy_cap = np.full((nc, 2), -1, dtype=np.int64)
        exec_deps = executed_departures(I, S)
        for key, cps in use.items():
            ap, h, kd = key
            # D5 (R-6) quota rows; the path master supports only rows whose left-hand side is the whole cell (true
            # whenever every copy departs after the announcement, e.g. static information)
            cap = None
            for tau, q in quota_rows(I, S, ap, h, kd, exec_deps):
                if tau is not None and any(net.cdep[c] < tau for c in cps):
                    raise NotImplementedError("path master: quota row on a subset of copies (perfect-information model)")
                cap = q if cap is None else min(cap, q)
            if cap < len({net.cflt[c] for c in cps}):
                self.cap_row[key] = row(-INF, cap)
                self.cap_val[key] = cap
                for c in cps:
                    self.copy_cap[c, 0 if kd == "d" else 1] = self.cap_row[key]
        # ---- end positions -------------------------------------------------------------------------------------
        cc = I.pos_pen
        c_fam, c_mod, c_cfg = gamma * (cc[0] - cc[1]), gamma * (cc[1] - cc[2]), gamma * cc[2]
        self.fam_of_model = {a.model: a.family for a in I.aircraft.values()}
        self.cfg_rows, self.mod_rows, self.fam_rows = defaultdict(list), defaultdict(list), defaultdict(list)
        self.pos_rows: List[int] = []
        self.pos_u_cost: Dict[int, float] = {}
        for ap, reqs in I.positions.items():
            by_model, by_fam = defaultdict(int), defaultdict(int)

            def add_pos(store, key, cnt, w):
                u = var(w)
                r = row(cnt, INF, [(u, 1.0)])
                store[key].append(r)
                self.pos_rows.append(r)
                self.pos_u_cost[r] = w

            for model, cfg, cnt in reqs:
                add_pos(self.cfg_rows, (ap, (model, cfg)), cnt, c_cfg)
                by_model[model] += cnt
                by_fam[self.fam_of_model.get(model, model)] += cnt
            for model, cnt in by_model.items():
                add_pos(self.mod_rows, (ap, model), cnt, c_mod)
            for fam, cnt in by_fam.items():
                add_pos(self.fam_rows, (ap, fam), cnt, c_fam)
        self._pos_cache: Dict[Tuple[str, tuple], List[int]] = {}
        # ---- convexity ---------------------------------------------------------------------------------------------
        self.conv_row = [row(float(cl.n), float(cl.n)) for cl in net.classes]
        # ---- lazy passenger rows (multi-leg itineraries) -------------------------------------------------------------
        z_of = {}
        self.lazy_z: List[int] = []
        self.lazy_V: List[int] = []                 # delay-link column v (+1) of a row, or -1
        self.lazy_L: List[int] = []
        self.lazy_lo: List[float] = []
        self.lazy_cps: List[List[int]] = []
        self.lazy_zc: List[float] = []              # coefficient of z in the row (1, or D_max for the aggregate link)
        self.lazy_wd: Dict[int, Dict[int, float]] = {}   # row -> {copy: coefficient magnitude} if not 1 (aggregate link)
        arc_by_f = {f: [(c, net.cdep[c], net.carr[c]) for c in net.copies_of_flight.get(f, [])] for f in net.flights}
        for inf, legs, open_legs, ccost in multi:
            z = var(ccost, 0.0, 1.0)
            for f in open_legs:
                self.lazy_z.append(z); self.lazy_L.append(self.L[f]); self.lazy_lo.append(0.0); self.lazy_cps.append([])
                self.lazy_V.append(-1); self.lazy_zc.append(1.0)
            # delay of a multi-leg itinerary only if intact, same linking as arp.model (R-1):
            #   'copy':      v_pc + z_p - y_c >= 0                    (cost dc on v_pc)
            #   'aggregate': w_p + D_max z_p - sum_c dc y_c >= 0       (cost 1 on w_p)
            last = legs[-1]
            if last in fset:
                dcs = [(c, inf.it.count * inf.delay_cost(net.carr[c] - inf.sched_arr)) for c in net.copies_of_flight.get(last, [])]
                dcs = [(c, dc) for c, dc in dcs if dc]
                for c, dc in dcs:
                    if dc > ccost + 1e-6:
                        raise ValueError(f"itinerary {inf.it.id}: delay cost {dc:.2f} exceeds cancellation cost {ccost:.2f}")
                if dcs and p.delay_link == "aggregate":
                    w = var(1.0, 0.0, INF)
                    self.lazy_wd[len(self.lazy_lo)] = dict(dcs)
                    self.lazy_z.append(z); self.lazy_zc.append(max(dc for _, dc in dcs)); self.lazy_L.append(-1)
                    self.lazy_lo.append(0.0); self.lazy_cps.append([c for c, _ in dcs]); self.lazy_V.append(w)
                elif dcs:
                    for c, dc in dcs:
                        v = var(dc, 0.0, 1.0)
                        self.lazy_z.append(z); self.lazy_zc.append(1.0); self.lazy_L.append(-1); self.lazy_lo.append(0.0)
                        self.lazy_cps.append([c]); self.lazy_V.append(v)
            for l1, l2 in zip(legs, legs[1:]):
                f1, f2 = I.flights[l1], I.flights[l2]
                mct = min(MIN_CONNECT, f2.dep - f1.arr)
                if l2 not in fset:
                    continue
                c2 = arc_by_f.get(l2, [])
                if l1 in S.executed:
                    a1 = S.executed[l1][1]
                    bad = [c for c, dep, arr in c2 if dep < a1 + mct]
                    if bad:
                        self.lazy_z.append(z); self.lazy_L.append(-1); self.lazy_lo.append(0.0); self.lazy_cps.append(bad)
                        self.lazy_V.append(-1); self.lazy_zc.append(1.0)
                    continue
                if l1 not in fset:
                    continue
                c1 = arc_by_f.get(l1, [])
                min_dep2 = min((dep for _, dep, _ in c2), default=None)
                for a in sorted({arr for _, _, arr in c1}):
                    if min_dep2 is None or min_dep2 >= a + mct:
                        continue
                    late1 = [c for c, _, arr in c1 if arr >= a]
                    early2 = [c for c, dep, _ in c2 if dep < a + mct]
                    if late1 and early2:
                        self.lazy_z.append(z); self.lazy_L.append(-1); self.lazy_lo.append(-1.0)
                        self.lazy_cps.append(late1 + early2)
                        self.lazy_V.append(-1); self.lazy_zc.append(1.0)
        nlz = len(self.lazy_lo)
        self.lazy_z = np.array(self.lazy_z, dtype=np.int64)
        self.lazy_L = np.array(self.lazy_L, dtype=np.int64)
        self.lazy_V = np.array(self.lazy_V, dtype=np.int64)
        self.lazy_zc = np.array(self.lazy_zc)
        self.lazy_lo = np.array(self.lazy_lo)
        self.lazy_r = np.array([r for r in range(nlz) for _ in self.lazy_cps[r]], dtype=np.int64)
        self.lazy_c = np.array([c for r in range(nlz) for c in self.lazy_cps[r]], dtype=np.int64)
        self.lazy_w = np.array([self.lazy_wd.get(r, {}).get(c, 1.0) for r in range(nlz) for c in self.lazy_cps[r]])
        self.lazy_added = np.zeros(nlz, dtype=bool)
        self.lazy_row_idx = np.full(nlz, -1, dtype=np.int64)
        self.n_lazy_rows = 0
        self.rows_of_copy: Dict[int, Dict[int, int]] = defaultdict(dict)    # copy -> {lazy r: hrow} (active lazy rows)
        self.cols_of_copy: Dict[int, Dict[int, int]] = defaultdict(dict)    # copy -> {pid: multiplicity}
        # ---- pass to HiGHS -------------------------------------------------------------------------------------------
        self.n_static = len(self.cost)
        self.n_rows = len(rows)
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(verbose))
        h.setOptionValue("threads", int(p.threads))       # set before the first run (HiGHS scheduler is process-wide)
        h.setOptionValue("presolve", "off")
        h.setOptionValue("solver", "simplex")
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.n_static, self.n_rows
        lp.col_cost_ = np.array(self.cost, dtype=float)
        lp.col_lower_ = np.array(self.lb, dtype=float)
        lp.col_upper_ = np.array(self.ub, dtype=float)
        lp.row_lower_ = np.array([r[0] for r in rows], dtype=float)
        lp.row_upper_ = np.array([r[1] for r in rows], dtype=float)
        cols = defaultdict(list)
        for r, terms in enumerate(self.static_terms):
            for c, v in terms:
                cols[c].append((r, v))
        start, index, value = [0], [], []
        for c in range(self.n_static):
            for r, v in cols.get(c, ()):
                index.append(r)
                value.append(v)
            start.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(start, dtype=np.int32)
        lp.a_matrix_.index_ = np.array(index, dtype=np.int32)
        lp.a_matrix_.value_ = np.array(value, dtype=float)
        h.passModel(lp)
        self.h = h
        self.lam: List[dict] = []
        self.active: List[int] = []
        self.pool_keys: Dict[tuple, int] = {}
        self.pos_of: Dict[int, int] = {}
        self.lp_time = 0.0
        self.last_x = None
        self.last_rd_full = None

    # ------------------------------------------------------------------------------------------------------------------
    def pos_rows_for(self, ap: str, kind) -> List[int]:
        key = (ap, kind)
        r = self._pos_cache.get(key)
        if r is None:
            r = (list(self.cfg_rows.get((ap, kind), [])) + list(self.mod_rows.get((ap, kind[0]), [])) +
                 list(self.fam_rows.get((ap, self.fam_of_model.get(kind[0], kind[0])), [])))
            self._pos_cache[key] = r
        return r

    def add_column(self, cls: AircraftClass, route: List[int], tag: str = "", keep: bool = False, it: int = 0) -> bool:
        """Add the route as a column of class `cls`.  Returns False if it is already in the LP."""
        net = self.net
        key = (cls.cid, tuple(route))
        pid = self.pool_keys.get(key)
        if pid is not None and self.lam[pid]["active"]:
            if keep:
                self.lam[pid]["keep"] = True
            return False
        if pid is None:
            cost, end, pen = net.route_info(cls, route, with_pen=True)
            kind = net.comm_of[cls.kid].kind
            ent: Dict[int, float] = defaultdict(float)
            for c in route:
                ent[self.cover_row[net.cflt[c]]] += 1.0
                for r in self.copy_cap[c]:
                    if r >= 0:
                        ent[int(r)] += 1.0
            ent[self.conv_row[cls.cid]] += 1.0
            if end is not None:
                for r in self.pos_rows_for(end, kind):
                    ent[r] += 1.0
            rec = dict(cid=cls.cid, route=list(route), cost=cost, end=end, pen=pen, tag=tag, keep=keep, active=False,
                       it=it, ent=dict(ent))
            pid = len(self.lam)
            self.lam.append(rec)
            self.pool_keys[key] = pid
        rec = self.lam[pid]
        ent = dict(rec["ent"])
        for c in rec["route"]:                       # entries in the active lazy rows containing the copies
            for r, hrow in self.rows_of_copy.get(c, {}).items():
                ent[hrow] = ent.get(hrow, 0.0) - self.lazy_wd.get(r, {}).get(c, 1.0)
        rows_ = sorted(ent)
        self.h.addCol(rec["cost"], 0.0, INF, len(rows_), np.array(rows_, dtype=np.int32),
                      np.array([ent[r] for r in rows_], dtype=float))
        rec["active"] = True
        rec["it"] = it
        rec["keep"] = rec["keep"] or keep
        self.pos_of[pid] = self.n_static + len(self.active)
        self.active.append(pid)
        for c in rec["route"]:
            self.cols_of_copy[c][pid] = self.cols_of_copy[c].get(pid, 0) + 1
        return True

    def purge(self, thr: float, it: int, min_age: int = 5) -> int:
        """Delete non-basic, zero-valued columns with reduced cost > thr (never the `keep` columns)."""
        sol = self.h.getSolution()
        x, cd = sol.col_value, sol.col_dual
        drop = []
        for pos, pid in enumerate(self.active):
            rec = self.lam[pid]
            c = self.n_static + pos
            if not rec["keep"] and x[c] < 1e-9 and cd[c] > thr and it - rec["it"] >= min_age:
                drop.append(pos)
        if not drop:
            return 0
        self.h.deleteCols(len(drop), np.array([self.n_static + p for p in drop], dtype=np.int32))
        gone = set(drop)
        for p in drop:
            pid = self.active[p]
            self.lam[pid]["active"] = False
            for c in self.lam[pid]["route"]:
                d = self.cols_of_copy[c]
                d.pop(pid, None)
        self.active = [pid for pos, pid in enumerate(self.active) if pos not in gone]
        self.pos_of = {pid: self.n_static + i for i, pid in enumerate(self.active)}
        if self.last_x is not None and len(self.last_x) == self.n_static + len(self.active) + len(drop):
            # keep the stored primal vector aligned with the remaining columns (the deleted ones are 0): separation
            # and the convergence test read last_x by position (audit Stage 2, F-17)
            self.last_x = np.delete(self.last_x, [self.n_static + p for p in drop])
        return len(drop)

    # ------------------------------------------------------------------------------------------------------------------
    def yvalues(self, x: np.ndarray) -> np.ndarray:
        """Copy-selection values y_c = sum_p n_pc lam_p from the primal vector."""
        y = np.zeros(self.net.ncopies)
        lens, cps, vals = [], [], []
        for i, pid in enumerate(self.active):
            v = x[self.n_static + i]
            if v > 1e-12:
                r = self.lam[pid]["route"]
                cps.extend(r)
                vals.extend([v] * len(r))
        if cps:
            np.add.at(y, np.array(cps, dtype=np.int64), np.array(vals))
        return y

    def _row_entries(self, r: int) -> Dict[int, float]:
        ent: Dict[int, float] = {}
        if self.lazy_z[r] >= 0:
            ent[int(self.lazy_z[r])] = float(self.lazy_zc[r])
        if self.lazy_L[r] >= 0:
            ent[int(self.lazy_L[r])] = -1.0
        if self.lazy_V[r] >= 0:
            ent[int(self.lazy_V[r])] = 1.0
        wd = self.lazy_wd.get(r, {})
        for c in self.lazy_cps[r]:
            w = wd.get(c, 1.0)
            for pid, m in self.cols_of_copy.get(c, {}).items():
                col = self.pos_of[pid]
                ent[col] = ent.get(col, 0.0) - m * w
        return ent

    def _add_lazy(self, r: int) -> None:
        ent = self._row_entries(r)
        idx = np.array(sorted(ent), dtype=np.int32)
        val = np.array([ent[i] for i in idx], dtype=float)
        hrow = self.h.getNumRow()
        self.h.addRow(float(self.lazy_lo[r]), INF, len(idx), idx, val)
        self.lazy_row_idx[r] = hrow
        self.lazy_added[r] = True
        for c in self.lazy_cps[r]:
            self.rows_of_copy[c][r] = hrow
        self.n_lazy_rows += 1
        # a row that contains a copy twice (late1 and early2 overlap) is counted via multiplicities in _row_entries

    def separate(self, x: np.ndarray, tol: float = 1e-6, limit: Optional[int] = None) -> int:
        """Add the lazy rows violated by x.  Returns the number of rows added."""
        if not len(self.lazy_lo):
            return 0
        y = self.yvalues(x)
        act = x[self.lazy_z] * self.lazy_zc
        hasL = self.lazy_L >= 0
        act[hasL] -= x[self.lazy_L[hasL]]
        hasV = self.lazy_V >= 0
        act[hasV] += x[self.lazy_V[hasV]]
        if len(self.lazy_r):
            act -= np.bincount(self.lazy_r, weights=y[self.lazy_c] * self.lazy_w, minlength=len(self.lazy_lo))
        viol = np.nonzero((act < self.lazy_lo - tol) & ~self.lazy_added)[0]
        if limit is not None and len(viol) > limit:
            viol = viol[np.argsort(act[viol] - self.lazy_lo[viol])[:limit]]
        for r in viol:
            self._add_lazy(int(r))
        return len(viol)

    def add_all_lazy(self) -> int:
        todo = np.nonzero(~self.lazy_added)[0]
        for r in todo:
            self._add_lazy(int(r))
        return len(todo)

    # ------------------------------------------------------------------------------------------------------------------
    def solve_once(self):
        """One LP solve, no separation.  Returns (objective, pricing-dual vector); stores last_x / last_rd_full."""
        t0 = time.time()
        self.h.run()
        self.lp_time += time.time() - t0
        st = self.h.modelStatusToString(self.h.getModelStatus())
        if st != "Optimal":
            raise RuntimeError(f"RMP LP status {st}")
        sol = self.h.getSolution()
        self.last_x = np.array(sol.col_value)
        self.last_rd_full = np.array(sol.row_dual)
        return self.h.getInfo().objective_function_value, self.pricing_vector(self.last_rd_full)

    def pricing_vector(self, rd_full: np.ndarray) -> np.ndarray:
        """[static row duals | per-copy duals]; the copy dual is the implied (maximal feasible) link dual."""
        net = self.net
        nc = net.ncopies
        pc = -rd_full[self._cover_of_copy]
        for k in (0, 1):
            r = self.copy_cap[:, k]
            ok = r >= 0
            pc[ok] -= rd_full[r[ok]]
        if self.n_lazy_rows:
            rows = self.lazy_row_idx[self.lazy_r]
            sel = rows >= 0
            pc += np.bincount(self.lazy_c[sel], weights=rd_full[rows[sel]] * self.lazy_w[sel], minlength=nc)
        return np.concatenate([rd_full[:self.n_rows], pc])

    def set_rmp_lp(self, mode: str) -> None:
        """RMP LP algorithm: 'simplex' (default, warm-started vertex duals) or 'ipm' (interior point without crossover:
        central, non-vertex duals -- the degeneracy test of the CG stall)."""
        if mode == "ipm":
            self.h.setOptionValue("solver", "ipm")
            self.h.setOptionValue("run_crossover", "off")
        else:
            self.h.setOptionValue("solver", "simplex")
            self.h.setOptionValue("run_crossover", "on")

    def tighten(self, rd_full: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.pricing_vector(rd_full)         # the implied copy duals are already the tight ones

    def solve_lp(self):
        """Solve the RMP LP, separating lazy rows until none is violated.  Returns (objective, pricing duals)."""
        while True:
            obj, pv = self.solve_once()
            if self.separate(self.last_x) == 0:
                return obj, pv

    def natural_duals(self) -> np.ndarray:
        """A sane starting dual vector: every flight is worth its passengers' cancellation cost; a missing aircraft
        costs its position penalty."""
        net = self.net
        v = defaultdict(float)
        for inf in self.itins:
            for f, _ in inf.it.legs:
                v[f] += inf.it.count * inf.cancel_cost_out
        pv = np.zeros(self.n_rows + net.ncopies)
        pv[self.n_rows:] = -np.array([v.get(net.cflt[c], 0.0) for c in range(net.ncopies)])
        for r, w in self.pos_u_cost.items():
            pv[r] = w
        return pv

    def duals_from_flow(self, link, pos) -> np.ndarray:
        """Pricing vector from the flow-LP duals (link rows by copy id, position rows in master order)."""
        assert len(link) == self.net.ncopies and len(pos) == len(self.pos_rows), (len(link), len(pos))
        pv = np.zeros(self.n_rows + self.net.ncopies)
        pv[self.n_rows:] = np.array(link) - self.ycost
        pv[np.array(self.pos_rows)] = np.array(pos)
        return pv

    def duals(self, pv: np.ndarray):
        return pv[self.n_rows:], pv[self.conv_row]

    def pos_credit(self, pv: np.ndarray, ap: str, kind) -> float:
        return -float(sum(pv[r] for r in self.pos_rows_for(ap, kind)))

    def lam_values(self) -> np.ndarray:
        return np.array(self.h.getSolution().col_value)[self.n_static:]

    def active_columns(self) -> List[dict]:
        return [self.lam[pid] for pid in self.active]

    # ------------------------------------------------------------------------------------------------------------------
    def fixed_lp(self, mult: Dict[int, float]) -> Tuple[float, np.ndarray]:
        """LP with every lam fixed to mult.get(pool id, 0) (lazy rows separated); returns objective and solution."""
        h = self.h
        k = len(self.active)
        idx = np.arange(self.n_static, self.n_static + k, dtype=np.int32)
        lo = np.array([mult.get(pid, 0.0) for pid in self.active], dtype=float)
        h.changeColsBounds(k, idx, lo, lo)
        try:
            obj, _ = self.solve_lp()
            x = np.array(h.getSolution().col_value)
        finally:
            h.changeColsBounds(k, idx, np.zeros(k), np.full(k, INF))
        return obj, x

    def forbid_penalised(self) -> int:
        clean = defaultdict(int)
        for pid in self.active:
            if self.lam[pid]["pen"] <= 0:
                clean[self.lam[pid]["cid"]] += 1
        pos = [i for i, pid in enumerate(self.active) if self.lam[pid]["pen"] > 0 and clean[self.lam[pid]["cid"]] > 0]
        if pos:
            idx = np.array([self.n_static + i for i in pos], dtype=np.int32)
            self.h.changeColsBounds(len(pos), idx, np.zeros(len(pos)), np.zeros(len(pos)))
        return len(pos)

    def unforbid(self) -> None:
        k = len(self.active)
        if k:
            self.h.changeColsBounds(k, np.arange(self.n_static, self.n_static + k, dtype=np.int32), np.zeros(k),
                                    np.full(k, INF))

    def solve_mip(self, time_limit: float, gap: float, start: Optional[np.ndarray] = None, threads: int = 1,
                  verbose: bool = False, rounds: int = 8):
        """MIP over the pool (lam integer).  Lazy itinerary rows are separated against the MIP solution and the MIP is
        re-solved until no row is violated, so the final objective is exact for the original model."""
        h = self.h
        k = len(self.active)
        idx = np.arange(self.n_static, self.n_static + k, dtype=np.int32)
        h.changeColsIntegrality(k, idx, np.array([highspy.HighsVarType.kInteger] * k))
        self.set_rmp_lp("simplex")
        h.setOptionValue("presolve", "on")
        h.setOptionValue("output_flag", bool(verbose))
        h.setOptionValue("mip_rel_gap", float(gap))
        t_end = time.time() + time_limit          # hard wall-clock deadline for the whole call (all separation rounds)

        def stop_at_deadline(e):                  # HiGHS' time_limit is not honoured in every phase (presolve, large root LP,
            if time.time() > t_end:               # heuristics); the interrupt callbacks are polled everywhere
                e.interrupt()

        h.cbMipInterrupt += stop_at_deadline
        h.cbSimplexInterrupt += stop_at_deadline
        h.cbIpmInterrupt += stop_at_deadline
        total = 0.0
        x = None
        status = "not run"
        info = None
        try:
            for rd_ in range(rounds):
                left = t_end - time.time()
                if rd_ > 0 and left <= 0:
                    status += " (deadline reached during separation)"
                    break
                h.setOptionValue("time_limit", h.getRunTime() + max(1.0, left))     # cumulative clock
                if start is not None:
                    hs = highspy.HighsSolution()
                    hs.col_value = list(map(float, start))
                    hs.value_valid = True
                    h.setSolution(hs)
                t0 = time.time()
                h.run()
                total += time.time() - t0
                info = h.getInfo()
                status = h.modelStatusToString(h.getModelStatus())
                x = np.array(h.getSolution().col_value) if info.primal_solution_status > 0 else None
                if x is None:
                    return None, info.objective_function_value, info.mip_dual_bound, status, total
                if time.time() > t_end:
                    status += " (deadline)"
                if self.separate(x) == 0:
                    return x, info.objective_function_value, info.mip_dual_bound, status, total
                start = None
            else:
                status += " (separation rounds exhausted)"
        finally:
            h.cbMipInterrupt -= stop_at_deadline
            h.cbSimplexInterrupt -= stop_at_deadline
            h.cbIpmInterrupt -= stop_at_deadline
        return x, info.objective_function_value, info.mip_dual_bound, status, total
