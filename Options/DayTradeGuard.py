"""Conservative day-trade counter/guard (MODELING ASSUMPTION, not broker logic).

Assumption: max `max_dt` day trades across the whole portfolio in any rolling `window` sessions.
A day trade = a closing event on a leg (contract id) that matches quantity OPENED THE SAME SESSION.
Counting is worst-case: each leg's same-day close counts 1 (a 4-leg condor opened and closed same day = 4),
partial closes of a same-day lot each count, and same-day lots are matched first (LIFO-for-today).
Package counts (unique package ids per same-day close) are tracked for comparison.
"""
from __future__ import annotations
from collections import defaultdict


class DayTradeGuard:
    def __init__(self, sessions, max_dt=3, window=5):
        self.sessions = list(sessions)
        self.idx = {d: i for i, d in enumerate(self.sessions)}
        self.max_dt, self.window = max_dt, window
        self.lots = defaultdict(list)      # leg_id -> [[open_date, qty_remaining], ...]
        self.events = []                   # audit log
        self.day_trades = []               # (date, leg_id, package_id, qty)

    def RecordOpen(self, date, leg_id, package_id, qty):
        self.lots[leg_id].append([date, float(qty)])
        self.events.append({"date": date, "action": "open", "leg_id": leg_id, "package_id": package_id, "qty": qty})

    def _same_day_qty(self, date, leg_id):
        return sum(q for d, q in self.lots[leg_id] if d == date and q > 0)

    def CountInWindow(self, date, by="leg"):
        i = self.idx[date]
        recent = [x for x in self.day_trades if i - self.window < self.idx[x[0]] <= i]
        if by == "package":
            return len({(x[0], x[2]) for x in recent})
        return len(recent)

    def NewDayTrades(self, date, closes):
        """closes: list of (leg_id, qty). Returns number of leg day-trades this close would add (worst case)."""
        return sum(1 for leg_id, q in closes if q > 0 and self._same_day_qty(date, leg_id) > 0)

    def CanClose(self, date, closes):
        return self.CountInWindow(date) + self.NewDayTrades(date, closes) <= self.max_dt

    def RecordClose(self, date, leg_id, package_id, qty):
        remaining = float(qty)
        lots = self.lots[leg_id]
        # same-day lots first (worst case)
        order = sorted(range(len(lots)), key=lambda k: (lots[k][0] != date, lots[k][0]))
        dt_qty = 0.0
        for k in order:
            if remaining <= 0:
                break
            take = min(lots[k][1], remaining)
            if take <= 0:
                continue
            lots[k][1] -= take; remaining -= take
            if lots[k][0] == date:
                dt_qty += take
        if remaining > 1e-9:
            raise ValueError(f"closing more than open on {leg_id}")
        self.lots[leg_id] = [l for l in lots if l[1] > 1e-12]
        if dt_qty > 0:
            self.day_trades.append((date, leg_id, package_id, dt_qty))
        self.events.append({"date": date, "action": "close", "leg_id": leg_id, "package_id": package_id, "qty": qty, "day_trade_qty": dt_qty})
        return dt_qty > 0
