"""
1HZ25V EVEN/ODD BOT — DUAL-ORDER MARKOV EDITION  v3
=====================================================
Trades DIGITEVEN / DIGITODD on Deriv 1HZ25V (Volatility 25 — 1s ticks).

═══════════════════════════════════════════════════════
STRATEGY: DUAL-ORDER MARKOV CONSENSUS
═══════════════════════════════════════════════════════

  ORDER-1  (10×10 matrix, 100 cells)
  ─────────────────────────────────
  Conditions on the last digit only.
    P(digit_t | digit_{t-1})
  Ready after min_sample_1 ticks (~400).
  Always available once warmed up.

  ORDER-2  (10×10×10 tensor, 1 000 cells)
  ────────────────────────────────────────
  Conditions on the last TWO digits.
    P(digit_t | digit_{t-2}, digit_{t-1})
  Captures pair-to-next patterns that order-1 averages away.
  e.g. after (3→7) even may differ from after (9→7).
  Needs ~10× more data: ready after min_sample_2 ticks (~2000).
  Falls back gracefully to order-1 while warming up.

  CONSENSUS GATE
  ──────────────
  Four possible states each evaluation:

    State                 Action
    ──────────────────────────────────────────────────────
    O2 ready, agree       Trade — highest confidence
    O2 ready, disagree    Skip — contradictory signals
    O2 not ready yet      Trade on O1 alone (fallback)
    O1 no edge            Skip

  "Agree" means both orders predict the same side (EVEN/ODD)
  with conservative probability above the breakeven threshold.
  Both EVs must also be positive.

═══════════════════════════════════════════════════════
PARAMETER RATIONALE FOR 1HZ25V
═══════════════════════════════════════════════════════

  Symbol ticks at 1/second — warmup is fast (400 s ≈ 7 min
  for order-1; 2000 s ≈ 33 min for order-2).

  Volatility 25% sits between R_10 and R_100; parameters
  are tuned to this middle ground:

    payout          0.95   (standard Even/Odd payout)
    min_sample_1    400    (order-1 warmup)
    min_sample_2    2000   (order-2 warmup — 100 pair-rows × ~20 obs each)
    edge_thresh     0.018  (+1.8% — tighter than R_10, looser than R_100)
    conf_sigma_1    1.6    (order-1 Wilson CI)
    conf_sigma_2    2.0    (order-2 Wilson CI — wider: smaller per-row n)
    max_horizon     5
    entropy_gate    3.15   (below log2(10)=3.3219; 25% vol is quite random)
    drift_thresh    15.5   (chi-sq df=9, ~p=0.08)
    trade_interval  20 ticks (~20 s at 1 Hz)
    martingale      1.35×, reset after 5 losses

═══════════════════════════════════════════════════════
V2 RESILIENCE (unchanged)
═══════════════════════════════════════════════════════
  1. FORCED UNLOCK   — "u" + Enter; auto-timeout after lock_timeout_seconds
  2. ORPHAN RECOVERY — buy-retry loop + profit_table fallback
  3. AUTO-RECONNECT  — exponential back-off; tick re-subscription;
                       in-flight contract preserved across reconnects
  4. SEND QUEUE      — serialised outgoing messages, no concurrent sends
  5. HEARTBEAT       — WebSocket ping every 30 s
  6. CONSOLE CMDS    — u (unlock)  s (stats)  q (quit)
"""

import csv
import json
import math
import os
import asyncio
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    raise SystemExit("websockets not installed.  Run: pip install websockets")


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # --- Deriv connection ---
    "api_token":    os.environ.get("DERIV_API_TOKEN", ""),
    "app_id":       1089,
    "symbol":       "R_10",             # Volatility 10 Index (1s ticks)

    # --- Payout ---
    # Even/Odd on R_10 pays ~87% (lower than 1HZ25V's 95%)
    # Break-even WR = 1/(1+0.87) = 53.5%
    "payout":       0.87,

    # --- Order-1 Markov ---
    "min_sample_1": 500,
    "conf_sigma_1": 1.5,

    # --- Order-2 Markov ---
    "min_sample_2": 2000,
    "conf_sigma_2": 1.8,

    # --- Shared signal parameters ---
    # Break-even = 0.535. Need to clear by 2.2% minimum.
    "edge_thresh":  0.022,
    "max_horizon":  5,

    # --- Entropy gate ---
    "entropy_gate": 3.10,

    # --- Non-stationarity ---
    "recent_win":   200,
    "ref_win":      500,
    "drift_thresh": 15.5,

    # --- Martingale ---
    "initial_stake":  0.35,
    "martingale_mul": 1.35,
    "max_losses":     4,
    "loss_limit":     30.0,

    # --- Trade pacing ---
    "trade_interval_ticks": 20,

    # --- Resilience ---
    "lock_timeout_seconds": 120,
    "buy_recv_retries":     8,
    "reconnect_delay_min":  2,
    "reconnect_delay_max":  60,
    "ws_ping_interval":     30,
    "orphan_poll_attempts": 4,
    "orphan_poll_interval": 3,

    # --- Logging ---
    "trades_csv":   "/tmp/r10_markov_eo_trades.csv",
    "stats_csv":    "/tmp/r10_markov_eo_stats.json",
}


# ============================================================================
# CSV TRADE LOGGER
# ============================================================================

CSV_FIELDS = [
    "timestamp", "direction", "horizon", "stake", "profit", "win",
    "consensus", "order_used", "conservative_p", "ev",
    "entropy", "chi2", "drift", "loss_streak", "balance_approx",
]


class TradeLogger:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(path).exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def record(self, direction: str, horizon: int, stake: float,
               profit: float, win: bool, consensus: str, order_used: int,
               conservative_p: float, ev: float, entropy: float,
               chi2: float, drift: bool, loss_streak: int,
               balance_approx: float):
        row = {
            "timestamp":      datetime.now().isoformat(),
            "direction":      direction,
            "horizon":        horizon,
            "stake":          round(stake, 2),
            "profit":         round(profit, 2),
            "win":            win,
            "consensus":      consensus,
            "order_used":     order_used,
            "conservative_p": round(conservative_p, 5),
            "ev":             round(ev, 5),
            "entropy":        round(entropy, 4),
            "chi2":           round(chi2, 3),
            "drift":          drift,
            "loss_streak":    loss_streak,
            "balance_approx": round(balance_approx, 2),
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ============================================================================
# HELPERS
# ============================================================================

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(tag: str, msg: str):
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)


# ============================================================================
# SHARED MARKOV MATH
# ============================================================================

def mat_mul(A, B):
    """10×10 matrix multiplication."""
    C = [[0.0] * 10 for _ in range(10)]
    for i in range(10):
        for k in range(10):
            if A[i][k] == 0:
                continue
            for j in range(10):
                C[i][j] += A[i][k] * B[k][j]
    return C

def mat_pow(P, n):
    """Matrix exponentiation by repeated squaring."""
    result = [[1.0 if i == j else 0.0 for j in range(10)] for i in range(10)]
    base   = [row[:] for row in P]
    while n > 0:
        if n % 2 == 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        n //= 2
    return result

def even_prob_from_row(row):
    """P(even) from a probability distribution over digits 0–9."""
    return sum(row[j] for j in (0, 2, 4, 6, 8))

def conservative_prob(prob, n, sigma):
    """Wilson lower-bound: pessimistic estimate of true probability."""
    if n <= 0:
        return prob
    z      = sigma
    denom  = 1 + z * z / n
    centre = (prob + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(prob * (1 - prob) / n + z * z / (4 * n * n))) / denom
    return centre - margin


# ============================================================================
# ENTROPY & CHI-SQUARE
# ============================================================================

def calc_entropy(digits):
    freq = [0] * 10
    for d in digits:
        freq[d] += 1
    N = len(digits)
    if N == 0:
        return 0.0
    H = 0.0
    for k in range(10):
        p = freq[k] / N
        if p > 0:
            H -= p * math.log2(p)
    return H

def chi_square_test(recent, reference):
    freq_r   = [0] * 10
    freq_ref = [0] * 10
    for d in recent:    freq_r[d]   += 1
    for d in reference: freq_ref[d] += 1
    n_r, n_ref = len(recent), len(reference)
    if n_r == 0 or n_ref == 0:
        return 0.0
    chi2 = 0.0
    for k in range(10):
        p_ref    = freq_ref[k] / n_ref
        expected = p_ref * n_r
        if expected > 0:
            chi2 += (freq_r[k] - expected) ** 2 / expected
    return chi2


# ============================================================================
# ORDER-1 MARKOV ENGINE
# ============================================================================

class Order1Engine:
    """
    Standard 10×10 first-order Markov chain.
    Conditions on the single most recent digit.
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.sigma  = cfg["conf_sigma_1"]
        # Raw count matrix [from_digit][to_digit]
        self.counts = [[0.0] * 10 for _ in range(10)]
        self.n_obs  = 0          # total transitions seen

    def update(self, prev: int, curr: int):
        """Record a transition prev → curr."""
        self.counts[prev][curr] += 1
        self.n_obs += 1

    def is_ready(self, min_sample: int) -> bool:
        return self.n_obs >= min_sample - 1   # n_obs = ticks - 1

    def _normalised_matrix(self):
        P = [[0.0] * 10 for _ in range(10)]
        for i in range(10):
            rs = sum(self.counts[i])
            if rs > 0:
                P[i] = [v / rs for v in self.counts[i]]
            else:
                P[i] = [0.1] * 10
        return P

    def compute(self, current_digit: int, payout: float,
                edge_thresh: float, max_horizon: int) -> dict | None:
        """
        Returns best-horizon signal dict or None if no edge found.
        Keys: side, horizon, ev, conservative_p, order, horizon_data
        """
        P1           = self._normalised_matrix()
        breakeven    = 1 / (1 + payout)
        min_prob     = breakeven + edge_thresh
        row_n        = max(sum(self.counts[current_digit]), 1)

        best_horizon = None
        best_ev      = -math.inf
        best_side    = None
        best_cons_p  = 0.0
        horizon_data = []

        for n in range(1, max_horizon + 1):
            Pn    = mat_pow(P1, n)
            e_p   = even_prob_from_row(Pn[current_digit])
            o_p   = 1 - e_p

            cons_e = conservative_prob(e_p, row_n, self.sigma)
            cons_o = conservative_prob(o_p, row_n, self.sigma)
            ev_e   = cons_e * payout - (1 - cons_e)
            ev_o   = cons_o * payout - (1 - cons_o)

            horizon_data.append(dict(n=n, e_prob=e_p, o_prob=o_p,
                                     cons_even=cons_e, cons_odd=cons_o,
                                     ev_even=ev_e, ev_odd=ev_o))

            if cons_e >= min_prob and ev_e > best_ev:
                best_ev, best_horizon, best_side, best_cons_p = ev_e, n, "EVEN", cons_e
            if cons_o >= min_prob and ev_o > best_ev:
                best_ev, best_horizon, best_side, best_cons_p = ev_o, n, "ODD",  cons_o

        if best_side is None:
            return None
        return dict(side=best_side, horizon=best_horizon, ev=best_ev,
                    conservative_p=best_cons_p, order=1,
                    horizon_data=horizon_data, row_n=int(row_n))


# ============================================================================
# ORDER-2 MARKOV ENGINE
# ============================================================================

class Order2Engine:
    """
    Second-order Markov chain: conditions on the last TWO digits.

    Structure: a dict of 100 rows, keyed by (prev2, prev1).
    Each row is a length-10 count array over the next digit.
    This is a 10×10×10 = 1 000-cell tensor stored sparsely.

    For multi-step look-ahead the order-2 chain is collapsed to
    an order-1 chain conditioned on the current (prev, curr) pair:
      - The 1-step row is counts[(prev, curr)] normalised.
      - For n > 1 we right-multiply by the order-1 transition matrix
        (the marginal), which is the standard approximation for
        projecting a higher-order chain forward.  This is exact at
        n=1 and an approximation at n>1 — acceptable here because
        order-2 edge at n=2+ is already weak due to ergodicity.
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.sigma  = cfg["conf_sigma_2"]
        # counts[(prev2, prev1)][next_digit]
        self.counts: dict[tuple, list] = defaultdict(lambda: [0.0] * 10)
        self.n_obs  = 0   # total (prev2, prev1, curr) triplets seen

        # We also keep the order-1 marginal for n>1 projection
        self.o1 = Order1Engine(cfg)

    def update(self, prev2: int, prev1: int, curr: int):
        """Record a triplet prev2 → prev1 → curr."""
        self.counts[(prev2, prev1)][curr] += 1
        self.n_obs += 1
        self.o1.update(prev1, curr)   # feed marginal O1

    def is_ready(self, min_sample: int) -> bool:
        return self.n_obs >= min_sample - 2

    def _row_for_pair(self, prev2: int, prev1: int):
        """Normalised probability row for pair (prev2, prev1)."""
        raw = self.counts.get((prev2, prev1), None)
        if raw is None:
            return [0.1] * 10    # unseen pair → uniform
        rs = sum(raw)
        if rs == 0:
            return [0.1] * 10
        return [v / rs for v in raw]

    def _pair_row_n(self, prev2: int, prev1: int) -> int:
        raw = self.counts.get((prev2, prev1), None)
        if raw is None:
            return 0
        return int(sum(raw))

    def compute(self, prev_digit: int, current_digit: int,
                payout: float, edge_thresh: float,
                max_horizon: int) -> dict | None:
        """
        Returns best-horizon signal dict or None.
        Keys: side, horizon, ev, conservative_p, order, horizon_data, row_n
        """
        breakeven = 1 / (1 + payout)
        min_prob  = breakeven + edge_thresh
        row_n     = max(self._pair_row_n(prev_digit, current_digit), 1)

        # 1-step distribution for (prev_digit, current_digit)
        row_1 = self._row_for_pair(prev_digit, current_digit)

        # For n>1 we need the order-1 marginal matrix
        P1_marginal = self.o1._normalised_matrix()

        best_horizon = None
        best_ev      = -math.inf
        best_side    = None
        best_cons_p  = 0.0
        horizon_data = []

        for n in range(1, max_horizon + 1):
            if n == 1:
                dist = row_1
            else:
                # Project: dist × P1_marginal^(n-1)
                # row_1 is a 1×10 vector; multiply by (n-1)-step matrix
                Pn_minus1 = mat_pow(P1_marginal, n - 1)
                dist = [0.0] * 10
                for k in range(10):
                    for j in range(10):
                        dist[j] += row_1[k] * Pn_minus1[k][j]

            e_p = even_prob_from_row(dist)
            o_p = 1 - e_p

            cons_e = conservative_prob(e_p, row_n, self.sigma)
            cons_o = conservative_prob(o_p, row_n, self.sigma)
            ev_e   = cons_e * payout - (1 - cons_e)
            ev_o   = cons_o * payout - (1 - cons_o)

            horizon_data.append(dict(n=n, e_prob=e_p, o_prob=o_p,
                                     cons_even=cons_e, cons_odd=cons_o,
                                     ev_even=ev_e, ev_odd=ev_o))

            if cons_e >= min_prob and ev_e > best_ev:
                best_ev, best_horizon, best_side, best_cons_p = ev_e, n, "EVEN", cons_e
            if cons_o >= min_prob and ev_o > best_ev:
                best_ev, best_horizon, best_side, best_cons_p = ev_o, n, "ODD",  cons_o

        if best_side is None:
            return None
        return dict(side=best_side, horizon=best_horizon, ev=best_ev,
                    conservative_p=best_cons_p, order=2,
                    horizon_data=horizon_data, row_n=int(row_n))


# ============================================================================
# DUAL-ORDER SIGNAL ENGINE  (consensus gate)
# ============================================================================

class DualOrderSignalEngine:
    """
    Wraps Order1Engine + Order2Engine.
    Maintains shared tick history, entropy, and drift state.
    Applies the consensus gate before returning a final signal.
    """

    def __init__(self, cfg: dict):
        self.cfg     = cfg
        self.ticks   = deque()

        self.o1 = Order1Engine(cfg)
        self.o2 = Order2Engine(cfg)

        # Shared state
        self.entropy        = 0.0
        self.last_chi2      = 0.0
        self.drift_detected = False

        # Drift-reset tracking (order-1 only; order-2 resets with it)
        self.matrix_build_tick = 0
        self.matrix_resets     = 0

    def add_tick(self, digit: int):
        ticks = self.ticks
        ticks.append(int(digit))
        n = len(ticks)

        # Feed order-1
        if n >= 2:
            self.o1.update(int(ticks[-2]), int(ticks[-1]))

        # Feed order-2
        if n >= 3:
            self.o2.update(int(ticks[-3]), int(ticks[-2]), int(ticks[-1]))

        # Shared derived state
        self._update_derived()

    def _update_derived(self):
        cfg   = self.cfg
        ticks = self.ticks
        self.entropy = calc_entropy(list(ticks)[-200:])
        n = len(ticks)
        win_r, win_ref = cfg["recent_win"], cfg["ref_win"]
        if n >= win_r + win_ref:
            recent    = list(ticks)[-win_r:]
            reference = list(ticks)[-(win_r + win_ref):-win_r]
            self.last_chi2      = chi_square_test(recent, reference)
            self.drift_detected = self.last_chi2 > cfg["drift_thresh"]
        else:
            self.drift_detected = False

    # ------------------------------------------------------------------
    # Main signal computation with consensus gate
    # ------------------------------------------------------------------

    def compute_signal(self) -> dict | None:
        """
        Returns a combined signal dict, or None if not enough data.

        Signal dict keys:
          side          — "EVEN" or "ODD"
          horizon       — ticks to expiry
          ev            — expected value (best of the two orders)
          conservative_p — conservative win probability
          order_used    — 1 or 2 (which order drove the decision)
          consensus     — "O1_ONLY" | "AGREE" | "DISAGREE"
          sig_o1        — raw order-1 signal (may be None)
          sig_o2        — raw order-2 signal (may be None)
          entropy       — current entropy
          chi2          — current chi-square statistic
          drift         — bool
          ticks_seen    — total ticks
        """
        cfg   = self.cfg
        ticks = self.ticks

        if not self.o1.is_ready(cfg["min_sample_1"]):
            return None    # not enough data for anything yet

        n_ticks       = len(ticks)
        current_digit = int(ticks[-1])
        prev_digit    = int(ticks[-2]) if n_ticks >= 2 else 0

        payout       = cfg["payout"]
        edge_thresh  = cfg["edge_thresh"]
        max_horizon  = cfg["max_horizon"]

        # ---- Order-1 signal -----------------------------------------------
        sig1 = self.o1.compute(current_digit, payout, edge_thresh, max_horizon)

        # ---- Order-2 signal (if ready) ------------------------------------
        o2_ready = self.o2.is_ready(cfg["min_sample_2"])
        sig2     = None
        if o2_ready:
            sig2 = self.o2.compute(prev_digit, current_digit,
                                   payout, edge_thresh, max_horizon)

        # ---- Consensus gate -----------------------------------------------
        # RULE: O1 trades on its own edge at horizon=5, always.
        #       O2 can only UPGRADE a trade to horizon=1 (AGREE state).
        #       O2 never blocks O1 — DISAGREE is logged but not a veto.
        #
        # States:
        #   O1_ONLY   — O2 still warming up; O1 trades at horizon=5
        #   AGREE     — both orders agree; O1 trades at horizon=1
        #   DISAGREE  — orders conflict; O1 still trades at horizon=5
        #               (O2 disagreement is a warning, not a block)
        #   NO_EDGE   — O1 has no edge; nothing to trade regardless of O2

        if sig1 is None:
            # O1 has no edge — nothing to trade
            consensus = "NO_EDGE"
            chosen    = None
        elif not o2_ready:
            # O2 still warming up — O1 trades alone at horizon=5
            consensus = "O1_ONLY"
            chosen    = sig1
        elif sig2 is not None and sig1["side"] == sig2["side"]:
            # Both agree — highest confidence, upgrade to horizon=1
            consensus = "AGREE"
            chosen    = sig1 if sig1["ev"] >= sig2["ev"] else sig2
        elif sig2 is not None and sig1["side"] != sig2["side"]:
            # O2 disagrees — O1 still trades at horizon=5, log the conflict
            consensus = "DISAGREE"
            chosen    = sig1
        else:
            # O2 ready but found no edge — O1 trades at horizon=5
            consensus = "O1_ONLY"
            chosen    = sig1

        # ---- Build return dict --------------------------------------------
        base = dict(
            entropy    = self.entropy,
            chi2       = self.last_chi2,
            drift      = self.drift_detected,
            ticks_seen = n_ticks,
            o1_ready   = True,
            o2_ready   = o2_ready,
            consensus  = consensus,
            sig_o1     = sig1,
            sig_o2     = sig2,
        )

        if chosen is None:
            base.update(side=None, horizon=None, ev=-math.inf,
                        conservative_p=0.0, order_used=0)
        else:
            # ── HORIZON RULE ────────────────────────────────────────
            # Always trade at horizon=5 (default, stable, broader edge).
            # Exception: when O1 and O2 AGREE, both Markov orders are
            # pointing the same way with high confidence — drop to
            # horizon=1 to act on the sharpest, most immediate signal.
            if consensus == "AGREE":
                forced_horizon = 1
            else:
                forced_horizon = 5

            base.update(
                side          = chosen["side"],
                horizon       = forced_horizon,
                ev            = chosen["ev"],
                conservative_p= chosen["conservative_p"],
                order_used    = chosen["order"],
            )
        return base

    def should_trade(self, sig: dict | None) -> tuple[bool, str]:
        cfg = self.cfg
        if sig is None:
            n = len(self.ticks)
            return False, f"Collecting data ({n}/{cfg['min_sample_1']})"
        if sig["drift"]:
            return False, f"Drift detected (chi2={sig['chi2']:.1f} > {cfg['drift_thresh']})"
        if sig["entropy"] >= cfg["entropy_gate"]:
            return False, f"Entropy too high ({sig['entropy']:.3f} >= {cfg['entropy_gate']})"
        if sig["side"] is None:
            return False, "No edge above threshold"
        return True, "OK"

    def handle_drift_reset(self):
        """Reset drift position marker; matrices are NOT cleared
        (clearing would discard all warmup data — we just note the reset
        and let the chi-square window slide forward naturally)."""
        self.matrix_build_tick = len(self.ticks)
        self.matrix_resets    += 1

    @property
    def o2_obs(self) -> int:
        return self.o2.n_obs


# ============================================================================
# MARTINGALE MANAGER
# ============================================================================

class MartingaleManager:
    def __init__(self, cfg: dict):
        self.initial_stake = cfg["initial_stake"]
        self.current_stake = cfg["initial_stake"]
        self.mul           = cfg["martingale_mul"]
        self.max_losses    = cfg["max_losses"]
        self.loss_limit    = cfg["loss_limit"]
        self.loss_streak   = 0
        self.total_profit  = 0.0
        self.wins          = 0
        self.losses        = 0

    def get_stake(self) -> float:
        return round(self.current_stake, 2)

    def record_win(self, profit: float):
        self.wins         += 1
        self.total_profit += profit
        self.loss_streak   = 0
        self.current_stake = self.initial_stake
        print(f"\n  WIN  +${profit:.2f} | stake RESET -> ${self.initial_stake:.2f}")
        self._print_stats()

    def record_loss(self, loss: float):
        self.losses       += 1
        self.total_profit += loss
        self.loss_streak  += 1
        print(f"\n  LOSS  ${abs(loss):.2f} | streak={self.loss_streak}")
        if self.loss_streak >= self.max_losses:
            print(f"  {self.max_losses} losses -> RESET to ${self.initial_stake:.2f}")
            self.current_stake = self.initial_stake
            self.loss_streak   = 0
        else:
            self.current_stake = round(self.current_stake * self.mul, 2)
            print(f"  Martingale L{self.loss_streak} | next stake ${self.current_stake:.2f}")
        self._print_stats()

    def can_trade(self) -> bool:
        return abs(self.total_profit) < self.loss_limit

    def _print_stats(self):
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"  {total} trades | W:{self.wins} L:{self.losses} | WR:{wr:.1f}%")
        print(f"  P&L ${self.total_profit:.2f} | next stake ${self.current_stake:.2f}")
        print(f"{'='*60}")


# ============================================================================
# DERIV CLIENT  v2  (send queue · filtered receive · orphan recovery)
# ============================================================================

class DerivClient:
    def __init__(self, cfg: dict):
        self.api_token = cfg["api_token"]
        self.app_id    = cfg["app_id"]
        self.symbol    = cfg["symbol"]
        self.cfg       = cfg
        self.endpoint  = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
        self.ws        = None
        self._send_queue: asyncio.Queue | None = None
        self._inbox:      asyncio.Queue | None = None
        self._send_task:  asyncio.Task  | None = None
        self._recv_task:  asyncio.Task  | None = None

    async def connect(self) -> bool:
        _log("WS", f"Connecting → {self.symbol}")
        self.ws = await websockets.connect(
            self.endpoint,
            ping_interval=self.cfg["ws_ping_interval"],
            ping_timeout=20,
            close_timeout=10,
        )
        self._send_queue = asyncio.Queue()
        self._inbox      = asyncio.Queue()
        self._start_io_tasks()
        await self.send({"authorize": self.api_token})
        resp = await self.receive_filtered("authorize", timeout=15)
        if resp is None or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("AUTH", f"Failed: {err}")
            return False
        auth = resp.get("authorize", {})
        _log("AUTH", f"OK | Balance: ${auth.get('balance', 0):.2f} {auth.get('currency','')}")
        return True

    def _start_io_tasks(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump(), name="send_pump")
        self._recv_task = asyncio.create_task(self._recv_pump(), name="recv_pump")

    async def _send_pump(self):
        while True:
            data, fut = await self._send_queue.get()
            try:
                await self.ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_queue.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self.ws:
                try:
                    await self._inbox.put(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            _log("RECV", f"Error: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def send(self, data: dict):
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_queue.put((data, fut))
        await fut

    async def receive(self, timeout: float = 10) -> dict:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def receive_filtered(self, msg_type: str, timeout: float = 10) -> dict | None:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    async def subscribe_ticks(self) -> bool:
        await self.send({"ticks": self.symbol, "subscribe": 1})
        resp = await self.receive_filtered("tick", timeout=10)
        if resp is None or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("TICK", f"Subscribe failed: {err}")
            return False
        _log("TICK", f"Subscribed to {self.symbol}")
        return True

    async def place_digit_trade(self, prediction: str, stake: float,
                                horizon: int) -> str | None:
        contract_type = "DIGITEVEN" if prediction == "EVEN" else "DIGITODD"

        # Step 1: proposal
        await self.send({
            "proposal":      1,
            "amount":        stake,
            "basis":         "stake",
            "contract_type": contract_type,
            "currency":      "USD",
            "duration":      horizon,
            "duration_unit": "t",
            "symbol":        self.symbol,
        })
        proposal = await self.receive_filtered("proposal", timeout=12)
        if proposal is None or "error" in proposal:
            err = (proposal or {}).get("error", {}).get("message", "timeout")
            _log("PROPOSAL", f"Error: {err}")
            return None
        proposal_id = proposal.get("proposal", {}).get("id")
        if not proposal_id:
            _log("PROPOSAL", "No proposal ID — skipping")
            return None

        # Step 2: buy with retry loop
        buy_time    = time.time()
        contract_id = None
        await self.send({"buy": proposal_id, "price": stake})
        for attempt in range(self.cfg["buy_recv_retries"]):
            resp = await self.receive_filtered("buy", timeout=8)
            if resp is None:
                _log("BUY", f"No response (attempt {attempt+1})")
                continue
            if "error" in resp:
                _log("BUY", f"Error: {resp['error'].get('message','')}")
                return None
            contract_id = resp.get("buy", {}).get("contract_id")
            if contract_id:
                break

        # Step 3: orphan recovery
        if not contract_id:
            _log("BUY", "No contract_id — querying profit_table")
            contract_id = await self._recover_orphaned_trade(stake, buy_time)
            if contract_id:
                _log("BUY", f"Orphan recovered → {contract_id}")
            else:
                _log("BUY", "Could not recover orphan — bot remains unlocked")
                return None

        _log("TRADE", f"{prediction}  ${stake:.2f}  {horizon}T  contract={contract_id}")

        try:
            await self.send({"proposal_open_contract": 1,
                             "contract_id": contract_id, "subscribe": 1})
        except Exception as exc:
            _log("TRADE", f"Could not subscribe to updates: {exc}")

        return contract_id

    async def _recover_orphaned_trade(self, stake: float,
                                      buy_time: float) -> str | None:
        for attempt in range(self.cfg["orphan_poll_attempts"]):
            await asyncio.sleep(self.cfg["orphan_poll_interval"])
            try:
                await self.send({"profit_table": 1, "description": 1,
                                 "sort": "DESC", "limit": 5})
                resp = await self.receive_filtered("profit_table", timeout=10)
                if not resp or "error" in resp:
                    continue
                for tx in resp.get("profit_table", {}).get("transactions", []):
                    if (abs(float(tx.get("buy_price", 0)) - stake) < 0.01 and
                            float(tx.get("purchase_time", 0)) >= buy_time - 5):
                        return str(tx.get("contract_id"))
            except Exception as exc:
                _log("ORPHAN", f"Poll {attempt+1} error: {exc}")
        return None

    async def poll_contract_result(self, contract_id: str) -> dict | None:
        try:
            await self.send({"proposal_open_contract": 1,
                             "contract_id": contract_id})
            resp = await self.receive_filtered("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc:
            _log("POLL", f"Error: {exc}")
        return None


# ============================================================================
# MAIN BOT
# ============================================================================

class R10MarkovEOBot:
    def __init__(self, cfg: dict = CONFIG):
        self.cfg    = cfg
        self.client = DerivClient(cfg)
        self.engine = DualOrderSignalEngine(cfg)
        self.risk   = MartingaleManager(cfg)
        self.tlog   = TradeLogger(cfg["trades_csv"])

        self.tick_count     = 0
        self.last_eval_tick = 0

        # Store last signal for CSV logging at settlement
        self._last_signal: dict | None = None

        self.current_contract:   dict | None  = None
        self.waiting_for_result: bool         = False
        self.lock_since:         float | None = None

        self._stop = False

    # ------------------------------------------------------------------
    # Lock helpers
    # ------------------------------------------------------------------

    def _unlock(self, reason: str = "manual"):
        if self.waiting_for_result:
            cid = (self.current_contract or {}).get("id", "?")
            _log("UNLOCK", f"Contract {cid} unlocked ({reason})")
        self.waiting_for_result = False
        self.current_contract   = None
        self.lock_since         = None

    def _check_lock_timeout(self):
        if not self.waiting_for_result or self.lock_since is None:
            return
        elapsed = time.monotonic() - self.lock_since
        timeout = self.cfg["lock_timeout_seconds"]
        if elapsed >= timeout:
            _log("TIMEOUT", f"Locked {elapsed:.0f}s > {timeout}s — auto-unlocking")
            self._unlock("timeout")

    # ------------------------------------------------------------------
    # Console listener
    # ------------------------------------------------------------------

    async def _console_listener(self):
        loop = asyncio.get_event_loop()
        _log("CMD", "Commands: [u]nlock  [s]tats  [q]uit")
        while not self._stop:
            try:
                cmd = (await loop.run_in_executor(None, input)).strip().lower()
                if cmd == "u":
                    self._unlock("user command")
                    print("  >> Lock released")
                elif cmd == "s":
                    self.risk._print_stats()
                    eng = self.engine
                    o2_pct = min(100, int(eng.o2_obs /
                                         max(self.cfg["min_sample_2"] - 2, 1) * 100))
                    print(f"  >> Ticks: {self.tick_count}  "
                          f"H={eng.entropy:.4f}  chi2={eng.last_chi2:.2f}  "
                          f"drift={'YES' if eng.drift_detected else 'no'}  "
                          f"O2 warmup: {o2_pct}%")
                    if self.current_contract:
                        elapsed = time.monotonic() - (self.lock_since or 0)
                        print(f"  >> Locked on {self.current_contract['id']} "
                              f"for {elapsed:.0f}s")
                elif cmd in ("q", "quit", "exit"):
                    _log("CMD", "Quit requested")
                    self._stop = True
                    break
            except (EOFError, KeyboardInterrupt):
                break

    # ------------------------------------------------------------------
    # Tick handler
    # ------------------------------------------------------------------

    async def on_tick(self, tick_data: dict):
        quote = tick_data.get("quote")
        if quote is None:
            return

        # Hardened digit extraction (handles float, int, stripped trailing zeros)
        quote_str = str(quote)
        if "." in quote_str:
            digit = int(quote_str.split(".")[-1][-1])
        else:
            digit = int(quote_str[-1])

        self.tick_count += 1
        self.engine.add_tick(digit)
        self._check_lock_timeout()

        if self.tick_count % 5 == 0:
            eng    = self.engine
            status = "WAITING" if self.waiting_for_result else "READY"
            o2_pct = min(100, int(eng.o2_obs /
                                  max(self.cfg["min_sample_2"] - 2, 1) * 100))
            print(
                f"\r  #{self.tick_count}  d={digit}  "
                f"H={eng.entropy:.3f}  "
                f"O2={o2_pct}%  {status}  {_ts()}",
                end="", flush=True,
            )

        interval = self.cfg["trade_interval_ticks"]
        if ((self.tick_count - self.last_eval_tick) >= interval
                and not self.waiting_for_result):
            print()
            self.last_eval_tick = self.tick_count
            await self.evaluate_and_trade()

    # ------------------------------------------------------------------
    # Evaluate and trade
    # ------------------------------------------------------------------

    async def evaluate_and_trade(self):
        if self.waiting_for_result:
            return

        sig        = self.engine.compute_signal()
        ok, reason = self.engine.should_trade(sig)

        print(f"\n{'='*60}")
        print(f"SIGNAL  (tick #{self.tick_count})")
        if sig:
            # Consensus status line
            consensus = sig.get("consensus", "?")
            o2_label  = "READY" if sig["o2_ready"] else f"warming ({self.engine.o2_obs}/{self.cfg['min_sample_2']-2})"
            print(f"  Consensus  : {consensus}  |  O2: {o2_label}")

            if sig["sig_o1"]:
                s1 = sig["sig_o1"]
                print(f"  O1 signal  : {s1['side']}  "
                      f"P={s1['conservative_p']*100:.2f}%  "
                      f"EV={s1['ev']:+.4f}  "
                      f"H={s1['horizon']}T  "
                      f"row_n={s1['row_n']}")
            else:
                print(f"  O1 signal  : no edge")

            if sig["o2_ready"]:
                if sig["sig_o2"]:
                    s2 = sig["sig_o2"]
                    print(f"  O2 signal  : {s2['side']}  "
                          f"P={s2['conservative_p']*100:.2f}%  "
                          f"EV={s2['ev']:+.4f}  "
                          f"H={s2['horizon']}T  "
                          f"row_n={s2['row_n']}")
                else:
                    print(f"  O2 signal  : no edge")

            if sig["side"]:
                print(f"  ── Final ──")
                print(f"  Side       : {sig['side']}")
                print(f"  Cons. P    : {sig['conservative_p']*100:.2f}%")
                print(f"  EV         : {sig['ev']:+.4f}")
                print(f"  Horizon    : {sig['horizon']} tick(s)")
                print(f"  Order used : {sig['order_used']}")

            print(f"  Entropy    : {sig['entropy']:.4f}  "
                  f"chi2={sig['chi2']:.2f}  "
                  f"drift={'YES' if sig['drift'] else 'no'}")

        print(f"  Decision   : {'TRADE' if ok else f'SKIP -- {reason}'}")
        print(f"{'='*60}")

        if not ok:
            if sig and sig.get("drift"):
                self.engine.handle_drift_reset()
                _log("DRIFT", f"Drift reset #{self.engine.matrix_resets} "
                              f"at tick {self.engine.matrix_build_tick}")
            return

        if not self.risk.can_trade():
            _log("RISK", f"Loss limit reached (${self.risk.total_profit:.2f}). Stopping.")
            self._stop = True
            return

        stake       = self.risk.get_stake()
        contract_id = await self.client.place_digit_trade(
            sig["side"], stake, sig["horizon"])

        if contract_id:
            self.current_contract   = {
                "id":         contract_id,
                "stake":      stake,
                "prediction": sig["side"],
                "horizon":    sig["horizon"],
                "time":       datetime.now(),
            }
            self._last_signal       = sig   # save for CSV at settlement
            self.waiting_for_result = True
            self.lock_since         = time.monotonic()
            print(f"  Trade locked — awaiting result...")
        else:
            print(f"  Trade placement failed — bot remains READY")

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def _is_settled(self, data: dict) -> bool:
        if data.get("is_settled"):
            return True
        for key in ("status", "contract_status"):
            if data.get(key, "").lower() in ("sold", "won", "lost"):
                return True
        return False

    async def handle_settlement(self, contract_data: dict):
        cid = contract_data.get("contract_id")
        if not self.current_contract or cid != self.current_contract["id"]:
            return None
        if not self._is_settled(contract_data):
            return None

        profit = float(contract_data.get("profit", 0))
        status = contract_data.get("status", "unknown")
        win    = profit > 0
        print(f"\n{'='*60}")
        print(f"RESULT  contract={cid}  status={status}  profit=${profit:.2f}")
        print(f"{'='*60}")

        # ── CSV logging ───────────────────────────────────────────
        sig = self._last_signal or {}
        try:
            self.tlog.record(
                direction      = self.current_contract.get("prediction", "?"),
                horizon        = self.current_contract.get("horizon", 0),
                stake          = self.current_contract.get("stake", 0.0),
                profit         = profit,
                win            = win,
                consensus      = sig.get("consensus", "?"),
                order_used     = sig.get("order_used", 0),
                conservative_p = sig.get("conservative_p", 0.0),
                ev             = sig.get("ev", 0.0),
                entropy        = sig.get("entropy", 0.0),
                chi2           = sig.get("chi2", 0.0),
                drift          = sig.get("drift", False),
                loss_streak    = self.risk.loss_streak,
                balance_approx = self.risk.initial_stake +
                                  self.risk.total_profit + profit,
            )
        except Exception as e:
            _log("CSV", f"Logging error: {e}")

        if win:
            self.risk.record_win(profit)
        else:
            self.risk.record_loss(profit)

        self._unlock("settlement")
        self._last_signal = None
        print("  Trade unlocked — ready for next trade")
        return self.risk.can_trade()

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    async def _reconnect(self) -> bool:
        delay, max_d, attempt = self.cfg["reconnect_delay_min"], \
                                self.cfg["reconnect_delay_max"], 0
        while not self._stop:
            attempt += 1
            _log("RECONNECT", f"Attempt {attempt} in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_d)
            await self.client.close()
            self.client = DerivClient(self.cfg)
            try:
                if not await self.client.connect():
                    continue
                if not await self.client.subscribe_ticks():
                    continue
                if self.waiting_for_result and self.current_contract:
                    cid  = self.current_contract["id"]
                    _log("RECONNECT", f"Re-attaching to contract {cid}")
                    data = await self.client.poll_contract_result(cid)
                    if data:
                        await self.handle_settlement(data)
                    await self.client.send({"proposal_open_contract": 1,
                                            "contract_id": cid, "subscribe": 1})
                _log("RECONNECT", "Reconnected successfully")
                return True
            except Exception as exc:
                _log("RECONNECT", f"Error: {exc}")
        return False

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self):
        cfg = self.cfg
        print("\n" + "="*60)
        print("R_10 EVEN/ODD BOT — DUAL-ORDER MARKOV  v3")
        print("="*60)
        print(f"  Symbol       : {cfg['symbol']}")
        print(f"  Stake        : ${cfg['initial_stake']}  "
              f"(mart. x{cfg['martingale_mul']}, reset @{cfg['max_losses']} losses)")
        print(f"  O1 warmup    : {cfg['min_sample_1']} ticks  "
              f"(sigma={cfg['conf_sigma_1']})")
        print(f"  O2 warmup    : {cfg['min_sample_2']} ticks  "
              f"(sigma={cfg['conf_sigma_2']})")
        print(f"  Edge thresh  : +{cfg['edge_thresh']*100:.1f}% over breakeven")
        print(f"  Entropy gate : {cfg['entropy_gate']}")
        print(f"  Drift chi2   : {cfg['drift_thresh']}")
        print(f"  Horizons     : 1–{cfg['max_horizon']} ticks")
        print(f"  Lock timeout : {cfg['lock_timeout_seconds']}s")
        print("="*60)
        print("\nConsensus logic:")
        print("  O2 not ready  → trade on O1 alone")
        print("  O2 ready, agree    → trade (highest confidence)")
        print("  O2 ready, disagree → skip")
        print("="*60 + "\n")

        if not await self.client.connect():
            return
        if not await self.client.subscribe_ticks():
            return

        print(f"Bot running!  O1 live after {cfg['min_sample_1']} ticks, "
              f"O2 live after {cfg['min_sample_2']} ticks.\n")

        console_task = asyncio.create_task(
            self._console_listener(), name="console_listener")

        try:
            while not self._stop:
                response = await self.client.receive(timeout=60)

                if "__disconnect__" in response:
                    _log("WS", "Connection lost — reconnecting")
                    if not await self._reconnect():
                        break
                    continue

                if not response:
                    try:
                        await self.client.ws.ping()
                    except Exception:
                        _log("WS", "Ping failed — reconnecting")
                        if not await self._reconnect():
                            break
                    continue

                if "tick" in response:
                    await self.on_tick(response["tick"])

                if "proposal_open_contract" in response:
                    result = await self.handle_settlement(
                        response["proposal_open_contract"])
                    if result is False:
                        break

                if "buy" in response:
                    result = await self.handle_settlement(response["buy"])
                    if result is False:
                        break

                if "transaction" in response:
                    trans = response["transaction"]
                    if "contract_id" in trans:
                        result = await self.handle_settlement({
                            "contract_id": trans.get("contract_id"),
                            "profit":      trans.get("profit", 0),
                            "status":      trans.get("action", ""),
                            "is_settled":  True,
                        })
                        if result is False:
                            break

                if "profit_table" in response and self.current_contract:
                    for tx in response["profit_table"].get("transactions", []):
                        if tx.get("contract_id") == self.current_contract["id"]:
                            result = await self.handle_settlement({
                                "contract_id": tx["contract_id"],
                                "profit":      (float(tx.get("sell_price", 0))
                                                - float(tx.get("buy_price", 0))),
                                "status":      "sold",
                                "is_settled":  True,
                            })
                            if result is False:
                                break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as exc:
            print(f"\nUnhandled error: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            console_task.cancel()
            await self.client.close()
            print("\nFINAL STATS")
            self.risk._print_stats()
            print(f"  Ticks processed  : {self.tick_count}")
            print(f"  O1 transitions   : {self.engine.o1.n_obs}")
            print(f"  O2 triplets      : {self.engine.o2_obs}")
            print(f"  Matrix resets    : {self.engine.matrix_resets}")
            print("\nGoodbye")


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    if not CONFIG["api_token"]:
        print("ERROR: DERIV_API_TOKEN environment variable is not set.")
        print("Set it in Railway → Variables before deploying.")
        return
    bot = R10MarkovEOBot(CONFIG)
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
