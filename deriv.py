"""
Deriv live tick stream + last-digit repeat model, in one file.

Setup:     pip install websockets
Live run:  python deriv_digit_stream.py                       (Ctrl+C to stop; logs to ticks.csv)
Analyze:   python deriv_digit_stream.py analyze [file] [symbol]   (stats on the logged ticks)
Self-test: python deriv_digit_stream.py selftest              (simulated data, no network)

For each tick it prints the price (2 decimals), the second-decimal digit, the
theoretical probability that digit repeats within the window, and the rate
actually observed in the data so far (with a 95% confidence interval check).
Every tick is appended to ticks.csv so you can test for non-randomness offline.
"""
import asyncio
import csv
import json
import os
import random
import sys
import time
from collections import deque
from math import erfc, exp, lgamma, log, sqrt

import websockets

# ---------------------------------------------------------------- settings
URL = "wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOLS = ["1HZ10V"]
APP_PING_EVERY = 30        # seconds between Deriv-level {"ping": 1} messages
GAP_WARN_SECONDS = 5       # warn if consecutive ticks are further apart than this
DIGIT_WINDOW_SECONDS = 3   # "same digit again within N seconds"
TICK_INTERVAL = 1.0        # 1.0 for 1HZ symbols; 2.0 for R_10 / R_50 / R_100
MIN_SAMPLES = 200          # don't trust the observed rate below this many samples
LOG_FILE = "ticks.csv"     # every tick is appended here; set to None to disable


# ------------------------------------------------------------- digit model
def wilson(hits: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion."""
    if trials == 0:
        return 0.0, 1.0
    p = hits / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return centre - half, centre + half


class DigitModel:
    def __init__(self, window_seconds: float = 3, tick_interval: float = 1.0,
                 max_history: int = 50_000):
        self.window = window_seconds
        self.n_ticks = max(1, round(window_seconds / tick_interval))
        self.ticks: deque[tuple[int, int]] = deque(maxlen=max_history)  # (epoch, digit)

    @staticmethod
    def digit_of(quote) -> int:
        """Second decimal place, e.g. 245.67 -> 7, 245.6 -> 0."""
        return int(f"{float(quote):.2f}"[-1])

    def add(self, epoch: int, quote) -> int:
        digit = self.digit_of(quote)
        self.ticks.append((epoch, digit))
        return digit

    def theoretical(self) -> float:
        """P(digit appears at least once in the next n ticks) if digits are uniform and independent."""
        return 1 - 0.9 ** self.n_ticks

    def empirical(self, digit: int | None = None) -> tuple[int, int]:
        """(hits, trials): how often a tick's digit reappeared within the window."""
        data = list(self.ticks)
        if not data:
            return 0, 0
        last_epoch = data[-1][0]
        hits = trials = 0
        for i, (epoch, d) in enumerate(data):
            if epoch + self.window > last_epoch:
                break                      # window not fully observed yet
            if digit is not None and d != digit:
                continue
            hit = False
            k = i + 1
            while k < len(data) and data[k][0] <= epoch + self.window:
                if data[k][1] == d:
                    hit = True
                    break
                k += 1
            trials += 1
            hits += hit
        return hits, trials

    def predict(self, quote) -> dict:
        digit = self.digit_of(quote)
        base = self.theoretical()
        out = {
            "digit": digit,
            "probability": base,           # the number to use
            "samples": 0,
            "empirical": None,
            "ci": None,
            "differs_from_theory": False,
        }
        hits, trials = self.empirical(digit)
        out["samples"] = trials
        if trials >= MIN_SAMPLES:
            lo, hi = wilson(hits, trials)
            out["empirical"] = hits / trials
            out["ci"] = (lo, hi)
            out["differs_from_theory"] = not (lo <= base <= hi)
        return out


# ------------------------------------------------------------ live stream
last_epoch: dict[str, int] = {}              # survives reconnects, used to detect missed ticks
models: dict[str, DigitModel] = {}           # one digit model per symbol


def handle_tick(tick: dict) -> None:
    symbol = tick.get("symbol")
    epoch = tick.get("epoch")
    quote = tick.get("quote")

    prev = last_epoch.get(symbol)
    if prev is not None and epoch is not None and epoch <= prev:
        return                               # duplicate or out-of-order tick (e.g. after reconnect)

    if LOG_FILE and epoch is not None:
        is_new = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["symbol", "epoch", "quote"])
            w.writerow([symbol, epoch, quote])   # raw quote, full precision

    if prev is not None and epoch is not None and epoch - prev > GAP_WARN_SECONDS:
        print(f"  !! gap of {epoch - prev}s on {symbol}; backfill with ticks_history if needed")
    if epoch is not None:
        last_epoch[symbol] = epoch

    try:
        price = f"{float(quote):.2f}"
    except (TypeError, ValueError):
        price = str(quote)                   # missing or non-numeric quote: show as-is

    stamp = time.strftime("%H:%M:%S", time.localtime(epoch or time.time()))

    try:
        model = models.setdefault(symbol, DigitModel(DIGIT_WINDOW_SECONDS, TICK_INTERVAL))
        model.add(epoch, quote)
        p = model.predict(quote)
    except (TypeError, ValueError):
        print(f"{stamp}  {symbol}  {price}")
        return

    line = (f"{stamp}  {symbol}  {price}  digit={p['digit']}  "
            f"P(repeat in {DIGIT_WINDOW_SECONDS}s)={p['probability']:.1%}")
    if p["empirical"] is not None:
        line += f"  observed={p['empirical']:.1%} (n={p['samples']})"
        if p["differs_from_theory"]:
            line += "  [outside CI]"
    print(line)


async def app_pinger(ws) -> None:
    """Application-level keepalive so the server doesn't treat the session as idle."""
    while True:
        await asyncio.sleep(APP_PING_EVERY)
        await ws.send(json.dumps({"ping": 1}))


async def stream(ws) -> None:
    # Subscriptions belong to a connection, so they must be re-sent after every reconnect
    for sym in SYMBOLS:
        await ws.send(json.dumps({"ticks": sym, "subscribe": 1}))

    pinger = asyncio.create_task(app_pinger(ws))
    try:
        async for raw in ws:
            msg = json.loads(raw)

            if "error" in msg:
                err = msg["error"]
                print(f"API error [{err.get('code')}]: {err.get('message')}")
                continue

            if msg.get("msg_type") == "tick" and "tick" in msg:
                handle_tick(msg["tick"])
    finally:
        pinger.cancel()


async def run_forever() -> None:
    backoff = 1
    while True:
        try:
            # ping_interval/ping_timeout = protocol-level keepalive that detects dead links
            async with websockets.connect(URL, ping_interval=20, ping_timeout=20) as ws:
                print("Connected")
                backoff = 1                  # reset after a successful connection
                await stream(ws)
            print("Server closed the connection")
        except (websockets.WebSocketException, OSError, asyncio.TimeoutError) as e:
            print(f"Connection problem: {type(e).__name__}: {e}")

        delay = backoff + random.random()    # jitter avoids reconnect stampedes
        print(f"Reconnecting in {delay:.1f}s")
        await asyncio.sleep(delay)
        backoff = min(backoff * 2, 60)


# ---------------------------------------------------------- offline analysis
def chi2_sf(x: float, df: int) -> float:
    """P(Chi-square_df >= x), via the regularised lower incomplete gamma series."""
    if x <= 0:
        return 1.0
    a, z = df / 2, x / 2
    term = total = 1.0 / a
    n = a
    for _ in range(10_000):
        n += 1
        term *= z / n
        total += term
        if term < total * 1e-14:
            break
    lower = total * exp(-z + a * log(z) - lgamma(a))
    return max(0.0, 1.0 - lower)


def normal_p(z: float) -> float:
    """Two-sided p-value for a standard normal z-score."""
    return erfc(abs(z) / sqrt(2))


def run_tests(digits: list[int], epochs: list[int], step: int) -> tuple[list, list]:
    """Return ([(name, statistic, p)], skipped-notes) for one segment of ticks."""
    n = len(digits)
    results, notes = [], []

    # 1. Are the 10 digits equally frequent?
    counts = [0] * 10
    for d in digits:
        counts[d] += 1
    exp_c = n / 10
    chi = sum((c - exp_c) ** 2 / exp_c for c in counts)
    results.append(("Digits uniform (chi2, df=9)", chi, chi2_sf(chi, 9)))

    # Consecutive-tick pairs, only where no ticks were missed between them
    pairs = [(digits[i], digits[i + 1]) for i in range(n - 1) if epochs[i + 1] - epochs[i] == step]
    m = len(pairs)

    # 2. Does the current digit change the distribution of the next one?
    if m >= 1000:
        table = [[0] * 10 for _ in range(10)]
        for a, b in pairs:
            table[a][b] += 1
        rows = [sum(r) for r in table]
        cols = [sum(table[i][j] for i in range(10)) for j in range(10)]
        chi = sum((table[i][j] - rows[i] * cols[j] / m) ** 2 / (rows[i] * cols[j] / m)
                  for i in range(10) for j in range(10) if rows[i] and cols[j])
        results.append(("Next digit independent of current (chi2, df=81)", chi, chi2_sf(chi, 81)))
    else:
        notes.append("transition test skipped (needs >= 1000 consecutive pairs)")

    if m >= 200:
        # 3. Does the next tick repeat the digit more or less than 10%?
        reps = sum(a == b for a, b in pairs)
        z = (reps - m * 0.1) / sqrt(m * 0.1 * 0.9)
        results.append(("Immediate repeat rate vs 10% (z)", z, normal_p(z)))

        # 4. Lag-1 correlation between consecutive digits
        xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
        mx, my = sum(xs) / m, sum(ys) / m
        sxy = sum((x - mx) * (y - my) for x, y in pairs)
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        r = sxy / sqrt(sxx * syy) if sxx and syy else 0.0
        results.append(("Lag-1 digit correlation (z = r*sqrt(n))", r * sqrt(m), normal_p(r * sqrt(m))))
    else:
        notes.append("repeat and correlation tests skipped (needs >= 200 pairs)")

    # 5. Your original question: repeat within the window vs 1 - 0.9**n
    model = DigitModel(DIGIT_WINDOW_SECONDS, TICK_INTERVAL, max_history=n + 1)
    model.ticks.extend(zip(epochs, digits))
    hits, trials = model.empirical()
    if trials >= MIN_SAMPLES:
        base = model.theoretical()
        z = (hits / trials - base) / sqrt(base * (1 - base) / trials)
        results.append((f"Repeat within {DIGIT_WINDOW_SECONDS}s vs {base:.1%} (z, overlapping windows)",
                        z, normal_p(z)))
    return results, notes


def analyze(path: str = LOG_FILE or "ticks.csv", symbol: str | None = None) -> None:
    if not os.path.exists(path):
        print(f"{path} not found. Run the live stream first to record ticks.")
        return

    by_symbol: dict[str, dict[int, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_symbol.setdefault(row["symbol"], {})[int(row["epoch"])] = row["quote"]

    symbol = symbol or SYMBOLS[0]
    if symbol not in by_symbol:
        print(f"No ticks for {symbol} in {path}. Symbols found: {list(by_symbol)}")
        return

    ticks = sorted(by_symbol[symbol].items())
    epochs = [e for e, _ in ticks]
    digits = [DigitModel.digit_of(q) for _, q in ticks]
    step = round(TICK_INTERVAL)
    n = len(digits)
    hours = (epochs[-1] - epochs[0]) / 3600
    print(f"{symbol}: {n:,} ticks over {hours:.1f} hours from {path}")
    if n < 1000:
        print("Too few ticks for a meaningful test. Let the stream run for hours.")
        return

    half = n // 2
    segments = {
        "First half": (digits[:half], epochs[:half]),
        "Second half": (digits[half:], epochs[half:]),
        "All data": (digits, epochs),
    }

    flagged: dict[str, set] = {}
    for label, (dg, ep) in segments.items():
        results, notes = run_tests(dg, ep, step)
        alpha = 0.05 / len(results)              # Bonferroni: we are running several tests
        print(f"\n{label} (n={len(dg):,}, flag if p < {alpha:.4f})")
        for name, stat, p in results:
            mark = "  <-- FLAGGED" if p < alpha else ""
            print(f"  {name:<62} stat={stat:8.2f}  p={p:.4f}{mark}")
            if p < alpha:
                flagged.setdefault(label, set()).add(name)
        for note in notes:
            print(f"  note: {note}")

    both = flagged.get("First half", set()) & flagged.get("Second half", set())
    print("\nVerdict")
    if both:
        print("  Flagged independently in BOTH halves, so worth a closer look:")
        for name in sorted(both):
            print(f"   - {name}")
        print("  Confirm on a brand-new recording before trusting it.")
    elif flagged:
        print("  Flagged in only one segment. A pattern that doesn't replicate in the")
        print("  other half is almost certainly chance.")
    else:
        print("  No test flagged. No evidence the digits are anything but random.")

    # Conditional table: P(next digit | current digit), rows = current, columns = next
    pairs = [(digits[i], digits[i + 1]) for i in range(n - 1) if epochs[i + 1] - epochs[i] == step]
    table = [[0] * 10 for _ in range(10)]
    for a, b in pairs:
        table[a][b] += 1
    print("\nP(next digit | current digit), % (every cell should be near 10.0)")
    print("      " + "".join(f"{j:>6}" for j in range(10)))
    for i in range(10):
        total = sum(table[i]) or 1
        print(f"  {i:>2}  " + "".join(f"{100 * table[i][j] / total:6.1f}" for j in range(10)))


# --------------------------------------------------------------- self-test
def selftest() -> None:
    """Simulated uniform random digits, one tick per second: observed should match theory."""
    random.seed(1)
    model = DigitModel(window_seconds=3, tick_interval=1.0)
    for t in range(200_000):
        model.add(t, f"{random.randint(0, 99999) / 100:.2f}")

    print(f"Theoretical P(repeat within 3 ticks): {model.theoretical():.4f}")
    hits, trials = model.empirical()
    lo, hi = wilson(hits, trials)
    print(f"Observed, all digits: {hits / trials:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  n={trials}")

    flagged = 0
    for d in range(10):
        h, n = model.empirical(d)
        lo, hi = wilson(h, n)
        off = not (lo <= model.theoretical() <= hi)
        flagged += off
        print(f"  digit {d}: {h / n:.4f}  [{lo:.4f}, {hi:.4f}]{'  <- outside CI' if off else ''}")
    print(f"Digits outside CI: {flagged}/10 (about 1 in 20 is expected by chance alone)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "analyze":
        analyze(*(sys.argv[2:4]))
    else:
        try:
            asyncio.run(run_forever())
        except KeyboardInterrupt:
            print("\nStopped")