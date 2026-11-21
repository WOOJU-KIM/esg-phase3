"""Global configuration for the 4x3 probe-card wafer test-time simulator."""

# ---- Wafer geometry (used by the dummy-data generator) ----
GRID_SIZE = 28            # 28 x 28 die grid
WAFER_RADIUS = 13.0       # in die units (die center must be inside)
RANDOM_SEED = 20261121

# ---- Probe card ----
CARD_ROWS = 4
CARD_COLS = 3
DUT_MAX = CARD_ROWS * CARD_COLS

# ---- Timing model ----
INDEX_TIME = 2.0          # move + chuck up/down per shot [s]
T_FULL = 60.0             # T(12) [s]
STEP_RATIO = 0.03         # 3 %p (additive) or 3 % (compound) per missing good die
TIME_MODE = "additive"    # "additive": 60*(1-0.03*(12-k)) | "compound": 60*0.97**(12-k)

# ---- Files ----
INPUT_CSV = "wafer_yield_input.csv"
RESULT_JSON = "simulation_results.json"
OUTPUT_HTML = "wafer_probe_simulation.html"

# ---- Solver budgets ----
ILP_SETCOVER_TIME_S = 30
MILP_TIME_S = 60
MILP_REL_GAP = 0.01
LNS_TIME_S = 150
LNS_SUB_TIME_S = 4
TSP_TIME_S = 15           # OR-Tools routing (TSP) time limit per method

# ---- Monte Carlo ----
MC_WAFERS = 2000
MC_SEED = 7


def shot_time(k: int, mode: str = None) -> float:
    """Test time for a shot with k good dies (k=0 -> early abort = 0 s)."""
    mode = mode or TIME_MODE
    if k <= 0:
        return 0.0
    if mode == "compound":
        return T_FULL * (1.0 - STEP_RATIO) ** (DUT_MAX - k)
    return T_FULL * (1.0 - STEP_RATIO * (DUT_MAX - k))


def time_table(mode: str = None):
    return [shot_time(k, mode) for k in range(DUT_MAX + 1)]


# Linear coefficients for additive mode: T(k) = A + B*k (k>=1)
LIN_A = T_FULL * (1.0 - STEP_RATIO * DUT_MAX)   # 38.4
LIN_B = T_FULL * STEP_RATIO                      # 1.8
