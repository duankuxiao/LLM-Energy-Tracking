import numpy as np

try:
    from .util import get_property
except ImportError:
    from util import get_property


class Solution:
    """Container for a decoded deployment plan and its objective values."""

    def __init__(self):
        self.chromosome = None
        self.obj = None
        self.sol = None
        self.imp_percent = None
        self.ratio = None


class LLMEnergy:
    """
    Unconstrained global deployment model.

    Training and inference can be redistributed across all countries as long as
    global training demand and global inference demand remain unchanged.
    """

    def __init__(self, scene="Base", rep="CP"):
        (
            region,
            gw_country,
            r_train,
            r_infer,
            self.emission_data,
            self.water_data,
            self.pue,
            self.wue,
            self.countries,
        ) = get_property(scene, rep)

        ny, nc = gw_country.shape
        self.dim = nc

        p_train, p_infer = 0.3, 0.7
        u_train, u_infer = 0.8, 0.5
        p_max, p_min = 0.88, 0.23

        # Upper bounds represent the annual electricity each country could absorb
        # if all training or all inference were allocated there.
        self.up_limit_train = (
            gw_country * (p_min + (p_max - p_min) * u_train * r_train.repeat(nc).reshape(-1, nc)) * 8760
        )
        self.up_limit_infer = (
            gw_country * (p_min + (p_max - p_min) * u_infer * r_infer.repeat(nc).reshape(-1, nc)) * 8760
        )

        # Benchmark results correspond to the default 30/70 workload split with
        # no cross-border reallocation.
        self.energy_benchmark = (
            self.evaluate(self.up_limit_train * p_train) + self.evaluate(self.up_limit_infer * p_infer)
        )
        self.need_train = np.sum(self.up_limit_train * p_train, axis=1)
        self.need_infer = np.sum(self.up_limit_infer * p_infer, axis=1)

    def evaluate(self, power):
        """Aggregate total water use and carbon emissions for a power matrix."""
        power_pue = power * self.pue
        water = (self.wue + self.water_data) * power_pue
        carbon = self.emission_data * power_pue
        return np.array([water.sum(), carbon.sum()])

    def adjustment(self, pct, capacity, need, is_overflow=False):
        """
        Repair a candidate allocation so annual demand exactly matches the target.

        The implementation follows the original heuristic and balances only
        against the first year because capacity scales monotonically over time.
        """
        overflow = (capacity * pct).sum(axis=1) - need

        if is_overflow and overflow.min() < -1e-9:
            print("Invalid solution:", overflow.min())
            raise SystemExit(1)

        y = 0
        y_overflow = overflow[y]

        while y_overflow < -1e-9:
            indices = np.arange(len(pct))[1 > pct]
            idx = np.random.choice(indices)
            add_pct = min(-y_overflow / capacity[y, idx], 1 - pct[idx])
            pct[idx] += add_pct
            y_overflow += capacity[y, idx] * add_pct

        while y_overflow > 1e-9:
            indices = np.arange(len(pct))[pct > 0]
            idx = np.random.choice(indices)
            add_pct = min(y_overflow / capacity[y, idx], pct[idx])
            pct[idx] -= add_pct
            y_overflow -= capacity[y, idx] * add_pct

        return pct

    def decode(self, chromosome):
        """Decode a chromosome into feasible training and inference allocations."""
        x_infer = chromosome.copy()
        x_infer = self.adjustment(x_infer, self.up_limit_infer, self.need_infer)
        x_train = self.adjustment(1 - x_infer, self.up_limit_train, self.need_train, True)
        return x_train, x_infer

    def __call__(self, chromosome):
        """Evaluate one chromosome and return a populated Solution object."""
        x_train, x_infer = self.decode(chromosome)
        pw_train = self.up_limit_train * x_train
        pw_infer = self.up_limit_infer * x_infer

        plan = Solution()
        plan.chromosome = x_infer

        energy = self.evaluate(pw_train) + self.evaluate(pw_infer)
        plan.sol = {"train": pw_train, "infer": pw_infer}
        plan.ratio = {
            "train": pw_train[0] / self.up_limit_train[0],
            "infer": pw_infer[0] / self.up_limit_infer[0],
        }
        plan.imp_percent = dict(
            zip(["water", "carbon"], (energy - self.energy_benchmark) / self.energy_benchmark)
        )
        plan.obj = (energy / self.energy_benchmark).sum()
        return plan
