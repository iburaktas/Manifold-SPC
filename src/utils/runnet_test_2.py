# main.py
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (1) Optional: keep BLAS single-threaded per *process* to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# (2) Regular imports from your project
from src.data.dataloader import DataLoader_
from src.utils.runner import Runner, ExperimentConfig  # adjust import to your layout

def make_runner():
    # Build dataloader
    data_loader = DataLoader_(
        exp_type="TE",
        fault_no=4,
        amplitude=0.1,
        N=10,
    )

    # Build config
    exp_config = ExperimentConfig()
    exp_config.n_ar = 400
    exp_config.ndim = 22
    exp_config.estimate_sig = True
    exp_config.sigma = 0.00039
    exp_config.mf_constants = [22,14,22]
    exp_config.d = 22
    exp_config.pickle_path = Path(Path(data_loader.data_path).parent / "TEpkl")

    # Create runner (set n_workers > 1 to use ProcessPoolExecutor in your run())
    return Runner(data_loader, exp_config, n_workers=8)

if __name__ == "__main__":
    runner = make_runner()
    # print(runner.pickles)
    runner.run()
