import argparse
from pathlib import Path
from src.data.dataloader import DataLoader_
from src.utils.runner import Runner
from src.utils.experimentconfig import ExperimentConfig


def make_runner(**kwargs) -> Runner:
    exp_type = kwargs.pop("exp_type")
    exp_name = kwargs.pop("exp_name")
    n_workers = kwargs.pop("n_workers")
    N = kwargs.pop("N")
    fault_no = kwargs.pop("fault_no", 4)
    amplitude = kwargs.pop("amplitude", 0.0)
    alpha = kwargs.pop("alpha", 0.05)
    sigma = kwargs.pop("sigma", None)
    sigma_estimate = kwargs.pop("sigma_estimate")
    sigma_estimate = bool(sigma_estimate)

    exp_config = ExperimentConfig()
    exp_config.exp_name = exp_name
    exp_config.alpha = alpha
    exp_config.estimate_sig = sigma_estimate
    #print(sigma_estimate)
    if exp_type == "TE":
        data_loader = DataLoader_(exp_type= "TE",N=N,fault_no=fault_no,amplitude=amplitude)
        exp_config.ndim = 22
        exp_config.sigma = 0.00039 if sigma is None else sigma
        exp_config.mf_constants = [22,14,22]
        exp_config.d = 22
        exp_config.pickle_path = Path(Path(data_loader.data_path).parent / "TEpkl")
    elif exp_type == "KolektorSDD":
        data_loader = DataLoader_(exp_type="KolektorSDD")
        exp_config.n_ar = 0
        exp_config.n_manifold = 330
        exp_config.n_cc = 17
        exp_config.methods = ["MF"]
        exp_config.sigma = 0.0808 if sigma is None else sigma
        exp_config.mf_constants = [1100,700,1100]
        n_workers = 1
        exp_config.p = 0
    elif exp_type == "SyntheticProcess":
        data_loader = DataLoader_(exp_type="SyntheticProcess",N=N,fault_no=fault_no,amplitude=amplitude)
        exp_config.sigma = 0.05 if sigma is None else sigma
        exp_config.mf_constants = [5,3,5]
        exp_config.d = 2
        exp_config.n_dim = 3
        exp_config.p = 10
    else:
        raise ValueError(f"Unsupported exp_type: {exp_type}")

    
    
    return Runner(data_loader, exp_config, n_workers=n_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--exp-type", type=str, required=True)
    parser.add_argument("--exp-name", type=str, default="Test")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--fault-no", type=int, default=4)
    parser.add_argument("--amplitude", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--sigma-estimate", type=int, choices=[0, 1], default=1)
    
    args = parser.parse_args()

    # Forward all args at once
    runner = make_runner(**vars(args))
    runner.run()


if __name__ == "__main__":
    main()
