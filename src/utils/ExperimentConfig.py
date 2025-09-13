import sys
from pathlib import Path
from typing import Iterable, Callable, Any, Optional, Literal, Dict, List,  Union

class ExperimentConfig:
    exp_name:str = 'Test'
    n_manifold: int = 700
    n_ar : Optional[int] = 400
    n_cc : int = 100
    methods : List[str] = ["MF","LPP","NPE","PCA"]
    alpha: Optional[float] = 0.05
    ndim : Optional[int] = 1
    n_neighbors : Optional[int] = 15
    LPP_weight : Optional[str] = "heat"
    pickle_path : Optional[Path] = None
    sigma: Optional[float] = 0.05
    estimate_sig: Optional[bool] = True
    p: Optional[int] = 20
    mf_constants: Optional[List[float]] = [3.0,2.0,4.0]
    d: Optional[int] = None