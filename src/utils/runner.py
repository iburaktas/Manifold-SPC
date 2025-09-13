import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from typing import Iterable, Callable, Any, Optional, Literal, Dict, List,  Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from src.utils.ControlChart.control_chart import DFEWMA
from src.utils.ControlChart.control_chart import DFUC
from src.utils.functions import *
import src.models as models
from src.utils.Filters import ARFilter
from src.utils.ExperimentConfig import ExperimentConfig

# for name in ["MF", "LPP", "NPE", "PCA"]:
    # cls = getattr(models, name)
    # print(name, "->", cls) 




class Runner:
    def __init__(
        self,
        dataloader: Iterable[Any],
        cfg: ExperimentConfig = ExperimentConfig(),
        n_workers: int = None,
        write_details: bool = False
        ):
        self.dataloader = dataloader
        self.cfg = cfg
        self.n_workers= n_workers
        self.write_details = write_details

        if cfg.pickle_path is not None:
            self.pickles = pickle_loader(cfg.pickle_path)
        else: 
            self.pickles = None

        self.project_root = Path(__file__).resolve().parent.parent.parent

        self.dirs = {}
        for method in self.cfg.methods:
            self.dirs[method] = os.path.join(self.project_root,"experiments",self.cfg.exp_name,method)

        [os.makedirs(rl_dir, exist_ok=True) for rl_dir in self.dirs.values()]

    def compute(self,data):
        m0,m1,m2 = self.cfg.n_manifold,self.cfg.n_ar if self.cfg.n_ar is not None else 0, self.cfg.n_cc
        n_oc = data.shape[0] - m0-m1-m2

        if self.pickles is not None:
            scales = self.pickles['scales']
            keep_mask = self.pickles['keep_mask']
        else:
            scales = 1
            keep_mask = np.ones(data.shape[1], dtype=bool)

        data = data/scales
        data = data[:,keep_mask]

        manifold_data = data[:m0]
        ar_data = data[m0:m0+m1]
        cc_data = data[m0+m1:m0+m1+m2]
        oc_data = data[-n_oc:]

        # print(manifold_data.shape)
        # print(ar_data.shape)
        # print(cc_data.shape)
        # print(oc_data.shape)
        models_dict={}

        for model_name in self.cfg.methods:
            model_cls = getattr(models, model_name)
            if model_name == 'MF':
                c0,c1,c2 = self.cfg.mf_constants
                model = model_cls(manifold_data, c0=c0, c1=c1,c2=c2,d=self.cfg.d,estimate_sig=self.cfg.estimate_sig,sigma = self.cfg.sigma)
            elif model_name == 'LPP':
                model = model_cls(n_components=self.cfg.ndim,n_neighbors=self.cfg.n_neighbors, weight=self.cfg.LPP_weight)
                model.fit(manifold_data)
            elif model_name == 'NPE':
                model = model_cls(n_components=self.cfg.ndim,n_neighbors=self.cfg.n_neighbors)
                model.fit(manifold_data)
            elif model_name == 'PCA':
                model = model_cls(n_components=self.cfg.ndim)
                model.fit(manifold_data)
            models_dict[model_name] = model
        
        ar_models_dict ={}
        
        
        for model_name in self.cfg.methods:
            manifold_model = models_dict[model_name]
            if model_name == "MF":
                transformed_data = manifold_model.deviations(ar_data) if m1 > 0 else np.empty((0, 1))
                model = ARFilter(transformed_data,max_p=self.cfg.p)
            else:
                transformed_data = manifold_model.transform(ar_data) if m1 > 0 else np.empty((0, manifold_model.n_components))
                model = ARFilter(transformed_data,max_p=self.cfg.p)
            ar_models_dict[model_name] = model

        #return ar_models_dict, cc_data, models_dict

        cc_dict= {}

        for model_name in self.cfg.methods:
            manifold_model = models_dict[model_name]
            ar_model = ar_models_dict[model_name]
            if model_name == "MF":
                cc_data_ = ar_model.predict_errors(manifold_model.deviations(cc_data))[ar_model.max_p:]
                cc = DFUC(IC_data=cc_data_,alpha=self.cfg.alpha)
            else:
                cc_data_ = ar_model.predict_errors(manifold_model.transform(cc_data))[ar_model.max_p:]
                cc = DFEWMA(IC_data=cc_data_,alpha=self.cfg.alpha)
            cc_dict[model_name] = cc

        #return cc_dict
        Flag = True
        iter = 0
        
        while Flag and iter < n_oc:
            oc_data_ = oc_data[iter].reshape(1,-1)
            for model_name in self.cfg.methods:
                manifold_model = models_dict[model_name]
                ar_model = ar_models_dict[model_name]
                cc = cc_dict[model_name]
                if cc.flag is True:
                    if model_name == "MF":
                        cc.iterate(ar_model.predict_errors(manifold_model.deviations(oc_data_).reshape(1,-1)))
                    else:
                        cc.iterate(ar_model.predict_errors(manifold_model.transform(oc_data_).reshape(1,-1)))
            Flag = any(cc.flag for cc in cc_dict.values())
            iter += 1
            #print(iter)
        return cc_dict

    def writer_(self,idx:int,res: dict[Union[DFEWMA, DFUC]]):
        for method in self.cfg.methods:
            directory = self.dirs[method]
            cc = res[method]
            rl = cc.runlength
            out_path = os.path.join(directory,f"ID-{idx}-rl-{rl}.csv")
            if self.write_details:
                arr = np.column_stack([cc.climit[:rl], cc.Tval[:rl]])
                np.savetxt(out_path, arr, delimiter=",", header="climit,Tvals", comments='')
            else:
                np.savetxt(out_path, np.atleast_1d(rl), delimiter=",")

    def process(self,idx,data):
        res = self.compute(data=data)
        self.writer_(idx,res)
        return

    def run(self):
        if self.n_workers in (None, 1):
            for idx, data in enumerate(self.dataloader):
                results = self.compute(data=data)
                self.writer_(idx=idx,res=results)
        else:
            max_workers = self.n_workers or max(1, (os.cpu_count() or 1))
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            # IMPORTANT on Windows: call this under if __name__ == "__main__"
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(self.process, idx, data)
                        for idx, data in enumerate(self.dataloader)]
                
                for fut in as_completed(futures):
                    _ = fut.result()
        self.rl_results(Path(os.path.join(self.project_root,"experiments",self.cfg.exp_name)))
            
    @staticmethod
    def rl_results(root: Path) -> None:
        results = []
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            rl_values = []
            for csv_file in folder.glob("*.csv"):
                try:
                    parts = csv_file.stem.split("-")
                    if "rl" in parts:
                        idx = parts.index("rl")
                        rl_val = float(parts[idx + 1])
                        rl_values.append(rl_val)
                except Exception:
                    continue    
            if rl_values:
                results.append({
                "folder": folder.name,
                "mean_rl": float(pd.Series(rl_values).mean()),
                "std_rl": float(pd.Series(rl_values).std()),
                "count": len(rl_values),
            })
        df = pd.DataFrame(results)
        out_path = root / "rl_summary.csv"
        df.to_csv(out_path, index=False)
        return
                    




