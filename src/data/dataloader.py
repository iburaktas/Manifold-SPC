# Loads Data for TennesseeEastman and KolektorSDD Experiments, also generates data for a stochastic process on a shpere

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Literal
import numpy as np
from pathlib import Path
import os
import random
import pandas as pd
from PIL import Image
import warnings

@dataclass
class DataLoader_:
    """
    Parameters
    ----------
    exp_type: Type of the experiment # Default None
    N: Optional[int] = 0 # Number of simulations
    amplitude: Optional[float] = 0.05
    fault_no: Optional[int] = 4
    exp_seed: Optional[int] = 1
    iter: Optional[int] = 0
    exp_seed_start: Optional[int] = 1

    """
    exp_type: Optional[Literal["TE", "KolektorSDD", "SyntheticProcess"]] = None
    N: Optional[int] = 0
    amplitude: Optional[float] = 0.05
    fault_no: Optional[int] = 4
    exp_seed: Optional[int] = 1
    iter: Optional[int] = 0
    exp_seed_start: Optional[int] = 1

    def __post_init__(self) -> None:
        self.file_path = self.__resolve_path__()


        if self.exp_type == "KolektorSDD":
            self.data_path = os.path.join(self.file_path,"KolektorSDD\Data")
        elif self.exp_type == "TE":
            self.data_path = os.path.join(self.file_path,"TennesseeEastmanProcess\Data")
        else:
            self.data_path = None

        if self.exp_type == "TE":
            self.paths_list = self.__path_maker_TE__()
        elif self.exp_type == "KolektorSDD":
            self.paths_list = self.__path_maker_KolektorSSD__()
        else:
            self.paths_list = None

        self.len = self.N if self.paths_list is None else len(self.paths_list)
        self.seed = self.exp_seed_start
        
    def __resolve_path__(self) -> Path:
        try:
            return Path(__file__).parent.resolve()
        except NameError:
            return Path.cwd().resolve()

    
    @staticmethod
    def __parse_filename_TE__(file):
        filename = file.stem  
        parts = filename.split('-')  # Split by '-'
        # Extracting information from filename
        parsed_data = {
            "RunID": parts[1],
            "fseed": parts[3],
            "t": parts[5],
            "ns": parts[7],
            "fault": parts[9],
            "t0": parts[11],
            "t1": parts[13],
            "amplitude": parts[15],
            "path": file.absolute() 
        }

        return parsed_data
    
    def __path_maker_TE__(self, randomness : bool=True,
                                    ns:int=10,
                                    start_idx:int=0) -> List[Path]:
        random.seed(self.exp_seed)
        csv_files = list(Path(self.data_path).glob("*.csv"))
        data = [self.__parse_filename_TE__(file) for file in csv_files]
        df = pd.DataFrame(data)


        df["fault"] = df["fault"].astype(int)
        df["amplitude"] = df["amplitude"].astype(float)
        df["RunID"] = df["RunID"].astype(int)
        df = df.sort_values(by="RunID")

        faulty_files = df[(df["fault"] == self.fault_no) & (df["amplitude"] == self.amplitude)]
        faulty_files = faulty_files[:self.N]
        ic_files = df[(df["amplitude"] == 0) & (~df["RunID"].isin(faulty_files["RunID"].unique()))]

        if randomness:
            ic_files = ic_files.sample(frac=1).reset_index(drop=True)

        if len(faulty_files) < self.N:
            raise ValueError(f"Not enough OC files available for TE Experiment")
        if len(ic_files) < self.N*(ns-1):
            raise ValueError(f"Not enough IC files available.")
        
        assigned_list = []

        for i in range(self.N):
            oc = faulty_files.iloc[i]["path"]
            ic = list(
                ic_files.iloc[start_idx + i*(ns-1) : start_idx + (i+1)*(ns-1)]["path"]
            )
            assigned_list.append([oc] + ic)

        return assigned_list
    
    def __read_images__(self,img_path: str | Path, 
                        size: tuple[int, int] = (1408, 512)) -> np.ndarray:
        
        img = Image.open(img_path).convert("L")
        img = img.transpose(Image.TRANSPOSE)
        img = img.resize(size)
        img_array = np.array(img, dtype=np.float32).flatten() / 255.0
        return img_array

    def __path_maker_KolektorSSD__(self) -> List[Path]:
        ic_paths = []
        oc_paths = []


        for folder in sorted(os.listdir(self.data_path)):
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue

            for i in range(8):
                img_filename = f"Part{i}.jpg"
                mask_filename = f"Part{i}_label.bmp"

                img_path = os.path.join(folder_path, img_filename)
                mask_path = os.path.join(folder_path, mask_filename)


                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    continue

                mask_array = self.__read_images__(mask_path)
                (oc_paths if np.any(mask_array > 0) else ic_paths).append(img_path)

        return [ic_paths + oc_paths]
    

    def __SyntheticProcess_generator__(self,
                                        n :int = 1500,
                                        d : int = 3,
                                        D : int= 6,
                                        sigma : float = 0.1,
                                        fault_start: int = 1200 )-> np.ndarray:
        np.random.seed(self.seed)
        padding = np.zeros((n, D - d))

        # Generate data
        data = self.__generate_stochastic_process_on_sphere__(n=n,d=d)
        data = np.hstack((data, padding))

        data_noisy = data + np.random.normal(scale = sigma, size=(n,D))
        data_noisy[fault_start:,self.fault_no] = data_noisy[fault_start:,self.fault_no]+self.amplitude*sigma
        self.seed += 1

        return data_noisy

    def __generate_stochastic_process_on_sphere__(self,n:int,
                                                  d:int,
                                                  sigma:float=0.3,
                                                  r:float = 1.0,
                                                  seed:int = None,
                                                  set_seed:bool =False)-> np.ndarray:
        '''
        Generates a stochastic process on d - 1 dimensional sphere
        Input:
        n       - number of points to be generated
        sigma   - sigma_x given in the paper
        d       - dimension of the hypersphere + 1
        r       - radius of the hypersphere
        Output:
        data    - np.array in shape of (n,d)
        '''
    
        if seed is not None and set_seed is True:
            np.random.seed(seed)

        X_0 = np.random.normal(scale=sigma, size=d)
        X_0 = X_0/np.linalg.norm(X_0)
        data = np.zeros((n,d))

        for i in range(n):
            gaussian_error = np.random.normal(scale=sigma, size=d)
            base = X_0 if i == 0 else data[i - 1]
            perturbed = base + gaussian_error
            data[i] = r * perturbed / np.linalg.norm(perturbed)
        return data
    
    def get_item(self,idx)-> np.ndarray:
        if self.exp_type == "TE":
            files = self.paths_list[idx]
            mats = [pd.read_csv(p, header=None).to_numpy()[:,1:] for p in files]
            return np.hstack(mats).astype(np.float32)
        elif self.exp_type == "KolektorSDD":
            mats = [self.__read_images__(img_path) for img_path in self.paths_list[0]]
            return np.vstack(mats).astype(np.float32)
        elif self.exp_type == "SyntheticProcess":
            return self.__SyntheticProcess_generator__()
        else:
            return None
    
    def path_maker(self)-> List[Path]:
        return None
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        for i in range(len(self)):
            yield self.get_item(i)
        
    #def next(self):
    #    res = self.get_item(self.iter)
    #    self.iter += 1
    #
    #    if self.iter == self.len:
    #        warnings.warn(
    #            "There are no further iterations, the DataLoader will reset.",
    #            UserWarning
    #        )
    #        self.iter = 0
    #    return res
