import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import math
import pandas as pd
from pathlib import Path
import random
from scipy.stats import chi, kstest, probplot
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats


def pickle_loader(path):
    pickle_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
    # Dictionary to store loaded objects with filenames as keys (without .pkl)
    pickles = {}

    for file in pickle_files:
        file_path = os.path.join(path, file)

        # Extract name from filename (remove .pkl extension)
        obj_name = os.path.splitext(file)[0]

        # Load the pickle file
        with open(file_path, "rb") as f:
            pickles[obj_name] = pickle.load(f)
    return pickles

def parse_filename(file):
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
        "path": file.resolve()
    }

    return parsed_data


def JobAssigner(n,fault_list,amplitude_list,data_path,experiment_path,start_idx=0,ns=10,pc_n=1,randomness=False,exp_seed=1):
    random.seed(exp_seed)
    job_file_path = os.path.join(experiment_path,"jobs.csv") 
    # print(fault_no)
    if os.path.exists(job_file_path):
        jobs_df = pd.read_csv(job_file_path)
        print("Jobs file already exists. It is loaded successfully.")
        return jobs_df
    else:
        print("Jobs file is being created...")
    print(data_path)
    csv_files = list(Path(data_path).glob("*.csv"))
    data = [parse_filename(file) for file in csv_files]
    df = pd.DataFrame(data)
    # print(df)

    df["fault"] = df["fault"].astype(int)
    df["amplitude"] = df["amplitude"].astype(float)
    df["RunID"] = df["RunID"].astype(int)
    df = df.sort_values(by="RunID")
    
    # print(df)
    # Separate faulty and normal (amplitude=0) files
    conditions = list(zip(fault_list, amplitude_list))
    faulty_files = df[df.apply(lambda row: (row['fault'], row['amplitude']) in conditions, axis=1)]
    # print(faulty_files)
    faulty_files = faulty_files[:len(fault_list)*n]
    # print(len(faulty_files))
    ic_files = df[(df["amplitude"] == 0) & (~df["RunID"].isin(faulty_files["RunID"].unique()))]
    if randomness:
        ic_files = ic_files.sample(frac=1).reset_index(drop=True)
    # print(len(ic_files))
    # Ensure enough files exist
    for fault, amp in conditions:
        count = len(df[(df["fault"] == fault) & (df["amplitude"] == amp)])
        if count < n:
            raise ValueError(f"Not enough OC files for fault {fault} and amplitude {amp}. Found {count}, needed {n}.")
    if len(ic_files) < n*(ns-len(fault_list)):
        raise ValueError(f"Not enough IC files available.")

    # print(len(ic_files))
    assigned_jobs = []
    conditions = list(zip(fault_list, amplitude_list))
    used_faulty_indices = set()
    used_ic_indices = set()

    # Precompute grouped faulty files for quick access
    condition_groups = {
        cond: faulty_files[
            (faulty_files["fault"] == cond[0]) & (faulty_files["amplitude"] == cond[1])
        ].reset_index(drop=True)
        for cond in conditions
    }

    # Track indexes used for each condition separately
    condition_cursors = {cond: 0 for cond in conditions}
    ic_cursor = 0

    for i in range(n):
        job_faulty_files = []

        for cond in conditions:
            group = condition_groups[cond]
            cursor = condition_cursors[cond]

            if cursor >= len(group):
                raise ValueError(f"Not enough faulty files for fault {cond[0]} and amplitude {cond[1]}")

            file_path = group.iloc[cursor]["path"]
            job_faulty_files.append(file_path)

            condition_cursors[cond] += 1  # Advance cursor for this condition

        # Fill with IC files
        ic_needed = ns - len(job_faulty_files)
        if ic_cursor + ic_needed > len(ic_files):
            raise ValueError("Not enough IC files to complete all jobs without reuse.")

        ic_sample = ic_files.iloc[ic_cursor:ic_cursor + ic_needed]
        ic_file_paths = ic_sample["path"].tolist()
        ic_cursor += ic_needed  # Move IC cursor forward

        job_files = job_faulty_files + ic_file_paths
        job_id = i + 1
        assigned_pc = random.choice(range(1, pc_n + 1))

        assigned_jobs.append((job_id, assigned_pc, job_files))


    max_files = max(len(files) for _, _, files in assigned_jobs) 
    column_names = ["OC_File"] + [f"IC_File_{i}" for i in range(1, max_files)]
    

    jobs_df = pd.DataFrame([
        [job[0], job[1]] + job[2] + [None] * (max_files - len(job[2])) 
        for job in assigned_jobs
    ], columns=["JobID", "AssignedPC"] + column_names)

    # Save jobs file
    jobs_df.to_csv(job_file_path, index=False)

    return jobs_df

def distance_to_hypersphere(data,d,r):
    '''
    True distance of noisy points to hypersphere
    Input:
    data - data points
    d - dimension of the hypersphere + 1
    r - radius of the hypersphere
    Output:
    true distance of noisy points to hypersphere
    '''
    projected_norm = np.linalg.norm(data[:,:d], axis=1)
    orthogonal_norm = np.linalg.norm(data[:,d:], axis=1)
    distance_to_projected_hypersphere = abs(projected_norm-r)
    final_distance = np.sqrt(orthogonal_norm**2+distance_to_projected_hypersphere**2)
    return final_distance.reshape(-1,1)


def generate_stochastic_process_on_sphere(n, sigma=0.3, d=3, r=1, seed=None, set_seed=False):
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




def threeD_plot_on_sphere(data,figure_path,n_to_plot=100,save=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')  
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgrey', alpha=0.3, linewidth=0)
    ax.plot(data[-n_to_plot:, 0], data[-n_to_plot:, 1], data[-n_to_plot:, 2],
            color='blue', marker='o', markersize=1)

    # Set limits and aspect ratio
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    # Save with transparent background
    if save is True:
        plt.savefig(figure_path, dpi=300, transparent=True)
    plt.show() 
    plt.close(fig)

def KS_test_QQ_plot(data,figure_path,sigma,df,plot_title="Q-Q Plot vs. Scaled Chi Distribution"):
    df = 1              
    scale =sigma
    data = np.array(data)
    scaled_data = data / scale

    # Q-Q plot against Chi(df)
    fig, ax = plt.subplots()
    probplot(scaled_data, dist=chi, sparams=(df,), plot=ax)
    ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.show()
    plt.close(fig)
    
    # K-S Test
    D, p_value = kstest(scaled_data, 'chi', args=(df,))
    print(f"K-S test statistic: {D:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Reject null: data does NOT follow scaled Chi distribution")
    else:
        print("Cannot reject null: data MAY follow scaled Chi distribution")

def estimated_ACF_plot(data,figure_path,fig_title):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(data, lags=10, alpha=0.05, ax=ax)
    plt.xlabel('Lag')
    plt.ylabel('Estimated Autocorrelation')
    plt.title(fig_title)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(figure_path, dpi=300)
    plt.show()
    plt.close(fig)


def estimated_ACF_plot_by_axis(data,figure_path,lags=20):
    acf_values_phase2 = [acf(data[:, i], nlags=lags, fft=True) for i in range(3)]

    # Update dimension labels to 'X axis', 'Y axis', 'Z axis'
    axis_labels = ['X axis', 'Y axis', 'Z axis']

    # Plot ACFs using subplots with updated axis labels
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i in range(3):
        axs[i].stem(range(lags + 1), acf_values_phase2[i])
        axs[i].axhline(0, color='black', linewidth=0.8)
        axs[i].set_title(f'{axis_labels[i]}')
        axs[i].set_xlabel('Lag')
        if i == 0:
            axs[i].set_ylabel('Autocorrelation')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(figure_path, dpi=300)
    plt.show()
    plt.close(fig)

def deviation_plots(deviations,figure_path,figure_title=None,start_index=1000,red_line=None):
    time_steps = np.arange(start_index, start_index + len(deviations))
    y_min = min(min(deviations), min(deviations))
    y_max = max(max(deviations), max(deviations))
    # Plot estimated distances
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, deviations, color='blue')
    if red_line:
        plt.axvline(
            x=start_index + red_line,
            color='red',
            linestyle='--',
            linewidth=2.5 
        )
    if figure_title:
        plt.title(figure_title)
    plt.ylabel('Deviations')
    plt.ylim(y_min, y_max)
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)

def test_normality_plot(data, figure_path):
    """
    Apply normality test to each dimension of the data and plot the p-values.
    Parameters:
    - data: np.ndarray of shape (m, d)
    """

    n_dims = data.shape[1]
    p_values = []

    for i in range(n_dims):
        x = data[:, i]
        _ , p = stats.shapiro(x)
        p_values.append(p)

    plt.figure(figsize=(10, 4))
    plt.bar(range(1,n_dims+1), p_values, color='skyblue')
    plt.axhline(0.05, color='red', linestyle='--', label='p = 0.05')
    plt.xlabel('Dimension index')
    plt.ylabel('p-value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300) 

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