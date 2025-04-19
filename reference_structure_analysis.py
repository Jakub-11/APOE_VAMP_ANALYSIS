import os
import argparse
import mdtraj as md
import numpy as np
from glob import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pyemma as pe
import pickle

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--system_name", type=str, required=True)
parser.add_argument("--distance_threshold", type=float, default=0.8, help='Distance threshold for monomer filtering')
parser.add_argument("--quality_threshold", type=float, default=0.6, help='RMSD threshold for classification')
args = parser.parse_args()

# --- System directory map ---
system_map = {
    "APOE3": "apoE3-dimer_goal",
    "APOE4": "apoE4-dimer_goal",
    "APOE3+TAU": "apoE3-dimer-tau_goal",
    "APOE4+TAU": "apoE4-dimer-tau_goal",
    "APOE3+SPA": "apoE3-dimer-3spa_goal",
    "APOE4+SPA": "apoE4-dimer-3spa_goal"
}

system_dir = system_map.get(args.system_name)
if not system_dir:
    raise ValueError(f"Unknown system name: {args.system_name}")

# --- Load trajectory paths and topology ---
traj_paths = sorted(glob(f"../APOE_data/{system_dir}/filtered_wrapped/*/output.wrapped.filtered.xtc"))
trajectory_names = [os.path.basename(os.path.dirname(p)) for p in traj_paths]
topology = f"../APOE_data/{system_dir}/filtered_wrapped/filtered.pdb"
print(f"Found {len(traj_paths)} trajectories.")

# --- Load and unscale distances for monomer filtering ---
with open(f'models/scaling/{args.system_name}-scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_data_paths = [f"data/input/{args.system_name}/{name}.npy" for name in trajectory_names]
trajs_scaled = [np.load(fpath) for fpath in input_data_paths]
trajs_unscaled = [scaler.inverse_transform(traj) for traj in trajs_scaled]
trajs_unscaled_full = np.concatenate(trajs_unscaled)

# --- Load reference structures ---
ref_dir = "RMSD references"
ref_paths = sorted(glob(f"{ref_dir}/*.pdb"))
reference_trajs = []
ref_names = []

for path in ref_paths:
    ref = md.load(path)
    backbone_indices = ref.topology.select('backbone')
    if len(backbone_indices) == 0:
        raise ValueError(f"No backbone atoms found in reference structure: {path}")
    reference_trajs.append(ref.atom_slice(backbone_indices))
    ref_name = os.path.splitext(os.path.basename(path))[0]
    ref_names.append(ref_name)

print(f"Loaded and sliced {len(reference_trajs)} reference structures.")

# --- Normalize names (remove "_flipped" suffix) ---
normalized_ref_names = [name.replace("_flipped", "") for name in ref_names]

# --- PyEMMA featurizer and load trajectories ---
feat = pe.coordinates.featurizer(topology)
trajs = [md.load(path, top=topology) for path in traj_paths]
trajs = [traj.atom_slice(feat.topology.select('backbone')) for traj in trajs]
all_frames = md.join(trajs)
print("Trajectories loaded and sliced to backbone.")

# --- Add RMSD features for each reference ---
for ref in reference_trajs:
    feat.add_minrmsd_to_ref(ref)
print("RMSD features added.")

# --- Transform frames to RMSD features ---
min_rmsds = feat.transform(all_frames)
min_rmsd_values = np.min(min_rmsds, axis=1)
assigned = np.argmin(min_rmsds, axis=1)
monomer_mask = np.all(trajs_unscaled_full > args.distance_threshold, axis=1)

# --- Final classification ---
final_classification = []
for i in range(len(assigned)):
    if monomer_mask[i]:
        final_classification.append("monomer")
    elif min_rmsd_values[i] > args.quality_threshold:
        final_classification.append("unclassified")
    else:
        final_classification.append(normalized_ref_names[assigned[i]])

# --- Count and merge classifications ---
classification_counts = Counter(final_classification)
total = sum(classification_counts.values())

# --- Print counts with percentages ---
print("\nAssignment counts (with percentages):")
for name, count in classification_counts.items():
    percent = 100 * count / total
    print(f"{name}: {count} ({percent:.1f}%)")

# --- Plot pie chart ---
plt.figure(figsize=(8, 8))
plt.pie(
    classification_counts.values(),
    labels=classification_counts.keys(),
    autopct='%1.1f%%',
    startangle=140
)
plt.title(f"Frame Assignments for {args.system_name}")
plt.tight_layout()
plt.savefig(f"{args.system_name}_assignments_pie.png")
plt.show()
