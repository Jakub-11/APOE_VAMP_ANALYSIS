import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Filter and cluster projected data.')
parser.add_argument('--system_name', type=str, required=True, help='Name of the system')
parser.add_argument('--threshold', type=float, default=0.8, help='Distance threshold for filtering')
parser.add_argument('--n_dimensions', type=int, default=5, help='Number of VAMP dimensions to use')
args = parser.parse_args()

n_clusters = args.n_dimensions + 1
print(f"Clustering in {args.n_dimensions} VAMP dimensions into {n_clusters} states.")

# Load scaler
with open(f'models/scaling/{args.system_name}-scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load projected trajectories
projection_dir = f'output/projected/vamp/{args.system_name}/'
projection_files = sorted(glob(os.path.join(projection_dir, '*.npy')))
projected_data = [np.load(pf) for pf in projection_files]
trajectory_names = [os.path.basename(pf)[:-4] for pf in projection_files]

# Load original distances and inverse scale them
input_data_paths = [f"data/input/{args.system_name}/{name}.npy" for name in trajectory_names]
trajs_scaled = [np.load(fpath) for fpath in input_data_paths]
trajs_unscaled = [scaler.inverse_transform(traj) for traj in trajs_scaled]

# Load the VAMP model (not actually used here, but keeping consistent with earlier workflow)
with open(f'models/vamp/{args.system_name}_vamp.pkl', 'rb') as f:
    vamp_estimator = pickle.load(f)

# Filter out monomeric frames (all distances > threshold)
filtered_projections = []
trajectory_indices = []
n_total = 0
n_monomeric = 0

for traj_idx, (proj, unscaled) in enumerate(zip(projected_data, trajs_unscaled)):
    n_total += len(unscaled)
    monomer_mask = np.all(unscaled > args.threshold, axis=1)
    n_monomeric += np.sum(monomer_mask)
    valid_proj = proj[~monomer_mask]
    filtered_projections.append(valid_proj)
    trajectory_indices.extend([traj_idx] * valid_proj.shape[0])

# Compute and report monomeric fraction
fraction_monomeric = n_monomeric / n_total if n_total > 0 else 0
print(f"Fraction of monomeric frames (all distances > {args.threshold} Å): {fraction_monomeric:.4f} ({fraction_monomeric*100:.2f}%)")

# Prepare data for clustering
all_filtered = np.concatenate(filtered_projections, axis=0)
print(f"Number of frames after filtering: {all_filtered.shape[0]}")

data_to_cluster = all_filtered[:, :args.n_dimensions]
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(data_to_cluster)


# Save per-trajectory assignments with -1 for filtered-out frames
assignment_dir = f'output/projected/assignments/{args.system_name}/'
os.makedirs(assignment_dir, exist_ok=True)

cluster_pointer = 0
for traj_idx, (proj, unscaled, traj_name) in enumerate(zip(projected_data, trajs_unscaled, trajectory_names)):
    monomer_mask = np.all(unscaled > args.threshold, axis=1)
    assignments = np.full(len(unscaled), -1, dtype=int)
    n_valid = np.sum(~monomer_mask)
    assignments[~monomer_mask] = labels[cluster_pointer:cluster_pointer + n_valid]
    cluster_pointer += n_valid

    # Save to individual .npy file
    assignment_path = os.path.join(assignment_dir, f'{traj_name}.npy')
    np.save(assignment_path, assignments)

print(f"Saved individual trajectory assignments to: {assignment_dir}")


# Visualization in 2D
print("Plotting clustering in first two VAMP dimensions...")

# Cluster percentages relative to total (including filtered-out)
cluster_sizes = {i: np.sum(labels == i) for i in range(n_clusters)}
cluster_percentages = {
    i: (cluster_sizes[i] / n_total) * 100 for i in range(n_clusters)
}
filtered_out_percentage = (n_monomeric / n_total) * 100

# Updated label names
label_names = [
    f"Cluster {i} ({cluster_percentages[i]:.1f}%)" for i in range(n_clusters)
]


# Create color palette and label mapping
palette = sns.color_palette('tab10', n_clusters)
label_map = {i: label_names[i] for i in range(n_clusters)}

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=all_filtered[:, 0],
    y=all_filtered[:, 1],
    hue=[label_map[l] for l in labels],
    palette=palette,
    s=10,
    linewidth=0,
    alpha=0.7
)

plt.xlabel('VAMP Dimension 1')
plt.ylabel('VAMP Dimension 2')
plt.title(
    f'Clustering in VAMP space (d={args.n_dimensions}, k={n_clusters})\n'
    f'Filtered-out frames: {filtered_out_percentage:.1f}%'
)

# Place legend outside
plt.legend(title='Cluster', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(rect=[0, 0, 0.85, 1])  # make room for legend

plot_path = f'output/cluster_plot_{args.system_name}_d{args.n_dimensions}_k{n_clusters}.png'
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Saved cluster plot to {plot_path}")
