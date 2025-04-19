import os
import argparse
import random
import numpy as np
import mdtraj as md

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Extract representative PDB frames from MD trajectories.')
    parser.add_argument('--system_name', type=str, required=True, help='System name (e.g., APOE3, APOE4+SPA)')
    parser.add_argument('--rep', type=int, default=10, help='Number of representative frames to save as .pdb')
    return parser.parse_args()

def resolve_system_name(name):
    """Map user-friendly name to internal directory name."""
    name_map = {
        "APOE3": "apoE3-dimer_goal",
        "APOE4": "apoE4-dimer_goal",
        "APOE3+TAU": "apoE3-dimer-tau_goal",
        "APOE4+TAU": "apoE4-dimer-tau_goal",
        "APOE3+SPA": "apoE3-dimer-3spa_goal",
        "APOE4+SPA": "apoE4-dimer-3spa_goal",
    }
    return name_map.get(name)

def load_assignments(assignments_dir):
    """Load clustering assignments from .npy files."""
    assignments_dict = {}
    for filename in os.listdir(assignments_dir):
        if filename.endswith(".npy"):
            filepath = os.path.join(assignments_dir, filename)
            assignments = np.load(filepath)
            traj_name = filename[:-4]  # Strip .npy extension
            assignments_dict[traj_name] = assignments
    return assignments_dict

def collect_frames(assignments_dict, system_name):
    """Collect all frames belonging to each cluster/state."""
    selected_frames = {}
    # Gather unique labels
    all_labels = np.unique(np.concatenate(list(assignments_dict.values())))
    for label in all_labels:
        selected_frames[label] = []

    for traj_name, assignments in assignments_dict.items():
        traj_path = f"../APOE_data/{system_name}/filtered_wrapped/{traj_name}/output.wrapped.filtered.xtc"
        for label in selected_frames:
            indices = np.where(assignments == label)[0]
            selected_frames[label].extend([(traj_path, idx) for idx in indices])

    return selected_frames

def load_selected_frames(selected_frames, topology_file):
    """Load specific frames from trajectories."""
    frames = []
    for traj_path, frame_index in selected_frames:
        traj = md.load(traj_path, frame=frame_index, top=topology_file)
        frames.append(traj)
    return frames

def save_combined_pdb(rep_traj, output_dir, state_label):
    """Save multiple frames into one .pdb file for each state."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'state_{state_label}.pdb')
    rep_traj.save_pdb(output_path)

def main():
    args = parse_args()
    system_name = resolve_system_name(args.system_name)

    if system_name is None:
        raise ValueError(f"Unrecognized system name: {args.system_name}")

    topology_file = f"../APOE_data/{system_name}/filtered_wrapped/filtered.pdb"
    assignments_dir = f'output/projected/assignments/{args.system_name}'
    output_pdb_dir = f'output/representative_structures/{args.system_name}/'

    # Load and organize assignments
    assignments_dict = load_assignments(assignments_dir)
    selected_frames_dict = collect_frames(assignments_dict, system_name)

    # Sample and save frames for each state
    for label, frames in selected_frames_dict.items():
        if not frames:
            print(f"No frames found for label: {label}")
            continue

        sampled_frames = random.sample(frames, min(args.rep, len(frames)))
        rep_traj = md.join(load_selected_frames(sampled_frames, topology_file))
        save_combined_pdb(rep_traj, output_pdb_dir, label)

if __name__ == "__main__":
    main()
