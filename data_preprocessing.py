import os
import pickle
import argparse
from glob import glob
import numpy as np
import pyemma as pe
from sklearn.preprocessing import StandardScaler
import mdtraj as md
from itertools import combinations



'''This script prepares input distance .npy arrays representing inter-residue inter-chain distances of APOE dimers and saves them into data/input subdirectory.
   The distances are scaled with StandardScaler which is saved into models/scaling subdirectory.'''


parser = argparse.ArgumentParser(description='Process system name.')
parser.add_argument('--system_name', metavar='system_name', type=str, nargs='?',
                    help='Name of the system')
args = parser.parse_args()

if args.system_name == "APOE3":
    system_name = "apoE3-dimer_goal"
elif args.system_name == "APOE4":
    system_name = "apoE4-dimer_goal"
elif args.system_name == "APOE3+TAU":
    system_name = "apoE3-dimer-tau_goal"
elif args.system_name == "APOE4+TAU":
    system_name = "apoE4-dimer-tau_goal"
elif args.system_name == "APOE3+SPA":
    system_name = "apoE3-dimer-3spa_goal"
elif args.system_name == "APOE4+SPA":
    system_name = "apoE4-dimer-3spa_goal"


"""Update the path to the "APOE_data" folder with MD trajectories accordingly."""
topology_file_path = f"../APOE_data/{system_name}/filtered_wrapped/filtered.pdb"
trajectories_paths = sorted(glob(f"../APOE_data/{system_name}/filtered_wrapped/*/output.wrapped.filtered.xtc"))
trajectory_names = [path.split('/')[-2] for path in trajectories_paths]
print(trajectory_names)


os.makedirs('data/input/{}'.format(args.system_name), exist_ok=True)

print("Creating featurizer for a given system.")
featurizer = pe.coordinates.featurizer(topology_file_path)
top = featurizer.topology

chain_A = [featurizer.select("residue {} and chainid 0 and mass > 1.008".format(res_num)) for res_num in range(22, 166)]
chain_B = [featurizer.select("residue {} and chainid 1 and mass > 1.008".format(res_num)) for res_num in range(22, 166)]

#add more chains if needed
combined_chains = chain_A + chain_B

print("Number of selected atoms in chain B: ", len(chain_A))
print("Number of selected atoms in chain B: ", len(chain_B))
print("Total number of residues: ", len(chain_A + chain_B))

chosen_pairs = []

for i in range(len(chain_A)):
    for j in range(len(chain_A), len(chain_A) + len(chain_B)):
        chosen_pairs.append((i, j))

print("Number of pairwise atomic distances/features : ", len(chosen_pairs))

featurizer.add_group_mindist(combined_chains, group_pairs=np.array(chosen_pairs))


training_source_all = pe.coordinates.source(trajectories_paths, featurizer)
input_all = np.concatenate(training_source_all.get_output())


scaler = StandardScaler().fit(input_all)

if not os.path.exists("models/scaling/"):
    os.makedirs("models/scaling/")
with open('models/scaling/{}-scaler.pkl'.format(args.system_name), 'wb') as file:
    pickle.dump(scaler, file)

for trajectory_path, trajectory_name in zip(trajectories_paths, trajectory_names):

    training_source = pe.coordinates.source(trajectory_path, featurizer)
    training_data_path = os.path.join("data/input/{}/{}.npy".format(args.system_name, trajectory_name))
    print("Computing features and saving at {}".format(training_data_path))
    input_flat = np.vstack(training_source.get_output())
    input_flat = scaler.transform(input_flat)
    np.save(training_data_path, input_flat)




