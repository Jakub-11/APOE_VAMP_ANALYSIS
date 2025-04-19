import os
import pickle
import argparse
import numpy as np
from glob import glob
from deeptime.util.data import TrajectoriesDataset
from deeptime.decomposition import VAMP, TICA

parser = argparse.ArgumentParser(description='Process system name.')
parser.add_argument('--system_name', metavar='system_name', type=str, nargs='?',
                    help='Name of the system')
args = parser.parse_args()

print("ANALYZING SYSTEM: ", args.system_name)

#Creating directories for storing results
os.makedirs('models/vamp/', exist_ok=True)
os.makedirs('output/projected/vamp/{}'.format(args.system_name), exist_ok=True)

timestep = 0.1 #timestep in the simulation expressed in nanoseconds
lagstep = 150

print("Using {}ns lagtime".format(timestep*lagstep))


# loading trajectories saved in separate files
training_data_paths = glob("data/input/{}/*.npy".format(args.system_name))
trajs = [np.load(file) for file in training_data_paths]
traj_names = [os.path.basename(x)[:-4] for x in training_data_paths]

print("Number of trajectories: ", len(trajs))
lengths =[x.shape[0] for x in trajs]
print("Number of frames: ", sum(lengths))
print("Number of timelagged pairs: ", sum(lengths) - len(trajs) * lagstep, " *")
print("*approximate number, can differ slightly in some edge cases involving trajectories shorter than the used lagstep.")


# VAMP estimation
vamp_path = os.path.join("models/vamp/",  "{}_vamp.pkl".format(args.system_name))
print("VAMP dimensionality reduction is estimated.")
vamp = VAMP(lagtime=lagstep, dim=10).fit(trajs)
with open(vamp_path, 'wb') as file:
    pickle.dump(vamp, file)
vamp_model = vamp.fetch_model()
vamp_trajectories_path = "output/projected/vamp/{}/".format(args.system_name)
for name, traj in zip(traj_names, trajs):
    with open(vamp_trajectories_path + name + '.npy', 'wb') as f:
        np.save(f, vamp_model.transform(traj))
            


