# First let's open the simulation file


import numpy as np
import xgboost
from matplotlib import pyplot as plt

from dataframes import SimData

show = False

data = SimData.from_pickle("data/Simulation.pkl")  # One row per track

# region Keep only the most muon-like track of each neutrino candidate
per_candidate_vars = np.column_stack(
    [
        data.closest_ray_track_dist,
        data.muon_ML_score,
        data.vert_x,
        data.vert_y,
        data.vert_z,
        data.energy,
    ]
)
# Get array that's True where per-candidate vars change, and add a True in front
new_candidate = np.r_[
    True, np.any(per_candidate_vars[1:] != per_candidate_vars[:-1], axis=1)
]
candidate = (
    np.cumsum(new_candidate) - 1
)  # Candidate index, shared by tracks within a candidate
data.n_tracks = np.bincount(candidate)[
    candidate
]  # Number of tracks in each row's candidate

# Sort tracks by candidate, then by muon log-likelihood (highest first)
order = np.lexsort((-data.particle_mu_LLH, candidate))
# Take the first track of each candidate
_, first_of_each = np.unique(candidate[order], return_index=True)
muon_tracks = order[first_of_each]
data = data.select(muon_tracks)  # One row per candidate, in file order
# endregion
# Drop entries where energy reconstruction failed
data = data.select(data.energy < 9000)
signal = data.labels()
if show:
    for name, values in data.items():
        plt.hist(values[signal], bins=200, label="Signal", weights=data.weight[signal])
        plt.hist(
            values[~signal], bins=200, label="Background", weights=data.weight[~signal]
        )
        plt.title(name)
        plt.xlabel(f"Simulated {name}")
        plt.ylabel("N")
        plt.legend()
        plt.show()

train_fraction = 0.9
train = np.random.random(len(data)) < train_fraction
train_data = data.select(train)
test_data = data.select(~train)
model = xgboost.XGBClassifier(
    eval_metric="auc",
    early_stopping_rounds=50,
    n_estimators=1000,
    learning_rate=0.3,
    max_depth=60,
)
model.fit(
    train_data.features(),
    train_data.labels(),
    eval_set=[(test_data.features(), test_data.labels())],
)
