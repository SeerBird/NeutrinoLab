# First let's open the simulation file
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
import pickle
import xgboost

show = False


@dataclass
class RealData:
    closest_ray_track_dist: np.ndarray
    lepton_track_length: np.ndarray
    longest_track_dist: np.ndarray
    muon_ML_score: np.ndarray
    end_z: np.ndarray
    end_y: np.ndarray
    end_x: np.ndarray
    track_ML_score: np.ndarray
    particle_mu_LLH: np.ndarray
    start_z: np.ndarray
    start_y: np.ndarray
    start_x: np.ndarray
    vert_x: np.ndarray
    vert_y: np.ndarray
    vert_z: np.ndarray
    energy: np.ndarray
    subevent: np.ndarray
    length_momentum: np.ndarray
    scat_momentum: np.ndarray

    def array(self):
        result = []
        for _attribute in dir(self):
            _dataset = self.__getattribute__(_attribute)
            if isinstance(_dataset, np.ndarray):
                result.append(_dataset)
        return np.asarray(result)

    def size(self):
        return len(self.energy)

    def select(self,selection):
        return self.__class__(*self.array()[:, selection])



@dataclass
class SimData(RealData):
    category: np.ndarray
    weight: np.ndarray
    neutrino_energy: np.ndarray
    neutrino_flight_dist: np.ndarray
    muon_momentum: np.ndarray




with (open("data/Simulation.pkl", 'rb') as infile):
    data = SimData(*np.asarray(pickle.load(infile).values.T, dtype=float))
    subevent, category = np.asarray(data.subevent, dtype=int), np.asarray(data.category, dtype=int)



'''
# We can plot, for example, the simulated energy for the neutrino
cosmic_events = category == 4
a = data.muon_momentum[cosmic_events]
e = data.energy[cosmic_events]
a = a[e < 9000]
print(f"Fraction of non-muon cosmic events:{
np.count_nonzero(np.isnan(a))
/ len(a)}")
'''
data = data.select(data.energy < 9000)
signal = data.category == 21
if show:
    for attribute in dir(data):
        _set = data.__getattribute__(attribute)
        if isinstance(_set, np.ndarray):
            plt.hist(_set[signal], bins=200, density=False, label="Signal",
                     weights=data.weight[signal])
            plt.hist(_set[~signal], bins=200, density=False, label="Background",
                     weights=data.weight[~signal])
            plt.title(attribute)
            plt.xlabel(f'Simulated {attribute}')
            plt.legend()
            plt.ylabel('N')
            plt.show()

train_fraction = 0.9
train = np.random.random(data.size())<train_fraction
train_sig = data.select(signal&train)
train_bkg = data.select((~signal)&train)
test_sig = data.select(signal&(~train))
test_bkg = data.select((~signal)&(~train))
model = xgboost.XGBClassifier(eval_metric='auc', early_stopping_rounds=50, n_estimators=1000,
                              learning_rate=0.3, max_depth=60)
model.fit(train_sig, train_bkg, eval_set=[(test_sig,test_bkg)])