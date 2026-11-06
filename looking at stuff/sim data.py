# First let's open the simulation file
import pickle
from dataclasses import dataclass, field, fields

import numpy as np
import xgboost
from matplotlib import pyplot as plt

show = False


def column(_name, dtype=float):
    """Dataclass field bound to a named column of the pickled DataFrame"""
    return field(metadata={"column": _name, "dtype": dtype})


@dataclass(eq=False)
class RealData:
    """Class representing a list of events, with their relevant variables"""

    closest_ray_track_dist: np.ndarray = column("Distance to closest cosmic ray track")
    lepton_track_length: np.ndarray = column("Length of lepton track")
    longest_track_dist: np.ndarray = column("Distance to longest track")
    muon_ML_score: np.ndarray = column("Topological muon ML score")
    track_ML_score: np.ndarray = column("Track vs shower ML score")
    particle_mu_LLH: np.ndarray = column("Particle ID LLh")
    start_x: np.ndarray = column("Start position of lepton track x component")
    start_y: np.ndarray = column("Start position of lepton track y component")
    start_z: np.ndarray = column("Start position of lepton track z component")
    end_x: np.ndarray = column("End position of lepton track x component")
    end_y: np.ndarray = column("End position of lepton track y component")
    end_z: np.ndarray = column("End position of lepton track z component")
    vert_x: np.ndarray = column("Interaction vertex position x component")
    vert_y: np.ndarray = column("Interaction vertex position y component")
    vert_z: np.ndarray = column("Interaction vertex position z component")
    energy: np.ndarray = column("Total energy")
    subevent: np.ndarray = column("Subevent", dtype=int)
    length_momentum: np.ndarray = column("Muon momentum (length estimate)")
    scat_momentum: np.ndarray = column("Muon momentum (scattering estimate)")

    @classmethod
    def from_frame(cls, frame):
        """Build from a DataFrame, matching columns by name (column order doesn't matter)"""
        return cls(**{f.name: frame[f.metadata["column"]].to_numpy(dtype=f.metadata["dtype"])
                      for f in fields(cls)})

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as infile:
            return cls.from_frame(pickle.load(infile))

    def __len__(self):
        return len(self.energy)

    def items(self):
        """(name, array) pairs for every field"""
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def select(self, selection):
        """New instance with the boolean mask / index array applied to every field"""
        return self.__class__(**{name: values[selection] for name, values in self.items()})

    @staticmethod
    def feature_names():
        """Fields usable as fit inputs: those present in real data"""
        return [f.name for f in fields(RealData)]

    def features(self):
        """(n_events, n_features) array of the real-data variables, ready for a fit"""
        return np.column_stack([getattr(self, name) for name in self.feature_names()])


@dataclass(eq=False)
class SimData(RealData):
    category: np.ndarray = column("Simulation category", dtype=int)
    weight: np.ndarray = column("Simulation weight")
    neutrino_energy: np.ndarray = column("Simulated neutrino energy")
    neutrino_flight_dist: np.ndarray = column("Simulated neutrino flight distance")
    muon_momentum: np.ndarray = column("Simulated muon momentum")

    def labels(self):
        return self.category == 21


data = SimData.from_pickle("data/Simulation.pkl")
data = data.select(data.energy < 9000)
signal = data.labels()
if show:
    for name, values in data.items():
        plt.hist(values[signal], bins=200, label="Signal", weights=data.weight[signal])
        plt.hist(values[~signal], bins=200, label="Background", weights=data.weight[~signal])
        plt.title(name)
        plt.xlabel(f"Simulated {name}")
        plt.ylabel("N")
        plt.legend()
        plt.show()

train_fraction = 0.9
train = np.random.random(len(data)) < train_fraction
train_data = data.select(train)
test_data = data.select(~train)
model = xgboost.XGBClassifier(eval_metric="auc", early_stopping_rounds=50, n_estimators=1000,
                              learning_rate=0.3, max_depth=60)
model.fit(train_data.features(),
          train_data.labels(),
          eval_set=[(test_data.features(), test_data.labels())])
