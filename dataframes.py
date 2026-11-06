import pickle
from dataclasses import dataclass, field, fields

import numpy as np


def column(_name, dtype=float):
    """Dataclass field bound to a named column of the pickled DataFrame"""
    return field(metadata={"column": _name, "dtype": dtype})


def derived():
    """Dataclass field not read from the file: starts as None and is computed after loading"""
    return field(default=None, metadata={"column": None})


@dataclass(eq=False, kw_only=True)
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
    # "Subevent" (track index within a candidate) is deliberately not loaded: its numbering is
    # unreliable in the EXT part of the simulation, and candidates are separated by other fields.
    length_momentum: np.ndarray = column("Muon momentum (length estimate)")
    scat_momentum: np.ndarray = column("Muon momentum (scattering estimate)")
    n_tracks: np.ndarray | None = (
        derived()
    )  # Number of tracks in the candidate this row came from

    @classmethod
    def from_frame(cls, frame):
        """Build from a DataFrame, matching columns by name (column order doesn't matter)"""
        return cls(
            **{
                f.name: frame[f.metadata["column"]].to_numpy(dtype=f.metadata["dtype"])
                for f in fields(cls)
                if f.metadata["column"] is not None
            }
        )

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as infile:
            frame = pickle.load(infile)
            return cls.from_frame(frame)

    def __len__(self):
        return len(self.energy)

    def items(self):
        """(name, array) pairs for every field"""
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def select(self, selection):
        """New instance with the boolean mask / index array applied to every field"""
        return self.__class__(
            **{
                _name: None if _values is None else _values[selection]
                for _name, _values in self.items()
            }
        )

    @staticmethod
    def feature_names():
        """Fields usable as fit inputs: those present in real data"""
        return [f.name for f in fields(RealData)]

    def features(self):
        """(n_events, n_features) array of the real-data variables, ready for a fit"""
        missing = [
            _name for _name in self.feature_names() if getattr(self, _name) is None
        ]
        if missing:
            raise ValueError(f"Derived features not computed yet: {missing}")
        return np.column_stack([getattr(self, _name) for _name in self.feature_names()])


@dataclass(eq=False, kw_only=True)
class SimData(RealData):
    category: np.ndarray = column("Simulation category", dtype=int)
    weight: np.ndarray = column("Simulation weight")
    neutrino_energy: np.ndarray = column("Simulated neutrino energy")
    neutrino_flight_dist: np.ndarray = column("Simulated neutrino flight distance")
    muon_momentum: np.ndarray = column("Simulated muon momentum")

    def labels(self):
        return self.category == 21
