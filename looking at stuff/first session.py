# First let's open the simulation file
import numpy as np
from matplotlib import pyplot as plt
import pickle

from pandas import DataFrame

with (open("data/MicroBooNE/Simulation.pkl", 'rb') as infile):
    (closest_ray_track_dist,
     lepton_track_length,
     longest_track_dist,
     category,
     muon_ML_score,
     end_z,
     end_y,
     end_x,
     track_ML_score,
     particle_LLH,
     start_z,
     start_y,
     start_x,
     vert_x,
     vert_y,
     vert_z,
     energy,
     subevent,
     length_momentum,
     scat_momentum,
     weight,
     neutrino_energy,
     neutrino_flight_dist,
     muon_momentum) = np.asarray(pickle.load(infile).values.T, dtype=float)
    subevent, category = np.asarray(subevent, dtype = int), np.asarray(category, dtype = int)
# Let's look at the variables contained in the dataset

# We can plot, for example, the simulated energy for the neutrino
cosmic_events = category == 4
print(f"Fraction of non-muon cosmic events:{
np.count_nonzero(np.isnan(muon_momentum[cosmic_events]))
/ len(muon_momentum[cosmic_events])}")
plt.hist(neutrino_energy, bins=100, density=True)
plt.xlabel('Simulated neutrino energy [GeV]')
plt.ylabel('Candidates')
plt.show()
