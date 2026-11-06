import pickle

import numpy as np

with (open("data/Data.pkl", 'rb') as infile):
    frame = pickle.load(infile)
    (closest_ray_track_dist,
     lepton_track_length,
     longest_track_dist,
     _,
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
     scat_momentum) = np.asarray(pickle.load(infile).values.T, dtype=float)
    subevent = np.asarray(subevent, dtype = int)