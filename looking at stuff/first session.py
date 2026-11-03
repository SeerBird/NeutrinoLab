#First let's open the simulation file
from matplotlib import pyplot as plt
import pickle

with open("data/MicroBooNE/Simulation.pkl", 'rb') as infile:
  data = pickle.load(infile)
#Let's look at the variables contained in the dataset

for variable_name in list(data):
  print(variable_name)
#We can plot, for example, the simulated energy for the neutrino


plt.hist(data['Simulated neutrino energy'], bins=100, density=True)
plt.xlabel('Simulated neutrino energy [GeV]')
plt.ylabel('Candidates')
plt.show()