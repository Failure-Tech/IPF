import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
from rdkit import Chem, DataStructs
import nltk
from rdkit.Chem import PandasTools, AllChem, Descriptors
import tensorboard as tb
import seaborn as sns

#imports for data
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

#saves .mol as image(for testing purposes; also this is currently a quercetin model, will import multiple models latter)
mol = Chem.MolFromMolFile('Quercetin.mol')
img = Draw.MolToImage(mol)
img.save('molecule_image.png')

#Chemical descriptors
mw = rdkit.Chem.Descriptors.MolWt(mol)
logp = rdkit.Chem.Descriptors.MolLogP(mol)
tpsa = rdkit.Chem.Descriptors.TPSA(mol)
num_h_donors = rdkit.Chem.Descriptors.NumHDonors(mol)
num_h_acceptors = rdkit.Chem.Descriptors.NumHAcceptors(mol)

#printing data(also for testing purposes)
print(f"Molecular Weight: {mw}")
print(f"LogP: {logp}")
print(f"Topological Polar Surface Area: {tpsa}")
print(f"Number of Hydrogen Donors: {num_h_donors}")
print(f"Number of Hydrogen Acceptors: {num_h_acceptors}")


# Import data
sdf_file_path = ''

supplier = Chem.SDMolSupplier(sdf_file_path) # reading from sd file
mol_list = (mol for mol in supplier if mol is not None) # change if necessary

# Data stuff

# Model stuff

class Qsar(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, out_size):
        super(Qsar, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)

        return out

model = Qsar() # put in values of stuff

learning_rate = 0.001

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training

# Evaluating

# Deploying model (may have to build front end for coolness)
