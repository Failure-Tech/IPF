import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
# import torchdrug
# from torchdrug.layers import GCN
# from torchdrug.models import MultiLayerProtection
import tensorboard as tb
import seaborn as sns
#imports for data
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

#saves .mol as image(for testing purposes; also this is currently a quercetin model, will import multiple models latter)
# mol = Chem.MolFromMolFile('Quercetin.mol')
# img = Draw.MolToImage(mol)
# img.save('molecule_image.png')

# #Chemical descriptors
# mw = rdkit.Chem.Descriptors.MolWt(mol)
# logp = rdkit.Chem.Descriptors.MolLogP(mol)
# tpsa = rdkit.Chem.Descriptors.TPSA(mol)
# num_h_donors = rdkit.Chem.Descriptors.NumHDonors(mol)
# num_h_acceptors = rdkit.Chem.Descriptors.NumHAcceptors(mol)

# #printing data(also for testing purposes)
# print(f"Molecular Weight: {mw}")
# print(f"LogP: {logp}")
# print(f"Topological Polar Surface Area: {tpsa}")
# print(f"Number of Hydrogen Donors: {num_h_donors}")
# print(f"Number of Hydrogen Acceptors: {num_h_acceptors}")

# Data stuff
df = pd.read_csv("./Competitions/Science_Fairs/IPF/assets/AID_294_data.csv")

# Model stuff

class GCN_Qsar(nn.Module):
    def __init__(self, gcn_input_dim, qsar_input_dim, hidden_dim, out_dim, num_layers):
        super(GCN_Qsar, self).__init__()
        # Define layers
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, out_size)
        self.gcn = GCN(gcn_input_dim, hidden_dim)
        self.qsar_mlp = MultiLayerProtection(qsar_input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_dim*2, out_dim)

    def forward(self, graph, qsar_features):
        gcn_output = self.gcn(graph, graph.node_features)
        qsar_output = self.qsar_mlp(qsar_features)
        x = torch.cat([gcn_output, qsar_output], dim=1)

        x = self.relu(x)
        x = self.linear(x)
        return x

class Qsar(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Qsar, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

#model = GCN_Qsar() # put in values of stuff

model = Qsar() # put in values for stuff

# Initializing Model

learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training

# Evaluating

# Deploying model (may have to build front end for coolness)