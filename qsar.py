# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import MinMaxScaler
# from copy import deepcopy as dc
# # import torchdrug
# # from torchdrug.layers import GCN
# # from torchdrug.models import MultiLayerProtection
# import tensorboard as tb
# import seaborn as sns
# #imports for data
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole

# #saves .mol as image(for testing purposes; also this is currently a quercetin model, will import multiple models latter)
# # mol = Chem.MolFromMolFile('Quercetin.mol')
# # img = Draw.MolToImage(mol)
# # img.save('molecule_image.png')

# # #Chemical descriptors
# # mw = rdkit.Chem.Descriptors.MolWt(mol)
# # logp = rdkit.Chem.Descriptors.MolLogP(mol)
# # tpsa = rdkit.Chem.Descriptors.TPSA(mol)
# # num_h_donors = rdkit.Chem.Descriptors.NumHDonors(mol)
# # num_h_acceptors = rdkit.Chem.Descriptors.NumHAcceptors(mol)

# # #printing data(also for testing purposes)
# # print(f"Molecular Weight: {mw}")
# # print(f"LogP: {logp}")
# # print(f"Topological Polar Surface Area: {tpsa}")
# # print(f"Number of Hydrogen Donors: {num_h_donors}")
# # print(f"Number of Hydrogen Acceptors: {num_h_acceptors}")

# # Data stuff
# df = pd.read_csv("./Competitions/Science_Fairs/IPF/assets/AID_294_data.csv")

# # Model stuff

# # class GCN_Qsar(nn.Module):
# #     def __init__(self, gcn_input_dim, qsar_input_dim, hidden_dim, out_dim, num_layers):
# #         super(GCN_Qsar, self).__init__()
# #         # Define layers
# #         # self.fc1 = nn.Linear(input_size, hidden_size)
# #         # self.fc2 = nn.Linear(hidden_size, hidden_size)
# #         # self.relu = nn.ReLU(hidden_size, hidden_size)
# #         # self.fc3 = nn.Linear(hidden_size, out_size)
# #         self.gcn = GCN(gcn_input_dim, hidden_dim)
# #         self.qsar_mlp = MultiLayerProtection(qsar_input_dim, hidden_dim)
# #         self.relu = nn.ReLU()
# #         self.linear = nn.Linear(hidden_dim*2, out_dim)

# #     def forward(self, graph, qsar_features):
# #         gcn_output = self.gcn(graph, graph.node_features)
# #         qsar_output = self.qsar_mlp(qsar_features)
# #         x = torch.cat([gcn_output, qsar_output], dim=1)

# #         x = self.relu(x)
# #         x = self.linear(x)
# #         return x

# class Qsar(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(Qsar, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.relu = nn.ReLU(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x

# #model = GCN_Qsar() # put in values of stuff

# model = Qsar() # put in values for stuff

# # Initializing Model

# learning_rate = 0.001
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Training

# # Evaluating

# # Deploying model (may have to build front end for coolness)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

# Data stuff
df = pd.read_csv("./Competitions/Science_Fairs/IPF/assets/AID_294_data.csv")
other_df = pd.read_csv("")

# Define the QSAR model
class Qsar(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Qsar, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Define the training function
def train_model(model, optimizer, criterion, inputs, targets, epochs=100, batch_size=32):
    losses = []
    for epoch in range(epochs):
        # Shuffle the data
        indices = torch.randperm(len(inputs))
        inputs_shuffled = inputs[indices]
        targets_shuffled = targets[indices]

        # Mini-batch training
        for i in range(0, len(inputs_shuffled), batch_size):
            input_batch = inputs_shuffled[i:i+batch_size]
            target_batch = targets_shuffled[i:i+batch_size]

            # Forward pass
            outputs = model(input_batch)
            loss = criterion(outputs, target_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the loss
            losses.append(loss.item())

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    return losses

# Prepare the data
# Assuming df contains your dataset with features and targets
# You need to preprocess the data and split it into inputs and targets
# For demonstration purposes, let's assume you have already preprocessed the data
inputs = torch.tensor(df[['feature1', 'feature2', 'feature3']].values, dtype=torch.float32)
targets = torch.tensor(df['target'].values, dtype=torch.float32)

# Normalize the inputs
scaler = MinMaxScaler()
inputs = scaler.fit_transform(inputs)

# Define the model, optimizer, and criterion
model = Qsar(input_dim=3, hidden_dim=64, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model
losses = train_model(model, optimizer, criterion, inputs, targets, epochs=100, batch_size=32)

# Plot the training loss
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
