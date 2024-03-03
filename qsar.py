# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import MinMaxScaler
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors

# # Define the QSAR model with increased capacity
# class Qsar(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(Qsar, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = nn.Linear(hidden_dim2, output_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x

# # Define the training function
# def train_model(model, optimizer, criterion, inputs, targets, epochs=100, batch_size=32):
#     losses = []
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for i in range(len(inputs)):
#             input_row = inputs[i].unsqueeze(0)
#             target = targets[i].unsqueeze(0).unsqueeze(0)

#             # Forward pass
#             output = model(input_row.unsqueeze(0))  # Adding unsqueeze to match the expected input shape
#             loss = criterion(output, target.unsqueeze(0))  # Adding unsqueeze to match the expected target shape

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         epoch_loss = total_loss / len(inputs)
#         losses.append(epoch_loss)

#         # Print the loss every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")

#     return losses

# # Load the dataset
# df = pd.read_csv("./assets/AID_294_data.csv")

# # Initialize lists to store features and targets
# features_list = []
# targets_list = []

# # Iterate over each row in the dataset
# for i in range(2, 159):
#     standard_values = pd.read_csv('./Assets/AID_294_data.csv', usecols=['Standard Value'])
#     smiles = pd.read_csv('./Assets/AID_294_data.csv', usecols=['PUBCHEM_EXT_DATASOURCE_SMILES'])
#     mol = Chem.MolFromSmiles(smiles.iloc[i]['PUBCHEM_EXT_DATASOURCE_SMILES'])

#     # Extract features
#     features = [Descriptors.MolWt(mol), 
#                 Descriptors.MolLogP(mol), 
#                 Descriptors.TPSA(mol), 
#                 Descriptors.NumHDonors(mol), 
#                 #Descriptors.NumHAcceptors(mol)
#                 ]
    
#     # Append to lists
#     features_list.append(features)
#     targets_list.append(float(standard_values.iloc[i]['Standard Value']))  # Convert to float

# # Convert lists to numpy arrays
# features_array = np.array(features_list)
# targets_array = np.array(targets_list)

# # Normalize the features
# scaler = MinMaxScaler()
# features_normalized = scaler.fit_transform(features_array)

# # Convert numpy arrays to PyTorch tensors
# inputs = torch.tensor(features_normalized, dtype=torch.float32)
# targets = torch.tensor(targets_array, dtype=torch.float32)

# # Define the model, optimizer, and criterion with increased capacity
# model = Qsar(input_dim=4, hidden_dim1=64, hidden_dim2=32, output_dim=1)  # Increased capacity
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adjusted learning rate
# criterion = nn.MSELoss()

# # Train the model with increased epochs
# losses = train_model(model, optimizer, criterion, inputs, targets, epochs=200, batch_size=32)  # Increased epochs

# # Plot the training loss
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import Descriptors

# # Define the QSAR model with increased capacity and regularization
# class Qsar(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(Qsar, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = nn.Linear(hidden_dim2, output_dim)
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(0.2)  # Dropout regularization

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)  # Apply dropout
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)  # Apply dropout
#         x = self.fc3(x)
#         return x

# # Define the training function
# def train_model(model, optimizer, criterion, train_loader, val_loader, epochs=100):
#     train_losses = []
#     val_losses = []
#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         train_loss = 0.0
#         for inputs, targets in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets have the correct shape
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * inputs.size(0)
#         train_loss /= len(train_loader.dataset)
#         train_losses.append(train_loss)
        
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in val_loader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets have the correct shape
#                 val_loss += loss.item() * inputs.size(0)
#             val_loss /= len(val_loader.dataset)
#             val_losses.append(val_loss)

#         # Print the loss every epoch
#         print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     return train_losses, val_losses

# # Load the dataset
# df = pd.read_csv("./assets/AID_294_data.csv")

# # Extract features and targets
# features_list = []
# targets_list = []
# for i in range(2, 159):
#     standard_values = pd.read_csv('./Assets/AID_294_data.csv', usecols=['Standard Value'])
#     smiles = pd.read_csv('./Assets/AID_294_data.csv', usecols=['PUBCHEM_EXT_DATASOURCE_SMILES'])
#     mol = Chem.MolFromSmiles(smiles.iloc[i]['PUBCHEM_EXT_DATASOURCE_SMILES'])
#     features = [Descriptors.MolWt(mol), 
#                 Descriptors.MolLogP(mol), 
#                 Descriptors.TPSA(mol), 
#                 Descriptors.NumHDonors(mol), 
#                 Descriptors.NumHAcceptors(mol)
#                 ]
#     features_list.append(features)
#     targets_list.append(float(standard_values.iloc[i]['Standard Value']))  # Convert to float

# # Convert lists to numpy arrays
# features_array = np.array(features_list)
# targets_array = np.array(targets_list)

# # Normalize the features
# scaler = MinMaxScaler()
# features_normalized = scaler.fit_transform(features_array)

# # Split the dataset into train and validation sets
# X_train, X_val, y_train, y_val = train_test_split(features_normalized, targets_array, test_size=0.2, random_state=42)

# # Convert numpy arrays to PyTorch tensors
# train_inputs = torch.tensor(X_train, dtype=torch.float32)
# train_targets = torch.tensor(y_train, dtype=torch.float32)
# val_inputs = torch.tensor(X_val, dtype=torch.float32)
# val_targets = torch.tensor(y_val, dtype=torch.float32)

# # Create DataLoader for train and validation sets
# train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
# val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# # Define the model, optimizer, and criterion with increased capacity and regularization
# model = Qsar(input_dim=5, hidden_dim1=64, hidden_dim2=32, output_dim=1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate
# criterion = nn.MSELoss()

# # Train the model with increased epochs
# train_losses, val_losses = train_model(model, optimizer, criterion, train_loader, val_loader, epochs=1000)

# # Plot the training and validation losses
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Losses')
# plt.legend()
# plt.show()

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

# Define the QSAR model with increased capacity and regularization
class Qsar(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Qsar, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout regularization

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x

# Define the training function with early stopping
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs=100, early_stop_patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = early_stop_patience

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets have the correct shape
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            print("Output: ", outputs.detach().numpy().flatten())
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets have the correct shape
                val_loss += loss.item() * inputs.size(0)
                for i in range(len(targets)):
                    print(f"Drug {i+1}: Predicted Residual Activity = {outputs[i]}, Actual Residual Activity = {targets[i]}")
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

        # Print the loss every epoch
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = early_stop_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses

# Load the dataset
df = pd.read_csv("./assets/AID_294_data.csv")

# Extract features and targets
features_list = []
targets_list = []
for i in range(2, 159):
    standard_values = pd.read_csv('./Assets/AID_294_data.csv', usecols=['Standard Value'])
    smiles = pd.read_csv('./Assets/AID_294_data.csv', usecols=['PUBCHEM_EXT_DATASOURCE_SMILES'])
    mol = Chem.MolFromSmiles(smiles.iloc[i]['PUBCHEM_EXT_DATASOURCE_SMILES'])
    features = [Descriptors.MolWt(mol), 
                Descriptors.MolLogP(mol), 
                Descriptors.TPSA(mol), 
                Descriptors.NumHDonors(mol), 
                Descriptors.NumHAcceptors(mol)
                ]
    features_list.append(features)
    targets_list.append(float(standard_values.iloc[i]['Standard Value']))  # Convert to float

# Convert lists to numpy arrays
features_array = np.array(features_list)
targets_array = np.array(targets_list)

# Normalize the features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features_array)

# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_normalized, targets_array, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
train_inputs = torch.tensor(X_train, dtype=torch.float32)
train_targets = torch.tensor(y_train, dtype=torch.float32)
val_inputs = torch.tensor(X_val, dtype=torch.float32)
val_targets = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader for train and validation sets
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Define the model, optimizer, and criterion with increased capacity and regularization
model = Qsar(input_dim=5, hidden_dim1=128, hidden_dim2=64, output_dim=1)  # Increased capacity
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adjusted learning rate
criterion = nn.MSELoss()

# Train the model with early stopping
train_losses, val_losses = train_model(model, optimizer, criterion, train_loader, val_loader, epochs=1000, early_stop_patience=20)

# Plot the training and validation losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
