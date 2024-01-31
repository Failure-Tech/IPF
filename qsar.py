import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem, Descriptors
import tensorboard as tb
import seaborn as sns

# Import data
sdf_file_path = ''

supplier = Chem.SDMolSupplier(sdf_file_path) # reading from sd file
mol_list = (mol for mol in supplier if mol is not None)