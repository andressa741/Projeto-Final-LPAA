# %%
# Import as bibliotecas 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Import do dataset
from sklearn.datasets import load_diabetes
X,y = load_diabetes(return_X_y = True, as_frame = True, scaled = False)
# %%
