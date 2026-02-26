
#%%
import pandas as pd
import matplotlib.pyplot as plt

#Grid search optimum B=0.357 sigma = 0.243 gamma = 0.107
#SSE 912.73

# Read the data
df = pd.read_csv(r"C:\Users\isabe\OneDrive\Documents\BME2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)

#Get values from variables in csv
day = df["day"].values
active = df["active reported daily cases"].values

S_0=17900
E_0=0
I_0=1
R_0=0

#initialize S,E,I,R as empty arrays or lists
S = []
E = []
I = []
R = []

#set first item in each list equal to initial values S0, E0, I0, R0
