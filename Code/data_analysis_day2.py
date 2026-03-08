
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Grid search optimum B=0.357 sigma = 0.243 gamma = 0.107
#SSE 912.73

# Load data from csv file
df = pd.read_csv(r"C:\Users\isabe\OneDrive\Documents\BME2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
day = df["day"].values
active = df["active reported daily cases"].values

# Set up variables
N = 17900
S0, E0, I0, R0 = N - 1, 0, 1, 0

# Euler function for SEIR
def run_euler(beta,sigma,gamma,total_days):
    S, E, I, R = [S0], [E0], [I0], [R0] #Initialize S,E,I,R as empty lists; set first item in list equal to initial values
    h = 1 #step size is 1 day

    for t in range(total_days):
        #Calculate derivatives from slide 6 class presentation
        dS = -(beta * S[t]* I[t]) / N
        dE = (beta * S[t] * I[t]) / N - (sigma * E[t])
        dI = (sigma * E[t]) - (gamma * I[t])
        dR = (gamma * I[t])

        #Update values
        S.append(S[t] + dS * h)
        E.append(E[t] + dE * h)
        I.append(I[t] + dI * h)
        R.append(R[t] + dR * h)
    return np.array(I)

# Optimzation (grid search)
# Use estimated R0 to make an estimate for beta and gamma. R0 = 1.93 so beta = 1.93 * gamma
beta_values = np.linspace(0.1,0.5,10) # Transmission rate range 0.1 to 0.5
gamma_values = np.linspace(0.05, 0.2, 10) # Recovery rate 5 to 20 days
sigma_values = np.linspace(0.1, 0.5, 10) # Incubation rate 2 to 10 days