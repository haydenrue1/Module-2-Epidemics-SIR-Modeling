#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

days = 121
N = 17900
S0, E0, I0, R0 = N - 1, 0, 1, 0

# Load data from csv file
#df = pd.read_csv(r"/Users/haydenrue/Desktop/Comp BME/Module 2/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
df = pd.read_csv(r"/Users/haydenrue/Desktop/Comp BME/Module 2/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#3.csv")
day = df["day"].values
active = df["active reported daily cases"].values

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


optimal_beta = .5
optimal_gamma = .2
optimal_sigma = .32222222222

I_future = run_euler(optimal_beta, optimal_sigma, optimal_gamma, days)

plt.figure(figsize=(10, 6))
plt.scatter(df['day'], df['active reported daily cases'])
plt.plot(range(len(I_future)), I_future, label="SEIR Prediction")
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Day vs Active Infections')
plt.show()


