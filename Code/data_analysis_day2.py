# ChatGPT was used to assist in implementing the grid search used to optimize the beta, gamma, and sigma parameters by minimizing SSE
# It was also used to help visualize the future infection trend and mark the predicted peak infection day on the plot
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from csv file
df = pd.read_csv(r"/Users/haydenrue/Desktop/Comp BME/Module 2/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
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

#Grid search to obtain optimal parameters
optimal_sse = np.inf
optimal_beta = None
optimal_sigma = None
optimal_gamma = None

total_days = len(day)

#Find lowest sse by running eulers method for each parameter
for beta in beta_values:
    for sigma in sigma_values:
        for gamma in gamma_values:
            I_model = run_euler(beta, sigma, gamma, total_days)
            I_model = I_model[:len(active)]
            sse = np.sum((active - I_model) **2)

            if sse < optimal_sse:
                optimal_sse = sse
                optimal_beta = beta
                optimal_gamma = gamma
                optimal_sigma = sigma

print("Optimal Beta:", optimal_beta)
print("Optimal Sigma:", optimal_sigma)
print("Optimal Gamma:", optimal_gamma)
print("Optimal SSE:", optimal_sse)

#Run model using new optimal parameters
I_best = run_euler(optimal_beta, optimal_sigma, optimal_gamma, total_days)

#Plot the model vs given data
plt.figure()
plt.scatter(day, active, label ="Day 2 Data")
plt.plot(day, I_best[:len(day)], label ="SEIR Model")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("SEIR Model Fit to Data")
plt.legend()
plt.show()

#Predict future trends
future_days = 200

#Run eulers method on future days and find peak infections and peak day
I_future = run_euler(optimal_beta, optimal_sigma, optimal_gamma, future_days)
peak_infections = np.max(I_future)
peak_day = np.argmax(I_future)
print("Peak infections:", peak_infections)
print("Peak occurs on day:", peak_day)

plt.figure()
plt.scatter(day, active, label="Observed Data")
plt.plot(range(len(I_future)), I_future, label="SEIR Prediction")
plt.axvline(peak_day, linestyle="--", label="Peak Day")
plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.legend()
plt.show()