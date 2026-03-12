##ChatGPT was used to assist in structuring the code to analyze the interventions of a vaccine campaign and testing + quarantine, as well as to help write the code to run the SEIR model with and without interventions and to visualize the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data from csv file
df = pd.read_csv(r"C:\Users\isabe\OneDrive\Documents\BME2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#3.csv")
day = df["day"].values
active = df["active reported daily cases"].values

#%%
# Set up variables
N = 17900
S0, E0, I0, R0 = N - 1, 0, 1, 0

# Euler function for SEIR
def run_euler(beta, sigma, gamma, total_days, intervention=False):
    S, E, I, R = [S0], [E0], [I0], [R0]
    h = 1

    vaccine_day1 = 70
    vaccine_day2 = 80
    vaccine_day3 = 90
    vaccinated_students = 2000
    efficacy = 0.90
    effectively_vaccinated = vaccinated_students * efficacy  # 1800

    for t in range(total_days):
        # Vaccine campaign on day 70, 80, and 90
        if intervention and t == vaccine_day1 or t==vaccine_day2 or t==vaccine_day3: # Check if it's the intervention day and if intervention is enabled
            actual_vaccinated = min(S[t], effectively_vaccinated) # Ensure we don't vaccinate more than the susceptible population
            S[t] = S[t] - actual_vaccinated #subtracting vaccinated from susceptible
            R[t] = R[t] + actual_vaccinated #adding vaccinated to recovered since they are now immune

        # Derivatives
        dS = -(beta * S[t] * I[t]) / N
        dE = (beta * S[t] * I[t]) / N - (sigma * E[t])
        dI = (sigma * E[t]) - (gamma * I[t])
        dR = (gamma * I[t])

        # Update values
        S.append(S[t] + dS * h)
        E.append(E[t] + dE * h)
        I.append(I[t] + dI * h)
        R.append(R[t] + dR * h)

    return np.array(S), np.array(E), np.array(I), np.array(R)

# Optimization (grid search)
beta_values = np.linspace(0.1, 0.5, 10) # Transmission rate range 0.1 to 0.5
gamma_values = np.linspace(0.05, 0.2, 10) # Recovery rate 5 to 20 days
sigma_values = np.linspace(0.1, 0.5, 10) # Incubation rate 2 to 10 days

optimal_sse = np.inf # Initialize optimal sum of squared errors to infinity
optimal_beta = None # Initialize optimal beta to None
optimal_sigma = None 
optimal_gamma = None

total_days = len(day) # Total number of days in the dataset

for beta in beta_values:
    for sigma in sigma_values:
        for gamma in gamma_values:
            S_model, E_model, I_model, R_model = run_euler(beta, sigma, gamma, total_days)
            I_model = I_model[:len(active)] # Ensure model predictions are the same length as observed data
            sse = np.sum((active - I_model) ** 2) # Calculate sum of squared errors between observed active cases and model predictions

            if sse < optimal_sse:
                optimal_sse = sse # Update optimal parameters if current sse is lower than the best found so far
                optimal_beta = beta # Update optimal beta
                optimal_gamma = gamma
                optimal_sigma = sigma

print("Optimal Beta:", optimal_beta)
print("Optimal Sigma:", optimal_sigma)
print("Optimal Gamma:", optimal_gamma)
print("Optimal SSE:", optimal_sse)

# Run model using optimal parameters
S_best, E_best, I_best, R_best = run_euler(optimal_beta, optimal_sigma, optimal_gamma, total_days) # Run the SEIR model with the optimal parameters to get the best fit for S, E, I, R over time

# Plot model fit
plt.figure()
plt.scatter(day, active, label="Observed Data")
plt.plot(day, I_best[:len(day)], label="SEIR Model Fit")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("SEIR Model Fit to Data")
plt.legend()
plt.show()

# Predict future trends
future_days = 200 

# No intervention
S_future, E_future, I_future, R_future = run_euler(optimal_beta, optimal_sigma, optimal_gamma, future_days, intervention=False) # Run the SEIR model for future predictions without intervention

# With vaccine campaign
S_vax, E_vax, I_vax, R_vax = run_euler(optimal_beta, optimal_sigma, optimal_gamma, future_days, intervention=True) # Run the SEIR model for future predictions with the vaccine campaign intervention starting on day 60

# Peaks
peak_infections = np.max(I_future) # Find the maximum number of infections in the future predictions without intervention
peak_day = np.argmax(I_future) # Find the day on which the peak number of infections occurs in the future predictions without intervention

peak_infections_vax = np.max(I_vax)
peak_day_vax = np.argmax(I_vax)

print("No intervention peak infections:", peak_infections)
print("No intervention peak day:", peak_day)
print("With vaccine peak infections:", peak_infections_vax)
print("With vaccine peak day:", peak_day_vax)

# Plot future comparison
plt.figure()
plt.scatter(day, active, label="Observed Data")
plt.plot(range(len(I_future)), I_future, label="No Intervention")
plt.plot(range(len(I_vax)), I_vax, label="Vaccine Campaign")
plt.axvline(70, linestyle="--", label="Vaccine Day 1")
plt.axvline(80, linestyle="--", label="Vaccine Day 2")
plt.axvline(90, linestyle="--", label="Vaccine Day 3")
plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.title("SEIR Prediction with Vaccine Intervention")
plt.legend()
plt.show()

#%% 

# Set up variables
N = 17900
S0, E0, I0, R0 = N - 1, 0, 1, 0

# Euler function for SEIR
def run_euler(beta, sigma, gamma, total_days, intervention=False):
    S, E, I, R = [S0], [E0], [I0], [R0]
    h = 1

    vaccine_day = 70
    vaccinated_students = 2000
    efficacy = 0.90
    effectively_vaccinated = vaccinated_students * efficacy  # 1800

    for t in range(total_days):
        # Vaccine campaign on day 70
        if intervention and t == vaccine_day:
            actual_vaccinated = min(S[t], effectively_vaccinated)
            S[t] = S[t] - actual_vaccinated
            R[t] = R[t] + actual_vaccinated

        # Derivatives
        dS = -(beta * S[t] * I[t]) / N
        dE = (beta * S[t] * I[t]) / N - (sigma * E[t])
        dI = (sigma * E[t]) - (gamma * I[t])
        dR = (gamma * I[t])

        # Update values
        S.append(S[t] + dS * h)
        E.append(E[t] + dE * h)
        I.append(I[t] + dI * h)
        R.append(R[t] + dR * h)

    return np.array(S), np.array(E), np.array(I), np.array(R)

# Optimization (grid search)
beta_values = np.linspace(0.1, 0.5, 10)
gamma_values = np.linspace(0.05, 0.2, 10)
sigma_values = np.linspace(0.1, 0.5, 10)

optimal_sse = np.inf
optimal_beta = None
optimal_sigma = None
optimal_gamma = None

total_days = len(day)

for beta in beta_values:
    for sigma in sigma_values:
        for gamma in gamma_values:
            S_model, E_model, I_model, R_model = run_euler(beta, sigma, gamma, total_days)
            I_model = I_model[:len(active)]
            sse = np.sum((active - I_model) ** 2)

            if sse < optimal_sse:
                optimal_sse = sse
                optimal_beta = beta
                optimal_gamma = gamma
                optimal_sigma = sigma

print("Optimal Beta:", optimal_beta)
print("Optimal Sigma:", optimal_sigma)
print("Optimal Gamma:", optimal_gamma)
print("Optimal SSE:", optimal_sse)

# Run model using optimal parameters
S_best, E_best, I_best, R_best = run_euler(optimal_beta, optimal_sigma, optimal_gamma, total_days)

# Plot model fit
plt.figure()
plt.scatter(day, active, label="Observed Data")
plt.plot(day, I_best[:len(day)], label="SEIR Model Fit")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("SEIR Model Fit to Data")
plt.legend()
plt.show()

# Predict future trends
future_days = 200

# No intervention
S_future, E_future, I_future, R_future = run_euler(optimal_beta, optimal_sigma, optimal_gamma, future_days, intervention=False)

# With vaccine campaign
S_vax, E_vax, I_vax, R_vax = run_euler(optimal_beta, optimal_sigma, optimal_gamma, future_days, intervention=True)

# Peaks
peak_infections = np.max(I_future)
peak_day = np.argmax(I_future)

peak_infections_vax = np.max(I_vax)
peak_day_vax = np.argmax(I_vax)

print("No intervention peak infections:", peak_infections)
print("No intervention peak day:", peak_day)
print("With vaccine peak infections:", peak_infections_vax)
print("With vaccine peak day:", peak_day_vax)

# Plot future comparison
plt.figure()
plt.scatter(day, active, label="Observed Data")
plt.plot(range(len(I_future)), I_future, label="No Intervention")
plt.plot(range(len(I_vax)), I_vax, label="Vaccine Campaign")
plt.axvline(70, linestyle="--", label="Vaccine Day")
plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.title("SEIR Prediction with Vaccine Intervention")
plt.legend()
plt.show()

#%%
def run_euler(beta, sigma, gamma, total_days, intervention_day=None):
    S, E, I, R = [S0], [E0], [I0], [R0]
    h = 1

    for t in range(total_days):
        current_gamma = gamma

        # Testing + quarantine starts on day 70
        if intervention_day is not None and t >= intervention_day:
            infectious_period = 1 / gamma
            new_infectious_period = infectious_period - 2

            if new_infectious_period <= 0:
                new_infectious_period = 0.1

            current_gamma = 1 / new_infectious_period

        dS = -(beta * S[t] * I[t]) / N
        dE = (beta * S[t] * I[t]) / N - (sigma * E[t])
        dI = (sigma * E[t]) - (current_gamma * I[t])
        dR = (current_gamma * I[t])

        S.append(S[t] + dS * h)
        E.append(E[t] + dE * h)
        I.append(I[t] + dI * h)
        R.append(R[t] + dR * h)

    return np.array(I)

future_days = 200

I_future_no_intervention = run_euler(optimal_beta, optimal_sigma, optimal_gamma, future_days)
I_future_with_quarantine = run_euler(optimal_beta, optimal_sigma, optimal_gamma, future_days, intervention_day=70)

peak_infections = np.max(I_future_with_quarantine)
peak_day = np.argmax(I_future_with_quarantine)

print("Peak infections with testing + quarantine:", peak_infections)
print("Peak occurs on day:", peak_day)

plt.figure()
plt.scatter(day, active, label="Observed Data")
plt.plot(range(len(I_future_no_intervention)), I_future_no_intervention, label="No Intervention")
plt.plot(range(len(I_future_with_quarantine)), I_future_with_quarantine, label="Testing + Quarantine")
plt.axvline(70, linestyle="--", label="Intervention Day")
plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.title("Testing + Quarantine Starting Day 70")
plt.legend()
plt.show()