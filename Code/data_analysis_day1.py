#Chat GPT was used to assist linearizing the eponential growth and estimating R0 from the regression

#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Read the data
df = pd.read_csv(r"/Users/haydenrue/Desktop/Comp BME/Module 2/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv", parse_dates=['date'], header=0, index_col=None)

#%%
import numpy as np

#Get values from variables in csv
t = df["day"].values
I = df["active reported daily cases"].values

#only examine first 45 days for early exponential growth
mask = t <= 45
t_exp = t[mask]
I_exp = I[mask]

#%%
#linearize to use polyfit and get r from slope

# Remove zeros before log
positive_mask = I_exp > 0
t_fit = t_exp[positive_mask]
I_fit = I_exp[positive_mask]

log_I = np.log(I_fit)

# Linear regression
coeffs = np.polyfit(t_fit, log_I, 1)
r = coeffs[0]
log_I0 = coeffs[1]
I0 = np.exp(log_I0)

# print growth rate and estimated initial infections
print("Estimated r:", r)
print("Estimated I0:", I0)

#%%
#Estimate R0

D = 5
R0 = np.exp(r * D)

print("Estimated R0:", R0)


#%%
# Create and label a plot of the active cases over time with exponential fit for growth rate

I_model = I0 * np.exp(r * t)

#Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['day'], df['active reported daily cases'])
plt.plot(t, I_model, label = "Exponential fit")
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Exponential Fit for Active Infections')
plt.legend()
plt.show()