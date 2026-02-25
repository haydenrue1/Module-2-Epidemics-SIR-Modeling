#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Read the data
df = pd.read_csv(r"C:\Users\isabe\OneDrive\Documents\BME2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#1.csv", parse_dates=['date'], header=0, index_col=None)

#%%
# Create and label a plot of the active cases over time
plt.figure(figsize=(10, 6))
plt.scatter(df['day'], df['active reported daily cases'])
plt.xlabel('Day')
plt.ylabel('Active Cases')
plt.title('Day vs Active Infections')
plt.show()