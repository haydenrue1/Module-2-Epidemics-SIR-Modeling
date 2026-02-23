# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025
#Chat GPT-4o was used to assist in writing code to sweep lambda values, create the for loop, and print statements to give optimal doses and best_diff

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt


#%% define drug models

# define toxicity levels for each drug (lambda)
metformin_lambda = 0.5

lisinopril_lambda = 0.8

escitalopram_lambda = 0.3

def metformin(x):   # mild toxicity, moderate efficacy
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity
def lisinopril(x):  # strong efficacy, higher toxicity
    efficacy = np.exp(-0.1*(x-7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity
def escitalopram(x):  # weaker efficacy, low toxicity
    efficacy = 0.6 * np.exp(-0.1*(x-4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity

#%% plot drug efficacies
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')
plt.title('Drug Efficacy vs Dosage')
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()
plt.show()

# %% Find optimal dosages for each drug

# First method: Steepest Ascent using the update rule

# first, need the first derivative (gradient)
def gradient(f, x, h=1e-4):
    """Central difference approximation for f'(x)."""
    return (f(x + h) - f(x - h)) / (2*h)

def steepest_ascent(f, x0, h_step=0.1, tol=1e-6, max_iter=1000):
    x = x0 # update initial guess
    for i in range(max_iter):
        grad = gradient(f, x)
        x_new = x + h_step * grad     
        
        if abs(x_new - x) < tol:      # convergence condition, when solution is 0
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin, opt_effect_metformin = steepest_ascent(metformin, x0=1.0)
print(f"Steepest Ascent Method - Optimal Metformin Dose: {opt_dose_metformin:.2f} mg")
print(f"Steepest Ascent Method - Optimal Metformin Effect: {opt_effect_metformin*100:.2f}%")

# lisinopril
opt_dose_lisinopril, opt_effect_lisinopril = steepest_ascent(lisinopril, x0=1.0)
print(f"Steepest Ascent Method - Optimal Lisinopril Dose: {opt_dose_lisinopril:.2f} mg")
print(f"Steepest Ascent Method - Optimal Lisinopril Effect: {opt_effect_lisinopril*100:.2f}%")

# escitalopram
opt_dose_escitalopram, opt_effect_escitalopram = steepest_ascent(escitalopram, x0=1.0)
print(f"Steepest Ascent Method - Optimal Escitalopram Dose: {opt_dose_escitalopram:.2f} mg")
print(f"Steepest Ascent Method - Optimal Escitalopram Effect: {opt_effect_escitalopram*100:.2f}%")


# combined drug
opt_dose_combined, opt_effect_combined = steepest_ascent(lambda x: metformin(x) + lisinopril(x) + escitalopram(x),x0=5.0)
print(f"Steepest Ascent Method - Optimal Combined Dose: {opt_dose_combined:.2f} mg")
print(f"Steepest Ascent Method - Optimal Combined Effect: {opt_effect_combined*100:.2f}%")

# %% Newton's method

# requires second derivative
def second_derivative(f, x, h=1e-4):
    """Central difference approximation for f''(x)."""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def newtons_method(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(f, x)
        hess = second_derivative(f, x)
        
        if hess == 0:  # avoid division by zero
            print("Zero second derivative. No solution found.")
            return x, f(x)
        
        x_new = x - grad / hess
        
        if abs(x_new - x) < tol:
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin_nm, opt_effect_metformin_nm = newtons_method(metformin, x0=1.0)
print(f"Newton's Method - Optimal Metformin Dose: {opt_dose_metformin_nm:.2f} mg")
print(f"Newton's Method - Optimal Metformin Effect: {opt_effect_metformin_nm*100:.2f}%")                

# lisinopril
opt_dose_lisinopril_nm, opt_effect_lisinopril_nm = newtons_method(lisinopril, x0=1.0)
print(f"Newton's Method - Optimal Lisinopril Dose: {opt_dose_lisinopril_nm:.2f} mg")
print(f"Newton's Method - Optimal Lisinopril Effect: {opt_effect_lisinopril_nm*100:.2f}%")

# escitalopram
opt_dose_escitalopram_nm, opt_effect_escitalopram_nm = newtons_method(escitalopram, x0=1.0)
print(f"Newton's Method - Optimal Escitalopram Dose: {opt_dose_escitalopram_nm:.2f} mg")
print(f"Newton's Method - Optimal Escitalopram Effect: {opt_effect_escitalopram_nm*100:.2f}%")

# combined drug
opt_dose_combined_nm, opt_effect_combined_nm = newtons_method(lambda x: metformin(x) + lisinopril(x) + escitalopram(x),x0=5.0)
print(f"Newton's Method - Optimal Combined Dose: {opt_dose_combined_nm:.2f} mg")
print(f"Newton's Method - Optimal Combined Effect: {opt_effect_combined_nm*100:.2f}%")

#Best lambda value to achieve the optimal dose for lisinopril
# 1) Target is the combined optimal dose from Newton's method result
target_dose = opt_dose_combined_nm

# 2) Save original lambda so it can be restored later
lisinopril_lambda_original = lisinopril_lambda

# 3) Sweep lambdas
lambda_values = np.linspace(0.0, 2.0, 201)

best_lambda = None
best_dose = None
best_effect = None
best_diff = np.inf

lisinopril_opt_doses = [] # store the lisinopril optimal dose for each lambda

for lam in lambda_values:
    lisinopril_lambda = lam # update global

    dose_l,effect_l = newtons_method(lisinopril, x0=1.0)
    lisinopril_opt_doses.append(dose_l)

    diff = abs(dose_l - target_dose)
    if diff < best_diff:
        best_diff = diff
        best_lambda = lam
        best_dose = dose_l
        best_effect = effect_l


# 4) Restore original lambda
lisinopril_lambda = lisinopril_lambda_original

print(f"Target combined dose = {target_dose:.4f} mg")
print(f"Best lisinopril lambda = {best_lambda:.4f}")
print(f"lisinopril optimal dose at best lambda = {best_dose:.4f} mg")
print(f"Difference = {best_diff:.4f} mg")
print(f"Lisinopril optimal effect at best lambda = {best_effect*100:.2f}%")


#Plot A: Combined curve vs Lisinopril curve at best lambda (both vs dose)

lisinopril_lambda = best_lambda
combined_curve = metformin(x) + lisinopril(x) + escitalopram(x)
lisinopril_curve_best = lisinopril(x)

plt.figure(figsize=(10,6))
plt.plot(x, combined_curve, label='Combined Effect', color='red', linestyle='--')
plt.plot(x, lisinopril_curve_best, label=f'Lisinopril (best lambda = {best_lambda:.3f})', color='blue')
plt.axvline(target_dose, linestyle=':', label=f'Target Combined Dose = {target_dose:.2f} mg')
plt.axvline(best_dose, linestyle=':', label=f'Lisinopril Optimal Dose = {best_dose:.2f} mg')
plt.title("Combined vs Lisinopril (at best lambda)")
plt.xlabel("Dosage (mg)")
plt.ylabel("Net Effect")
plt.legend()
plt.show()

#Restore original again after plotting
lisinopril_lambda = lisinopril_lambda_original

print(f"Lisinopril optimal dose at best lambda = {best_dose:.4f} mg") #prints the closest optimal dose
print(f"Difference = {best_diff:.4f} mg") # shows how far the optimal dose is from the target(combined)

# 1) We would pick Lisinopril as it has the highest optimal drug effect of 0.8573 compared to Metformin at 0.7753 and Escitalopram at 0.5960
# 2) The optimal drug effect is 85.73%
# 3) This occurs at a dose of 6.80 mg for Lisinopril
# 4) Yes, as we sweep lambda we can see the value for the optimal dose changes. Lowering lambda pushes the optimal dose much higher
# 5) The closest dose for lisinopril is 6.5002 mg, which is 1.0976 mg away from the target combined dose