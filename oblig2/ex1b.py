import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2025)

n = 10_000
theta_0 = 1 #choosing theta_0 = 1


#draw n samples form exponential 
X = np.random.exponential(theta_0, n)
theta_hat = np.mean(X)

X_misspec = np.random.uniform(0, 2*theta_0, n) #letting it have the same mean
theta_hat_misspec = np.mean(X_misspec)


theta_star = []
theta_star_misspec = []
for i in range(n):
    X = np.random.exponential(theta_hat, n)
    theta_star.append(np.mean(X))

    X_misspec = np.random.exponential(theta_hat_misspec, n)
    theta_star_misspec.append(np.mean(X_misspec))
    
#sample from chi squared
exact_samples = theta_hat * np.random.chisquare(2*n, n)/(2*n)

exact_sampels_misspec = theta_hat_misspec * np.random.chisquare(2*n, n)/(2*n)

normal_samples = np.random.normal(theta_hat, (theta_hat**2)/np.sqrt(n), n)
normal_samples_misspec = np.random.normal(theta_hat_misspec, (theta_hat_misspec**2)/np.sqrt(n), n)

ax, fig = plt.subplots(1, 2, figsize=(10, 5))

fig[0].hist(theta_star, bins=50, alpha=0.5, label='Parametric Bootstrap')
fig[0].hist(exact_samples, bins=50, alpha=0.5, label='True Hjort')
fig[0].hist(normal_samples, bins=50, alpha=0.5, label='Normal')
fig[0].set_xlabel(r'$\hat \theta$')
fig[0].set_ylabel('Frequency')
fig[0].set_title('Exponential (Correctly Specified)')
fig[0].legend(loc='lower left')

fig[1].hist(theta_star_misspec, bins=50, alpha=0.5, label='Parametric Bootstrap')
fig[1].hist(exact_sampels_misspec, bins=50, alpha=0.5, label='True Hjort')
fig[1].hist(normal_samples_misspec, bins=50, alpha=0.5, label='Normal')
fig[1].set_xlabel(r'$\hat \theta$')
fig[1].set_title('Uniform (Misspecified)')
fig[1].legend(loc='lower right', fontsize='small')
plt.savefig('figs/ex1b.pdf', format='pdf', dpi=300)
plt.show()
