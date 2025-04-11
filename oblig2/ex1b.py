import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2025)


n = 1000   
B = 10000    
theta_0 = 1   

# Correctly specified: Exponential data
X = np.random.exponential(theta_0, n)
theta_hat = np.mean(X)

# Misspecified: Uniform data
X_misspec = np.random.uniform(0, 2 * theta_0, n)
theta_hat_misspec = np.mean(X_misspec)

# Parametric Bootstrap
theta_star = []
theta_star_misspec = []
for _ in range(B):
    # Bootstrap under assumed exponential model
    X_star = np.random.exponential(theta_hat, n)
    theta_star.append(np.mean(X_star))
    X_star_misspec = np.random.exponential(theta_hat_misspec, n)
    theta_star_misspec.append(np.mean(X_star_misspec))

# Exact distribution samples 
exact_samples = theta_hat * np.random.chisquare(2 * n, B) / (2 * n)
exact_samples_misspec = theta_hat_misspec * np.random.chisquare(2 * n, B) / (2 * n)

# Normal approximation samples
normal_samples = np.random.normal(theta_hat, theta_hat / np.sqrt(n), B)
normal_samples_misspec = np.random.normal(theta_hat_misspec, theta_hat_misspec / np.sqrt(n), B)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


ax1.hist(theta_star, bins=50, alpha=0.5, label='Bootstrap')
ax1.hist(exact_samples, bins=50, alpha=0.5, label='Exact (Hjort)')
ax1.hist(normal_samples, bins=50, alpha=0.5, label='Normal Approx.')
ax1.set_xlabel(r'$\hat{\theta}^*$')
ax1.set_ylabel('Frequency')
ax1.set_title('Exponential (Correctly Specified)')
ax1.legend()

ax2.hist(theta_star_misspec, bins=50, alpha=0.5, label='Bootstrap')
ax2.hist(exact_samples_misspec, bins=50, alpha=0.5, label='Exact (Hjort)')
ax2.hist(normal_samples_misspec, bins=50, alpha=0.5, label='Normal Approx.')
ax2.set_xlabel(r'$\hat{\theta}^*$')
ax2.set_title('Uniform (Misspecified)')
ax2.legend()
plt.tight_layout()
plt.savefig('figs/ex1b.pdf', format='pdf', dpi=300)
plt.show()