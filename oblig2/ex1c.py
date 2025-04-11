import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2025)


n = 1000   
B = 10_000    
theta_0 = 1   

# Correctly specified: Exponential data
X = np.random.exponential(theta_0, n)

# Misspecified: Uniform data
X_misspec = np.random.uniform(0, 2*theta_0, n)

theta_star_non_parametric = np.zeros(B)
theta_star_non_parametric_misspec = np.zeros(B)
for i in range(B):
    X_star = np.random.choice(X, n)
    theta_star_non_parametric[i] = np.mean(X_star)
    
    X_star_misspec = np.random.choice(X_misspec, n)
    theta_star_non_parametric_misspec[i] = np.mean(X_star_misspec)

theta_hat = np.mean(X)
sigma2_hat = np.var(X)

theta_hat_misspec = np.mean(X_misspec)
sigma2_hat_misspec = np.var(X_misspec)

normal_approx = np.random.normal(theta_hat, np.sqrt(sigma2_hat/n), B)
normal_approx_misspec = np.random.normal(theta_hat_misspec, np.sqrt(sigma2_hat_misspec/n), B)


ax, fig = plt.subplots(1, 2, figsize=(10, 5))

fig[0].hist(theta_star_non_parametric, bins=100, alpha=0.5, label='Non-parametric bootstrap')
fig[0].hist(normal_approx, bins=100, alpha=0.5, label='Normal approximation')
fig[0].set_xlabel(r'$\hat \theta$')
fig[0].set_ylabel('Frequency')
fig[0].legend(loc='lower right', fontsize='small')
fig[0].set_title('Exponential (Correctly specified)')

fig[1].hist(theta_star_non_parametric_misspec, bins=100, alpha=0.5, label='Non-parametric bootstrap')
fig[1].hist(normal_approx_misspec, bins=100, alpha=0.5, label='Normal approximation')

fig[1].set_xlabel(r'$\hat \theta$')
fig[1].legend(loc='lower right', fontsize='small')
fig[1].set_title('Uniform (Misspecified)')
plt.savefig('figs/ex1c.pdf', format='pdf', dpi=300)
plt.show()

