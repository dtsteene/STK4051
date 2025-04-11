import numpy as np
import pandas as pd
from scipy.stats import bootstrap

np.random.seed(2025)

n = 20
theta_0 = 1


X = np.random.exponential(scale=theta_0, size=n)


X_misspec = np.random.uniform(low=0, high=2*theta_0, size=n)

theta_star_nested = np.zeros(n)
theta_star_nested_misspec = np.zeros(n)

for i in range(n):
    X_star = np.random.choice(X, n)
    X_star_misspec = np.random.choice(X_misspec, n)

    for i in range(n):
        #parametric nested bootstrap, idk what we want here
        X_star = np.random.exponential(scale=np.mean(X_star), size=n)
        theta_star_nested[i] = np.mean(X_star)

        X_star_misspec = np.random.exponential(scale=np.mean(X_star_misspec), size=n)
        theta_star_nested_misspec[i] = np.mean(X_star_misspec)

theta_hat = np.mean(X)
theta_hat_misspec = np.mean(X_misspec)
CI_nested_boot = np.quantile(theta_star_nested, [0.025, 0.975])
CI_nested_boot_misspec = np.quantile(theta_star_nested_misspec, [0.025, 0.975])


std_correct = theta_hat / np.sqrt(n)
CI_theoretical_correct = (
    float(theta_hat - 1.96*std_correct), 
    float(theta_hat + 1.96*std_correct)
)

std_misspec = theta_hat_misspec / np.sqrt(n)
CI_theoretical_misspec = (
    float(theta_hat_misspec - 1.96*std_misspec),
    float(theta_hat_misspec + 1.96*std_misspec)
)


methods = ['Percentile', 'BCa']
data = {'Correct': X, 'Misspecified': X_misspec}
results = []

for key, value in data.items():
    for method in methods:
        ci = bootstrap(
            data=(value,),
            statistic=np.mean,
            confidence_level=0.95,
            n_resamples=n,
            method=method
        )
        results.append({
            "source": "bootstrap",
            "key": key,
            "method": method,
            r"CI$_{low}$": round(ci.confidence_interval.low, 2),
            r"CI$_{high}$": round(ci.confidence_interval.high, 2)
        })

# 
results.append({
    "source": "theoretical",
    "key": "Correct",
    "method": "theoretical",
     r"CI$_{low}$": round(CI_theoretical_correct[0], 2),
     r"CI$_{high}$": round(CI_theoretical_correct[1], 2)
})
results.append({
    "source": "theoretical",
    "key": "Misspecified",
    "method": "theoretical",
     r"CI$_{low}$": round(CI_theoretical_misspec[0], 2),
     r"CI$_{high}$": round(CI_theoretical_misspec[1], 2)
})

#append nested 
results.append({
    "source": "nested boostrap",
    "key": "Correct",
    "method": "np choice and percentile",
     r"CI$_{low}$": round(CI_nested_boot[0], 2),
     r"CI$_{high}$": round(CI_nested_boot[1], 2)
})

results.append({
    "source": "nested boostrap",
    "key": "Misspecified",
    "method": "np choice and percentile",
     r"CI$_{low}$": round(CI_nested_boot_misspec[0], 2),
     r"CI$_{high}$": round(CI_nested_boot_misspec[1], 2)
})


df = pd.DataFrame(results)
print(df)
print(df.to_latex())