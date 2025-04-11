import numpy as np
import pandas as pd
from scipy.stats import bootstrap, chi2


np.random.seed(2025)

n = 20
B = 1000   
B2 = 1000
theta_0 = 1 


X = np.random.exponential(scale=theta_0, size=n)
theta_hat = np.mean(X)

X_misspec = np.random.uniform(low=0, high=2*theta_0, size=n)
theta_hat_misspec = np.mean(X_misspec)

theta_star = np.zeros(B)
theta_star_misspec = np.zeros(B)



R = np.zeros(B)
R_misspec = np.zeros(B)
for i in range(B):
    X_star = np.random.choice(X, n)
    
    X_star_mean = np.mean(X_star)
    theta_star[i] = X_star_mean
    
    X_star_misspec = np.random.choice(X_misspec, n)
    X_star_misspec_mean = np.mean(X_star_misspec)
    theta_star_misspec[i] = X_star_misspec_mean
    
    count = 0
    count_misspec = 0
    for j in range(B2):
        X_star_star = np.random.exponential(scale=np.mean(X_star), size=n)
        theta_star_star = np.mean(X_star_star)
        count += theta_star_star < theta_hat 
        
        X_star_misspec = np.random.exponential(scale=np.mean(X_star_misspec), size=n)
        theta_star_star_misspec = np.mean(X_star_misspec)
        count_misspec += theta_star_star_misspec < theta_hat_misspec
        
    
    R[i] = count / B
    R_misspec[i] = count_misspec / B


R.sort()
q1, q2 =np.quantile(R, [0.025, 0.975])

R_misspec.sort()
q1_misspec, q2_misspec =np.quantile(R_misspec, [0.025, 0.975])

    
theta_hat = np.mean(X)
theta_hat_misspec = np.mean(X_misspec)
CI_nested_boot = np.quantile(theta_star, [q1, q2])
CI_nested_boot_misspec = np.quantile(theta_star_misspec, [q1_misspec, q2_misspec])

sum_X = np.sum(X)

CI_theoretical_correct = (
    2 * sum_X / chi2.ppf(0.975, 2*n),
    2 * sum_X / chi2.ppf(0.025, 2*n)
)

std_misspec = theta_hat_misspec / np.sqrt(n)
CI_theoretical_misspec = (
    float(theta_hat_misspec - 1.96*std_misspec),
    float(theta_hat_misspec + 1.96*std_misspec)
)


results = []


results.append({
    "source": "theoretical",
    "key": "Correct",
    "method": "theoretical",
     r"CI$_{low}$": round(CI_theoretical_correct[0], 2),
     r"CI$_{high}$": round(CI_theoretical_correct[1], 2)
})
#append nested 
results.append({
    "source": "Nested boostrap",
    "key": "Correct",
    "method": "np choice and percentile",
     r"CI$_{low}$": round(CI_nested_boot[0], 2),
     r"CI$_{high}$": round(CI_nested_boot[1], 2)
})

methods = ['Percentile', 'BCa']
data = {'Correct': X, 'Misspecified': X_misspec}

for key, value in data.items():
    for method in methods:
        ci = bootstrap(
            data=(value,),
            statistic=np.mean,
            confidence_level=0.95,
            n_resamples=B,
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
    "key": "Misspecified",
    "method": "theoretical",
     r"CI$_{low}$": round(CI_theoretical_misspec[0], 2),
     r"CI$_{high}$": round(CI_theoretical_misspec[1], 2)
})


results.append({
    "source": "Nested boostrap",
    "key": "Misspecified",
    "method": "np choice and percentile",
     r"CI$_{low}$": round(CI_nested_boot_misspec[0], 2),
     r"CI$_{high}$": round(CI_nested_boot_misspec[1], 2)
})


df = pd.DataFrame(results)
print(df)
print(df.to_latex())