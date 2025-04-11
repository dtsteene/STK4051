
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def block_bootstrap(X, block_length, B, statistic=np.mean):
    """
    Performs a block bootstrap on the entire time series X with the given block length.
    Parameters:
        X: array-like, the original time series data.
        block_length: int, the length of each block.
        B: int, number of bootstrap replicates.
        statistic: function, the function to compute the statistic (default is np.mean).
    Returns:
        B bootstrap estimates of the statistic.
    """
   
    X = np.asarray(X)
    n = X.shape[0]
   
    num_blocks = int(np.ceil(n / block_length))
    
    all_blocks = np.array([X[i:i+block_length] for i in range(n - block_length + 1)])
    
    #grab B random blocks num_blocks times 
    block_indices = np.random.randint(0, all_blocks.shape[0], size=(B,num_blocks))
    bootstrap_samples = all_blocks[block_indices].reshape(-1, B)
    
    estimates = statistic(bootstrap_samples, axis=0)
    return estimates


def block_bootstrap_on_block_deleted_dataset(X, block_length, d, i, B, statistic=np.mean, sub_statistic=np.var):
    """
    Performs block bootstrap on the dataset with a block deleted.
    Really unsure if we are supposed to delete d blocks or just d elements. It
    says d blocks in the textbook, but the indexation seems to imply d elements.
    !!I'm deleting d elements!!
    """
    X_reduced = np.delete(X, np.s_[i-1 : i-1+d])
    return block_bootstrap(X_reduced, block_length, B, statistic, sub_statistic)

def jackknife_plus_bootstrapping(X, B=1000, statistic=np.mean, sub_statistic=np.var,l0=None, d=None):
    """
    Implements the Jackknife Plus Bootstrapping algorithm for estimating an optimal block length.
    Parameters:
        X: array-like, the input time series data.
        B: int, the number of bootstrap replicates to use.
        statistic: function, the statistic computed over bootstrap replicates.
    Returns:
        l_opt: int, the estimated optimal block length.
    """
    n = len(X)
    if l0 is None:
        l0 = int(round(n**(1/5)))

    phi_l0 = sub_statistic(block_bootstrap(X, l0, B, statistic))
    
    phi_2l0 = sub_statistic(block_bootstrap(X, 2 * l0, B, statistic))
    
    B_hat = 2 * (phi_l0 - phi_2l0)
    
    if d is None:
        d = int(round(n**(1/3) * l0**(2/3)))
    
    num_deletions = n - l0 - d + 2
    phi_tilde_list = []
    
    for i in range(1, num_deletions + 1):
       
        phi_i = sub_statistic(block_bootstrap_on_block_deleted_dataset(X, l0, d, i, B, statistic))
        phi_tilde = (((n - l0 + 1) * phi_l0 - (n - l0 - d - 1) * phi_i) / d)
        phi_tilde_list.append(phi_tilde)
    
    phi_tilde_array = np.array(phi_tilde_list)

    V_hat = (d / (n - l0 - d + 1)) * (1 / (n - l0 - d + 2)) * np.sum((phi_tilde_array - phi_l0) ** 2)

    c1_hat = (n ** 3 / l0) * V_hat
    
    c2_hat = n * l0 * B_hat
    
    l_opt = int(round(((2 * c2_hat**2 / c1_hat) ** (1/3)) * (n ** (1/3))))
    
    return l_opt

def autocorrelation(X, lag=1):
    """
    Computes the autocorrelation of a time series X at a given lag.
    Parameters:
        X: array-like, the time series data.
        lag: int, the lag at which to compute the autocorrelation.
    Returns:
        The autocorrelation value at the specified lag.
    """
    X = X.reshape(-1)
    mean_X = np.mean(X)
    
    numerator = np.sum((X[lag:] - mean_X) * (X[:-lag] - mean_X))
    denominator = np.sum((X - mean_X) ** 2)
    
    return numerator / denominator



def set_of_blocks(X, l):
    """Creates a set of blocks from the time series X, for
    a given block length. Used for block of blocks bootstrap.

    Args:
        X: array-like, the input time series data.
        l: int, the estimated optimal block length.
    """
    n = len(X)
    num_blocks = n // l
    blocks = np.array([X[i *l:(i + 1) * l] for i in range(num_blocks)])
    return blocks


def test_block_bootstrap():
    np.random.seed(2025)
    n = 1000
    block_length = 10
    B = 1000
    X = np.random.normal(size=n)
    
    # Test the block bootstrap function
    bootstrap_var_of_mean = block_bootstrap(X, block_length, B)
    print("Bootstrap variance of the mean:", bootstrap_var_of_mean)

def ex_b(d=None, disp=True):
    #test_block_bootstrap()
    #read csv using pandas
   
    df = pd.read_csv('discoveries.csv')
 
    X = np.array(df['Discoveries'].values)
    mean_X = np.mean(X)
    
    def bias(x, axis=0):
        return np.mean(x - mean_X, axis=axis)
    
    
    l_opt_bias = jackknife_plus_bootstrapping(X, B=1000, statistic=np.var, d=d, sub_statistic=bias)
    
    l_opt_bias = jackknife_plus_bootstrapping(X, B=1000, statistic=bias, d=d)
    l_opt_mean = jackknife_plus_bootstrapping(X, B=1000, statistic=np.mean, d=d)
    
    if disp:
        print("Optimal block length (variance):", l_opt_var)
        print("Optimal block length (bias):", l_opt_bias)
        print("Optimal block length (mean):", l_opt_mean)

    
    return l_opt_var, l_opt_bias,  l_opt_mean


def ex_c():
    #test b for a bunch of different d
    
    d = list(range(1, 60))
    l_opt_var = []
    l_opt_bias = []
    l_opt_mean = []
    
    for i in d:
        l_opt_var_i, l_opt_bias_i, l_opt_mean_i = ex_b(d=i, disp=False)
        l_opt_var.append(l_opt_var_i)
        l_opt_bias.append(l_opt_bias_i)
        l_opt_mean.append(l_opt_mean_i)
    return l_opt_var, l_opt_bias, l_opt_mean
    
    
def ex_d():
    df = pd.read_csv('discoveries.csv')
 
    X = np.array(df['Discoveries'].values)
    X_2_blocks = set_of_blocks(X, 2)
    l_opt_var = jackknife_plus_bootstrapping(X_2_blocks, B=1000, statistic=np.var)
    l_opt_bias = jackknife_plus_bootstrapping(X_2_blocks, B=1000, statistic=np.mean)
    #use l_opt
    block_bootstrap(X_2_blocks, l_opt_var, B=1000, statistic=autocorrelation)
    block_bootstrap(X_2_blocks, l_opt_bias, B=1000, statistic=autocorrelation)
    
    
    
    
if __name__ == "__main__":

    #test_block_bootstrap()
    #quit()
    #read csv using pandas
    ex_d()
    quit()
    #ex_b()
    
    l_opt_var, l_opt_bias, l_opt_mean = ex_c()
    
    plt.plot(l_opt_var, label='variance')
    plt.plot(l_opt_bias, label='bias')
    plt.plot(l_opt_mean, label='mean')
    plt.xlabel('d')
    plt.ylabel('Optimal block length')
    plt.title('Optimal block length for bias and variance')
    plt.legend()
    plt.savefig('figs/ex2b.pdf', format='pdf', dpi=300)
    plt.show()