
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def block_bootstrap(
    X: np.ndarray, 
    block_length: int, 
    B: int, 
    statistic=np.mean):
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
    n = len(X)
    num_blocks = int(np.ceil(n / block_length))
    
    all_blocks = np.array(
        [X[i:i+block_length] for i in range(n - block_length + 1)]
    )
    block_indices = np.random.randint(
        0, all_blocks.shape[0], size=(B, num_blocks)
    )
    bootstrap_samples = all_blocks[block_indices].reshape(B, -1)  # (B, num_blocks * block_length)
    estimates = statistic(bootstrap_samples, axis=1) 
  
    return estimates



def block_bootstrap_on_block_deleted_dataset(
    X: np.ndarray, 
    block_length: int, 
    d: int, i :int, B : int, 
    statistic=np.mean
    ):
    """
    Performs a block bootstrap on the time series X with the given block length,
    excluding the i-th block and d blocks around it.
    Parameters:
        X: array-like, the original time series data.
        block_length: int, the length of each block.
        d: int, number of blocks to exclude around the i-th block.
        i: int, index of the block to exclude.
        B: int, number of bootstrap replicates.
        statistic: function, the function to compute the statistic (default is np.mean).
    Returns:
        B bootstrap estimates of the statistic.
    """
    X = np.asarray(X)
    n = len(X)
    num_blocks = int(np.ceil(n / block_length))
    
    all_blocks = np.array(
        [X[j:j+block_length] for j in range(n - block_length + 1)]
    )
    # exclude blocks 
    exclude_indices = np.arange(i-1, i-1 + d)
    valid_indices = np.setdiff1d(
        np.arange(len(all_blocks)), exclude_indices
    )
    
    if len(valid_indices) == 0:
        raise ValueError("No blocks remain after deletion.")
    
    block_indices = np.random.choice(
        valid_indices, size=(B, num_blocks), replace=True
        )
    bootstrap_samples = all_blocks[block_indices].reshape(B, -1)
    estimates = statistic(bootstrap_samples, axis=1) 
    
    return estimates

def block_bootstrap_on_element_deleted_dataset(
    X:np.ndarray,
    block_length:int,
    d:int, 
    i:int,
    B:int, 
    statistic=np.mean
):
    """
    Performs a block bootstrap on the time series X with the given block length,
    excluding the i-th element and d elements around it.
    Parameters:
        X: array-like, the original time series data.
        block_length: int, the length of each block.
        d: int, number of elements to exclude around the i-th element.
        i: int, index of the element to exclude.
        B: int, number of bootstrap replicates.
        statistic: function, the function to compute the statistic (default is np.mean).
    Returns:
        B bootstrap estimates of the statistic.
    """
    X_reduced = np.delete(X, np.s_[i-1 : i-1+d])
    return block_bootstrap(X_reduced, block_length, B, statistic)

def jackknife_plus_bootstrapping(
    X: np.ndarray, 
    B=1000, 
    statistic=np.mean, 
    sub_statistic=np.var,
    l0=None, 
    d=None
    ):
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

    phi_l0 = sub_statistic(
        block_bootstrap(X, l0, B, statistic)
    )
    
    phi_2l0 = sub_statistic(
        block_bootstrap(X, 2 * l0, B, statistic)
    )
    
    B_hat = 2 * (phi_l0 - phi_2l0)
    
    if d is None:
        d = int(round(n**(1/3) * l0**(2/3)))
    
    num_deletions = n - l0 - d + 2
    phi_tilde_list = []
    
    for i in range(1, num_deletions + 1):
       
        phi_i = sub_statistic(
            block_bootstrap_on_element_deleted_dataset(
                X, l0, d, i, B, statistic
            )
        )
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
    
    numerator = np.sum(
        (X[lag:] - mean_X) * (X[:-lag] - mean_X)
    )
    denominator = np.sum((X - mean_X) ** 2)
    
    return numerator / denominator

def vectorized_autocorrelation(X, lag=1, axis=1):
    """
    Computes the autocorrelation at a specified lag for each row in X.
    
    Parameters:
        X: array-like, shape (B, n) or (n,)
           If X is 1D, it is converted to a single-row 2D array.
        lag: int, the lag at which to compute autocorrelation.

    Returns:
        autocorr: np.ndarray, shape (B,)
            The autocorrelation of each row at the specified lag.
    """
    X = np.atleast_2d(X)
    
    mean = np.mean(X, axis=axis, keepdims=True)
    
    numerator = np.sum(
        (X[:, lag:] - mean) * (X[:, :-lag] - mean),
        axis=axis
    )
    
    # row-wise total squared deviations
    denominator = np.sum((X - mean) ** 2, axis=axis)
    
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
    blocks = np.array(
        [X[i *l:(i + 1) * l] for i in range(num_blocks)]
    )
    return blocks


def test_block_bootstrap():
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
    
    
    l_opt_var_mean = jackknife_plus_bootstrapping(
        X, B=1000, statistic=np.mean, d=d, sub_statistic=np.var
    )
    l_opt_bias_mean = jackknife_plus_bootstrapping(
        X, B=1000, statistic=np.mean, d=d, sub_statistic=bias
    )
    l_opt_var_var = jackknife_plus_bootstrapping(
        X, B=1000, statistic=np.var, d=d, sub_statistic=np.var
    )
    l_opt_bias_var = jackknife_plus_bootstrapping(
        X, B=1000, statistic=np.var, d=d, sub_statistic=bias
    )
    
    
    if disp:
        print(f"Optimal block length for variance of mean: {l_opt_var_mean}")
        print(f"Optimal block length for bias of mean: {l_opt_bias_mean}")
        print(f"Optimal block length for variance of variance: {l_opt_var_var}")
        print(f"Optimal block length for bias of variance: {l_opt_bias_var}")
    
    return l_opt_var_mean, l_opt_bias_mean , l_opt_var_var, l_opt_bias_var


def ex_c():
    #test b for a bunch of different d
    
    d = list(range(1, 70))
    l_opt_var_mean = []
    l_opt_bias_mean = []
    l_opt_var_var = []
    l_opt_bias_var = []
    
    for i in tqdm(d):
        l_opt_var_mean_i, l_opt_bias_mean_i, l_opt_var_var_i, l_opt_bias_var_i = ex_b(d=i, disp=False)
        l_opt_var_mean.append(l_opt_var_mean_i)
        l_opt_bias_mean.append(l_opt_bias_mean_i)
        l_opt_var_var.append(l_opt_var_var_i)
        l_opt_bias_var.append(l_opt_bias_var_i)
    #plot the results
    plt.plot(d, l_opt_var_mean, label='Variance of mean')
    plt.plot(d, l_opt_bias_mean, label='Bias of mean')
    plt.plot(d, l_opt_var_var, label='Variance of variance')
    plt.plot(d, l_opt_bias_var, label='Bias of variance')
    plt.xlabel('d')
    plt.ylabel('Optimal block length')
    plt.title('Optimal block length for bias and variance (Element deletion)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    plt.grid()
    plt.savefig('figs/ex2c_elm.pdf', format='pdf', dpi=300)
    plt.show()
    
    
    
def ex_d():
    df = pd.read_csv('discoveries.csv')
 
    X = np.array(df['Discoveries'].values)
    X_2_blocks = set_of_blocks(X, 2)
    autocorrelation_global = autocorrelation(X, lag=1)
    
    def bias_autocorr(x, axis=0):
        return np.mean(x - autocorrelation_global, axis=axis)
    
    l_opt_bias = jackknife_plus_bootstrapping(
        X_2_blocks, B=1000, statistic=vectorized_autocorrelation, 
        sub_statistic=bias_autocorr
    )
    l_opt_var = jackknife_plus_bootstrapping(
        X_2_blocks, B=1000, statistic=vectorized_autocorrelation, 
    sub_statistic=np.var
    )
    
    var_autocorr = np.var(
        block_bootstrap(X_2_blocks, l_opt_var, B=1000, statistic=vectorized_autocorrelation), 
        axis=0
    )
    bias_autocorr = bias_autocorr(
    block_bootstrap(X_2_blocks, l_opt_bias, B=1000, statistic=vectorized_autocorrelation), axis=0
    )
    
    print("1-lag autocorrelation:", autocorrelation_global)
    print("Bias of 1-lag autocorrelation:", bias_autocorr)
    print("Variance of 1-lag autocorrelation:", var_autocorr)
    

    
    
if __name__ == "__main__":
    #set seed
    np.random.seed(2025)

    #test_block_bootstrap()
    #quit()
    #read csv using pandas
    ex_d()
    ex_b()
    ex_c()