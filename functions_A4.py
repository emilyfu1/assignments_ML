import numpy as np
import dask
from dask import delayed, compute

def bernoulli_bandit_thompson(T, theta, seed=420):
    rng = np.random.default_rng(seed)

    # get the amount of treatments from their true expected outcomes
    k = len(theta)
    
    # initialise arrays to store the selected treatments and observed outcomes
    D_t = np.zeros(T, dtype=int)
    Y_t = np.zeros(T)
    
    # the posterior for theta_d at time t + 1 is a Beta distribution with parameters
    # start with uniform prior over theta on [0, 1]^k
    # 1 + number of successes
    alpha = np.ones(k)
    # 1 + number of failures
    beta = np.ones(k)
    
    for t in range(T):
        # sample from the Beta distribution for each arm
        samples = rng.beta(alpha, beta)
        
        # select the treatment with the highest sample
        action = np.argmax(samples)
        D_t[t] = action + 1
        
        # assume that outcome follows a Bernoulli distribution with parameter theta[action]
        outcome = rng.binomial(1, theta[action])
        Y_t[t] = outcome
        
        # update alpha and beta parameters of posterior based on the observed outcome
        if outcome:
            # another success
            alpha[action] += 1
        else:
            # not success
            beta[action] += 1
            
    return D_t, Y_t

def calculate_replication(T, theta, seed=420):
    # format of theta
    theta = np.asarray(theta, dtype=float)

    # get the actions and outcomes for T time steps
    D_t, Y_t = bernoulli_bandit_thompson(T, theta, seed=seed)

    # best treatment and its theta value
    best_arm = np.argmax(theta)
    best_theta = np.max(theta)
    
    # get the theta value of the selected arm at each time step
    theta_Dt = theta[D_t - 1]  # fix python array indexing
    optimal_indicator = (D_t - 1 == best_arm).astype(float)  # fix python array indexing
    # calculate regret at each time step
    regret_t = best_theta - theta_Dt

    return Y_t.astype(float), theta_Dt, optimal_indicator, regret_t

def evaluate_bandit(T, theta, R, seed=420):
    # format of theta
    theta = np.asarray(theta, dtype=float)

    # actually do it for the R replications
    tasks = [delayed(calculate_replication)(T, theta, seed + r) for r in range(R)]

    # compute the results in parallel using dask
    results = compute(*tasks, scheduler='threads')

    # aggregate results across replications
    Y_all = np.array([res[0] for res in results])
    thetaDt_all = np.array([res[1] for res in results])
    opt_all = np.array([res[2] for res in results])
    regret_all = np.array([res[3] for res in results])

    # averagevalues of Y_t, theta_Dt, optimal indicator, and regret_t across the R replications
    avg_Y_t = Y_all.mean(axis=0)
    avg_theta_Dt = thetaDt_all.mean(axis=0)
    avg_optimal = opt_all.mean(axis=0)
    avg_regret_t = regret_all.mean(axis=0)

    return avg_Y_t, avg_theta_Dt, avg_optimal, avg_regret_t
