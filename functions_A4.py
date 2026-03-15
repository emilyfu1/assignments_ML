import numpy as np
import dask
from dask import delayed, compute

# question 1a, question 2a
def bernoulli_bandit(T, theta, seed=420, M=1000, exploration=False):
    # for generating from distributions
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
        # case for exploration sampling
        if exploration:
            # sample from the Beta distribution for each treatment
            draws = rng.beta(alpha, beta, size=(M, k))
            # but now need to actually get a distribution i need to like approximate it
            # find which treatment is best in each simulated world
            best_arms = np.argmax(draws, axis=1)
            # count how often each treatmnet is the largest, use those frequencies as an estimate for p
            p_t = np.bincount(best_arms, minlength=k) / M
            # print(p_t)

            # calculate q_t for exploration sampling
            weights = p_t * (1 - p_t)
            denom = weights.sum()
            
            # in the limit, half the observations are assigned to the best treatment
            if denom > 0:
                q_t = weights / denom
            # here, one of the p_t is 1 and the rest are 0
            # so simulated posterior probabilities have already converged to putting all the weight on one treatment
            else:
                # put all the weight on the best treatment, but still explore the other treatments a little bit
                best_arm = np.argmax(p_t)
                q_t = np.ones(k) * (0.5 / (k - 1))
                q_t[best_arm] = 0.5

            # choose with probabilities given by q_t instead of directly from the beta distribution
            action = rng.choice(k, p=q_t)
            D_t[t] = action
        # thompson sampling
        else:
            # sample from the Beta distribution for each treatment
            samples = rng.beta(alpha, beta)
            # print(samples)
            
            # find which treatment gave the highest sample and select that treatment
            action = np.argmax(samples)
            # store which treatment this corresponds to
            D_t[t] = action
        
        # assume that outcome follows a Bernoulli distribution with parameter theta[action]
        outcome = rng.binomial(1, theta[action])
        Y_t[t] = outcome
        # print(outcome)
        
        # update alpha and beta parameters of posterior based on the observed outcome
        if outcome == 1:
            # another success
            alpha[action] += outcome
        else:
            # not success
            beta[action] += 1 - outcome
    
    # mean for each treamtnet after period T (beta distribution expected value)
    posterior_means = alpha / (alpha + beta)

    # treatment with highest posterior mean at time T
    d_highest_posterior_mean = np.argmax(posterior_means)
    
    # hash return
    return ({'D_t': D_t, 
             'Y_t': Y_t, 
             'd_highest_posterior_mean': d_highest_posterior_mean})

# question 1b
def calculate_replication(T, theta, seed=420, exploration=False):
    # format of theta
    theta = np.asarray(theta, dtype=float)

    # get the actions and outcomes for T time steps
    result = bernoulli_bandit(T, theta, seed=seed, exploration=exploration)
    D_t = result['D_t']
    Y_t = result['Y_t']
    d_highest_posterior_mean = result['d_highest_posterior_mean']

    # best treatment and its theta value
    best_arm = np.argmax(theta)
    best_theta = np.max(theta)
    
    # get the theta value of the selected treatment at each time step
    theta_Dt = theta[D_t]
    # how many times the optimal treatment was selected at each time step
    optimal_indicator = (D_t == best_arm).astype(float)
    # calculate regret at each time step
    regret_t = best_theta - theta_Dt

    # calculate policy regret at time T
    policy_regret = best_theta - theta[d_highest_posterior_mean]
    # indicator for whether the policy at time T selects the best treatment
    best_arm_policy_indicator = float(d_highest_posterior_mean == best_arm)

    return ({'Y_t': Y_t,
             'theta_Dt': theta_Dt,
             'optimal_indicator': optimal_indicator,
             'regret_t': regret_t,
             'd_highest_posterior_mean': d_highest_posterior_mean,
             'policy_regret': policy_regret,
             'best_arm_policy_indicator': best_arm_policy_indicator})

# question 1b
def evaluate_bandit(T, theta, R, seed=420, exploration=False):
    # format of theta
    theta = np.asarray(theta, dtype=float)

    # actually do it for the R replications
    tasks = [delayed(calculate_replication)(T, theta, seed + r, exploration) for r in range(R)]

    # compute the results in parallel using dask
    results = compute(*tasks, scheduler='threads')

    # aggregate results across replications
    Y_all = np.array([res['Y_t'] for res in results])
    thetaDt_all = np.array([res['theta_Dt'] for res in results])
    opt_all = np.array([res['optimal_indicator'] for res in results])
    regret_all = np.array([res['regret_t'] for res in results])
    d_star_all = np.array([res['d_highest_posterior_mean'] for res in results])
    policy_regret_all = np.array([res['policy_regret'] for res in results])
    best_arm_policy_all = np.array([res['best_arm_policy_indicator'] for res in results])

    # averagevalues of Y_t, theta_Dt, optimal indicator, and regret_t across the R replications
    avg_Y_t = Y_all.mean(axis=0)
    avg_theta_Dt = thetaDt_all.mean(axis=0)
    avg_optimal = opt_all.mean(axis=0)
    avg_regret_t = regret_all.mean(axis=0)

    # cumulative average regret and average of cumulative average regret across the R replications
    avg_cum_regret_t = np.cumsum(avg_regret_t)
    avg_cumavg_regret_t = avg_cum_regret_t / np.arange(1, T + 1)

    # probability of choosing the best treatment
    avg_policy_regret = policy_regret_all.mean()
    prob_best_arm_policy = best_arm_policy_all.mean()

    return ({'avg_Y_t': avg_Y_t,
             'avg_theta_Dt': avg_theta_Dt,
             'avg_optimal': avg_optimal,
             'avg_regret_t': avg_regret_t,
             'avg_cum_regret_t': avg_cum_regret_t,
             'avg_cumavg_regret_t': avg_cumavg_regret_t,
             'avg_policy_regret': avg_policy_regret,
             'prob_best_arm_policy': prob_best_arm_policy,
             'd_star_all': d_star_all})