import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.stats.proportion import proportion_confint

"""
N Number of random variables in the signal
T Number of time-steps to measure
M Number of measurements per time-step
K Number of anomalous random variables
mu0 mean of the null distributions
mu1 mean of the anomalous distribution
sigma0 variance of the null distribution
sigma1 variance of the anomalous distribution
"""


class MMVOSGA:
    def __init__(self, N, T, M, K, mu0, mu1, sigma0, sigma1,
                 thresh, conf=True, t_var=True):
        self.N = N
        self.T = T
        self.M = M
        self.K = K
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.idxT = np.arange(1, self.T + 1, 1)
        self.idxM = np.arange(1, self.M + 1, 1)

        # True if we should use a binomial proportion
        # confidence interval as a stopping condition
        # False if we should just use a hard cutoff
        self.conf = conf

        # TODO: Reject threshold if not in (0,1) when conf=True
        # TODO: Reject threshold if not > 1 when conf=False
        # Int number of runs if conf=False,
        # else float in (0,1): interval length tolerance
        self.thresh = thresh

        # True if we should use time-varying measurements
        # False if we should use fixed-time measurements
        self.t_var = t_var


# Stopping condition
    def keep_going(self, successes, trials):
        # Based on binomial proportion confidence interval
        if self.conf:
            low, high = proportion_confint(successes, trials,
                                           method='jeffreys')
            return high - low > self.thresh
        # Based on a hard threshold
        else:
            return trials < self.thresh

    # Construct the NxT signal with K anomalous rows
    def get_signal(self, N, T, K):
        support = np.random.choice(N, K, replace=False)
        X = np.random.normal(self.mu0, self.sigma0, (N, T, 1))
        for k in support:
            X[k] = np.random.normal(self.mu1, self.sigma1, (T, 1))
        return X, support

    # Construct a Gaussian TxMxN measurement matrix
    def get_measurement_matrix(self, N, T, M):
        # If time varying measurements, then generate a different measurement
        # matrix for each time-step
        if self.t_var:
            return np.random.normal(0, 1, (T, M, N))
        # If fixed time measurements, then reuse the same measurement
        # matrix for each time-step
        else:
            m = np.random.normal(0, 1, (M, N))
            return np.array([m for t in range(T)])

    # Run a single signal recovery experiment and return the results
    def recover_support(self, N, T, M, K):
        X, support = self.get_signal(N, T, K)
        phi = self.get_measurement_matrix(N, T, M)
        y = np.array([np.dot(phi[t], X[:, t]) for t in range(T)])
        xi = np.array([1 / float(T) * np.sum(
            [np.dot(y[t].T, phi[t, :, n]) ** 2
             for t in range(T)])
                       for n in range(N)])
        support = set(support)
        support_predicted = set(xi.argsort()[-K:][::-1])
        return support, support_predicted, int(support == support_predicted)

    def record_experiment(self, m_times, m_scores, m_runs):
        """
        Records:
        -) The number of runs taken to return the success proportion
        -) The time it took for the simulation to run
        -) A file containing the recovery scores for the simulation
        -) A heat-map image of the recovery scores
        """
        np.savetxt('results/runs_N%s_T%s_M%s_K%s_runs%s_tv%s.csv' % (
            self.N, self.T, self.M, self.K, self.thresh, self.t_var),
                   m_runs, delimiter=',')
        np.savetxt('results/times_N%s_T%s_M%s_K%s_runs%s_tv%s.csv' % (
            self.N, self.T, self.M, self.K, self.thresh, self.t_var),
                   m_times, delimiter=',')
        np.savetxt('results/scores_N%s_T%s_M%s_K%s_runs%s_tv%s.csv' % (
            self.N, self.T, self.M, self.K, self.thresh, self.t_var),
                   m_scores, delimiter=',')
        sns.heatmap(np.matrix(m_scores), square=True,
                    xticklabels=self.idxT,
                    yticklabels=self.idxM,
                    cbar=True)
        plt.xlabel('t')
        plt.ylabel('m')
        plt.title('K=%s' % self.K)
        plt.savefig('results/OSGA_N%s_T%s_M%s_K%s_runs%s_tv%s.pdf' % (
            self.N, self.T, self.M, self.K, self.thresh, self.t_var))

    def run_experiment(self):
        m_times = []
        m_scores = []
        m_runs = []
        for m in self.idxM:
            time_m0 = time.time()
            t_times = []
            t_scores = []
            t_runs = []
            for t in self.idxT:
                time_t0 = time.time()
                success_count = 0
                trial_count = 0
                keep_going = True
                while keep_going:
                    _, _, is_success = \
                        self.recover_support(self.N, t, m, self.K)
                    trial_count += 1
                    success_count += is_success
                    keep_going = self.keep_going(success_count, trial_count)
                t_runs.append(trial_count)
                t_scores.append(success_count / float(trial_count))
                time_t1 = time.time()
                t_times.append(time_t1 - time_t0)
            time_m1 = time.time()
            print(m, time_m1 - time_m0)
            m_runs.append(t_runs)
            m_scores.append(t_scores)
            m_times.append(t_times)
        self.record_experiment(m_times, m_scores, m_runs)


if __name__ == '__main__':
    MMVOSGA(100, 100, 100, 1, 0, 7, 1, 1, 0.1,
            conf=True, t_var=True).run_experiment()
