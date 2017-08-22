import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
    def __init__(self, N, T, M, K, mu0, mu1, sigma0, sigma1, runs):
        self.N = N
        self.T = T
        self.M = M
        self.K = K
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.runs = runs
        self.times = []

    # Construct the NxT signal with K anomalous rows
    def get_signal(self, N, T, K):
        support = np.random.choice(N, K, replace=False)
        X = np.random.normal(self.mu0, self.sigma0, (N, T, 1))
        for k in support:
            X[k] = np.random.normal(self.mu1, self.sigma1, (T, 1))
        return X, support

    # Construct a Gaussian TxMxN measurement matrix
    def get_measurement_matrix(self, N, T, M):
        return np.random.normal(0, 1, (T, M, N))

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

    def run_experiment(self):
        """
        Records:
        1) The time it took for the simulation to run
        2) A file containing the recovery scores for the simulation
        3) A heat-map image of the recovery scores
        """
        idxT = np.arange(1, self.T + 1, 1)
        idxM = np.arange(1, self.M + 1, 1)
        m_scores = []
        m_times = []
        for m in idxM:
            time_m0 = time.time()
            t_scores = []
            t_times = []
            for t in idxT:
                time_t0 = time.time()
                success_count = 0
                for r in range(self.runs):
                    _, _, is_success = \
                        self.recover_support(self.N, t, m, self.K)
                    success_count += is_success
                t_scores.append(success_count / float(self.runs))
                time_t1 = time.time()
                t_times.append(time_t1-time_t0)
            time_m1= time.time()
            print(time_m1-time_m0)
            m_times.append(t_times)
            m_scores.append(t_scores)
        np.savetxt('times_N%s_T%s_M%s_K%s_runs%s.csv'%(
            self.N, self.T, self.M, self.K, self.runs),
                   m_times, delimiter=',')
        np.savetxt('scores_N%s_T%s_M%s_K%s_runs%s.csv'%(
            self.N, self.T, self.M, self.K, self.runs),
                   m_scores, delimiter=',')
        sns.heatmap(np.matrix(m_scores), square=True,
                    xticklabels=idxT,
                    yticklabels=idxM,
                    cbar=False)
        plt.xlabel('t')
        plt.ylabel('m')
        plt.title('K=%s' % self.K)
        plt.savefig('OSGA_N%s_T%s_M%s_K%s_runs%s' % (
            self.N, self.T, self.M, self.K, self.runs))


if __name__ == '__main__':
    for k in np.arange(1, 11, 1):
        MMVOSGA(100, 100, 100, k, 0, 7, 1, 1, 100).run_experiment()
