# Implementation of meta-medoids learning algorithm.
# Builds on metalearner, but identifies medoids and computes based on these

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import entropy


class MetaMedoids(object):
    def __init__(self, policy, obs_shape, action_shape, K=10, num_iters=100,
                 num_medoids=6, medoid_size=10, lr=1e-2):
        self.algo = policy(0, obs_shape, action_shape)
        # Used for initializing new policies
        self.policy = policy
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        # Update with list of dict objects keyed by 'trajectory' and 'policy', keyed by training iteration
        self.memory = {}
        self.K = K  # how few can we go?
        self.num_iters = num_iters  # how many training iters per episode

        self.num_medoids = num_medoids  # Number of medoids to find
        self.medoid_size = medoid_size  # Number of points to calcualte medoids with

        self.episode = 0  # current episode
        self.KDEs = {}  # KDE for entirety of this update iteration

        self.env = None  # environment
        self.episode_len = 100
        self.optimizer = None  # specify Adam or SGD
        self.gamma = lr

        self.current_policy = None
        self.meta_divergences = {}  # Also start at 1 (should match memory)
        self.meta_samples = {}
        self.meta_occupancies = {}

    def update_meta_property(self, property, item, iteration):
        """Due to indexing, helper method to help update meta properties"""
        try:
            property[iteration].append(item)
        except:
            property[iteration] = [item]

    def sample(self, policy):  # tasks are entity types encountered
        obs_n = self.env.reset()
        observations = []
        for _ in range(self.episode_len):
            ep_reward = 0
            act_n = []
            act_n.append(policy.action(obs_n[0]))
            obs_n, reward_n, done_n, _ = self.env.step(act_n)
            policy.rewards.append(reward_n[0])
            ep_reward += reward_n[0]
            observations.append(obs_n[0])
        policy.finish_episode(self.optimizer, self.gamma)
        return observations, ep_reward

    def adapt(self, policy, optimizer, K):
        # After sampling multiple times, get updates
        policy.update(optimizer, K)
        # self.current_policy.update(self.optimizer, self.K)

    def train(self, K, iter_num):
        """
        Run k updates (K or 1), adapt, and run another, saving this trajectory and the adapted model
        :iter_num: the current iteration (used for indexing)
        """
        if K > 1:  # K + 1 shots overall?
            for _ in range(K):
                self.sample(self.current_policy)
            self.adapt(self.current_policy, self.optimizer,
                       self.K)  # policy.update(optimizer, K)
            # new trajectory with updated policy
            trajectory, ep_reward = self.sample(self.current_policy)
            kde = self.calculate_KDE(trajectory)
            update = {'trajectory': trajectory,
                      'policy': self.current_policy,
                      'kde': kde,
                      'reward': ep_reward,
                      'optimizer': self.optimizer,
                      'update': iter_num}
            try:
                self.memory[iter_num].append(update)  # could really be a list
            except:
                self.memory[iter_num] = [update]
        else:
            trajectory = self.sample(policy)
        return trajectory, ep_reward

    def update(self):
        """Update and find nearest neighbor from before to initialize at"""
        new_trajectory = self.train(policy)

    def calculate_KDE(self, states, bw=np.logspace(-1, 1, 20), cv=5):
        """Given state trajectories, calculate KDE"""
        params = {'bandwidth': bw}
        grid = GridSearchCV(KernelDensity(), params, cv=cv)
        grid.fit(states)
        kde = grid.best_estimator_
        return kde

    def sample_kde(self, kde, num_samples=100, seed=0):
        """Sample datapoints from given kde"""
        return kde.sample(num_samples, random_state=seed)

    def calculate_JSD(self, p, q, base=np.e):
        """Calculate Jensen-Shannon divergence"""
        p, q = np.asarray(p), np.asarray(q)
        p, q = p / p.sum(), q / q.sum()  # normalize to probabilities
        m = 1. / 2 * (p + q)
        return entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.

    def calculate_occupancy(self, kde, policy, sample):
        state_freq = np.exp(kde.score(sample.reshape(1, -1)))
        _, action_freq = policy.action(sample, distribution=True)
        return action_freq.detach() * state_freq

    def calculate_KNN(self, K, policy, trajectory, iteration, sample_num=100):
        """Return K nearest policies for a given policy and training iteration"""
        kde_policy = self.calculate_KDE(trajectory)
        kde_all = self.KDEs[iteration]

        saved_policies = [i['policy'] for i in self.memory[iteration]]
        samples = self.meta_samples[iteration]

        sample_divergences = []
        for sample_ix, sample in enumerate(samples):
            om = self.calculate_occupancy(kde_policy, policy, sample)
            sample_divergence = np.zeros(len(saved_policies))
            for n in range(len(saved_policies)):
                saved_om = self.meta_occupancies[iteration][n][sample_ix]
                jsd = self.calculate_JSD(om.squeeze(), saved_om.squeeze())
                sample_divergence[n] = jsd

            sample_probs = np.exp(kde_all.score(sample.reshape(1, -1)))
            sample_divergences.append(sample_divergence * sample_probs)

        divergence = np.sum(sample_divergences, axis=0)
        closest_ixs = divergence.argsort()[:K]
        return closest_ixs

    def get_updated_policies(self, K, policy, trajectory, iteration,
                             sample_num=100):
        closest_ixs = self.calculate_KNN(
            K, policy, trajectory, iteration, sample_num)
        policies = []
        new_iteration = len(self.memory)
        updated_policies = self.memory[new_iteration]
        for ix in range(len(updated_policies)):
            if ix in closest_ixs:
                policies.append(updated_policies[ix])
        return policies  # pick the one with the best reward?

    def calculate_distances(self, iterations=[1]):
        """
        Given training iterations, calculate nearby policies for all policies
        :iterations: list of iterations, default consider initial
        """
        for i in iterations:
            saved_policies = self.memory[i]
            # Load relevant saved experiences
            trajectories = []
            policies = []
            kdes = []

            for p in saved_policies:
                trajectories.extend(p['trajectory'])
                policies.append(p['policy'])
                kdes.append(p['kde'])

            print(trajectories)

            num_policies = len(policies)

            kde_all = self.calculate_KDE(trajectories)
            samples = self.sample_kde(kde_all, 100)

            self.KDEs[i] = kde_all
            self.meta_samples[i] = samples

            sample_divergences = []

            for sample_ix, sample in enumerate(samples):
                probs_n = []
                sample_divergence = np.zeros([num_policies, num_policies])
                for ix in range(num_policies):
                    occupancy_measure = self.calculate_occupancy(
                        kdes[ix], policies[ix], sample)
                    probs_n.append(occupancy_measure)
                    # Save occupancy measures for comparison later
                    key = sample_ix

                    # om_stat = {'-'.join(map(str, sample)): occupancy_measure}

                    try:
                        self.meta_occupancies[i][ix][key] = occupancy_measure
                    except:
                        try:
                            self.meta_occupancies[i][ix] = {
                                key: occupancy_measure}
                        except:
                            self.meta_occupancies[i] = {
                                ix: {key: occupancy_measure}}

                for a in range(num_policies):
                    for b in range(num_policies):
                        sample_divergence[a][b] = self.calculate_JSD(
                            probs_n[a].squeeze(), probs_n[b].squeeze())

                sample_prob = np.exp(kde_all.score(sample.reshape(1, -1)))
                sample_divergences.append(sample_divergence * sample_prob)
            # Compute average divergence over entirety
            divergence = np.sum(sample_divergences, axis=0)  # or sum?
            self.meta_divergences[i] = divergence

    def update_medoids(self, iteration, k=6):
        """Update medoids"""
        divergences = self.meta_divergences[iteration]
        _, medoid_ix = self.calculate_medoids(divergences, k)
        medoid_policies = []
        for i, policy in enumerate(self.memory[iteration]):
            if i in medoid_ix:
                medoid_policies.append(policy)
        self.medoid_policies = medoid_policies
        # Save medoid policies for next round
        self.memory[iteration + 1] = medoid_policies
        return medoid_policies

    def calculate_medoids(self, distances, k=6):
        """
        Method to calculate medoids given distance matrix, taken from
        https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py
        """
        m = distances.shape[0]  # number of points

        # Pick k random medoids.
        curr_medoids = np.array([-1] * k)
        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.array([random.randint(0, m - 1)
                                     for _ in range(k)])
        # Doesn't matter what we initialize these to.
        old_medoids = np.array([-1] * k)
        new_medoids = np.array([-1] * k)

        # Until the medoids stop updating, do the following:
        while not ((old_medoids == curr_medoids).all()):
            # Assign each point to cluster with closest medoid.
            clusters = assign_points_to_clusters(curr_medoids, distances)

            # Update cluster medoids to be lowest cost point.
            for curr_medoid in curr_medoids:
                cluster = np.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(
                    cluster, distances)

            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]

        return clusters, curr_medoids

    def assign_points_to_clusters(self, medoids, distances):
        distances_to_medoids = distances[:, medoids]
        clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
        clusters[medoids] = medoids
        return clusters

    def compute_new_medoid(self, cluster, distances):
        mask = np.ones(distances.shape)
        mask[np.ix_(cluster, cluster)] = 0.
        cluster_distances = np.ma.masked_array(
            data=distances, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)

        # def update_nearest_neighbors(self, K, iteration):
        #     """Update adjacency list of K nearest neighbors, given prior policies"""
        #     divergences = self.meta_divergences[iteration]
