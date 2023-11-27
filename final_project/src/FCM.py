import numpy as np


class FCM:  # Fuzzy C-Means Clustering
    def __init__(self, data, num_clusters, m=2, max_epochs=1000, tol=1e-2):
        self.data = data
        self.num_clusters = num_clusters
        self.m = m
        self.max_epochs = max_epochs
        self.tol = tol
        self.centers = data[np.random.choice(data.shape[0], num_clusters, replace=False)]
        self.membership = None

    def get_distance(self, point):
        return np.linalg.norm(point - self.centers, axis=1)

    def update_membership(self, data):
        distances = np.array([self.get_distance(d) for d in data])
        distances = np.maximum(distances, 1e-8)  # Replace zeros with 1e-8
        return 1 / np.sum((distances[:, np.newaxis] / distances[:, :, np.newaxis]) ** (2 / (self.m - 1)), axis=1)

    def update_center(self):
        return np.sum((self.membership[:, :, np.newaxis] ** self.m) * self.data[:, np.newaxis, :], axis=0) / np.sum(self.membership[:, :, np.newaxis] ** self.m, axis=0)

    def __harden_membership(self, membership):
        hardened = []

        for x in range(membership.shape[0]):
            hardened.append(membership[x].argmax())

        return np.array(hardened)

    def fit(self):  # FCM main algorithm
        num_nodes, _ = self.data.shape

        for _ in range(self.max_epochs):
            old_centers = np.copy(self.centers)

            # update membership degrees of the data points
            self.membership = self.update_membership(self.data)

            # update the centers of the clusters
            self.centers = self.update_center()

            # convergence ?
            if np.linalg.norm(self.centers - old_centers) < self.tol:
                break

        # fuzzy partition coefficient
        fpc = np.sum(self.membership**self.m) / num_nodes

        return self.centers, self.__harden_membership(self.membership), fpc

    def predict(self, data):
        preds = self.update_membership(data)
        return self.__harden_membership(preds)

