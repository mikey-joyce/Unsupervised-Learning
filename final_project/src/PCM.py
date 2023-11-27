import numpy as np
from FCM import FCM


class PCM(FCM): # A possibilistic approach to clustering; Possibilistic C-Means (PCM)
    def __init__(self, data, num_clusters, m=2, max_epochs=1000, tol=1e-2):
        FCM.__init__(self, data, num_clusters, m, max_epochs, tol)

    def update_membership(self):
        membership = np.zeros((self.data.shape[0], self.num_clusters))

        for i in range(self.data.shape[0]):
            for j in range(self.num_clusters):
                numerator = np.linalg.norm(self.data[i] - self.centers[j])
                denominator = self.tol

                for k in range(self.num_clusters):
                    dist = np.linalg.norm(self.data[i] - self.centers[k])
                    dist = np.maximum(dist, 1e-8)  # Replace zeros with 1e-8
                    denominator += (numerator / dist) ** (1 / (self.m - 1))

                membership[i][j] = 1 / denominator

        return membership

    def update_center(self):
        centers = np.zeros((self.num_clusters, self.data.shape[1]))

        for j in range(self.num_clusters):
            denominator = np.sum(self.membership[:, j] ** self.m)

            for k in range(self.data.shape[0]):
                centers[j] += (self.membership[k, j] ** self.m) * self.data[k]

            centers[j] /= denominator

        return centers
