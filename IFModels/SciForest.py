import numpy as np
import random

EPSILON = 1e-10  # division by zero önlemi

def c(n):
    if n > 2:
        return 2 * (np.log(n - 1) + np.euler_gamma) - (2 * (n - 1) / n)
    elif n == 2:
        return 1
    else:
        return 0

class Node:
    def __init__(self, left, right, normal_vector, bias, means, stds, size=None):
        self.left = left
        self.right = right
        self.normal_vector = normal_vector
        self.bias = bias
        self.size = size
        self.means = means
        self.stds = stds

    def is_external(self):
        return self.left is None and self.right is None

class ITree:
    def __init__(self, X, current_height, max_height, k_planes, extension_level):
        self.tree = self.fit(X, current_height, max_height, k_planes, extension_level)

    def fit(self, X, current_height, max_height, k_planes, extension_level):
        if current_height > max_height or len(X) < 2:
            return Node(None, None, None, None, None, None, len(X))

        best_sep = np.finfo(np.float64).min
        best_normal_vector = None
        best_bias = None
        best_means = None
        best_stds = None
        best_z = None

        for _ in range(k_planes):
            sep, bias, normal_vector, means, stds, z = self.find_split(X, extension_level)
            if sep > best_sep:
                best_normal_vector = normal_vector
                best_bias = bias
                best_means = means
                best_stds = stds
                best_z = z

        if best_z is None or best_bias is None:
            return Node(None, None, None, None, None, None, len(X))
        else:
            X_l = X[np.where(best_z < best_bias)]
            X_r = X[np.where(best_z >= best_bias)]

        return Node(
            ITree(X_l, current_height + 1, max_height, k_planes, extension_level).tree,
            ITree(X_r, current_height + 1, max_height, k_planes, extension_level).tree,
            best_normal_vector,
            best_bias,
            best_means,
            best_stds
        )

    def find_split(self, X, extension_level):
        z = np.zeros(X.shape[0])
        nv = np.zeros(X.shape[1])
        means = np.zeros(X.shape[1])
        stds = np.zeros(X.shape[1])
        i = 0

        features_to_try = list(range(X.shape[1]))
        np.random.shuffle(features_to_try)

        at_least_one = False

        while i < extension_level + 1:
            if not features_to_try:
                if at_least_one:
                    break
                else:
                    return np.finfo(np.float64).min, None, None, None, None, None

            feature = features_to_try.pop()
            y = X[:, feature]
            c_val = np.random.uniform(-1, 1)
            y_std = y.std()

            if y_std == 0:
                continue
            at_least_one = True

            y_mean = y.mean()
            nv[feature] = c_val
            means[feature] = y_mean
            stds[feature] = y_std
            z += c_val * ((y - y_mean) / (y_std + EPSILON))
            i += 1

        best_sep = np.finfo(np.float64).min
        best_s = z[0]
        std = np.std(z)

        for s in z:
            z_l = z[z < s]
            z_r = z[z >= s]

            if len(z_l) == 0 or len(z_r) == 0:
                continue

            sep = (std - ((np.std(z_l) + np.std(z_r)) / 2)) / std
            if sep > best_sep:
                best_sep = sep
                best_s = s

        return best_sep, best_s, nv, means, stds, z

    def path_length(self, x, node=None, current_length=0):
        if node is None:
            node = self.tree

        if node.is_external():
            return current_length + c(node.size)

        z = 0
        for feature, normal_vector in enumerate(node.normal_vector):
            if normal_vector != 0:
                z += normal_vector * ((x[feature] - node.means[feature]) / (node.stds[feature] + EPSILON))

        if z < node.bias:
            return self.path_length(x, node.left, current_length + 1)
        else:
            return self.path_length(x, node.right, current_length + 1)

class SCiForest:
    def __init__(self, n_trees, sample_size, k_planes, extension_level):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.k_planes = k_planes
        self.max_height = np.ceil(np.log2(sample_size))
        self.forest = []
        self.n = None
        self.extension_level = extension_level

    def fit(self, X):
        self.n = X.shape[0]

        if self.extension_level == 'full':
            self.extension_level = min(X.shape[1] - 1, 3)  # güvenli test

        if self.extension_level > X.shape[1] - 1:
            raise Exception('Too high extension level for the dataset dimension.')

        # Tek işlemli fit
        for i in range(self.n_trees):
            sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
            self.forest.append(self._fit_tree(sample, i))

    def _fit_tree(self, sample, i):
        return ITree(sample, 0, self.max_height, self.k_planes, self.extension_level)

    def anomaly_score(self, x):
        expected_path_length = np.mean([tree.path_length(x) for tree in self.forest])
        return 2 ** (-expected_path_length / c(self.sample_size))

    def predict(self, X):
        return np.array([self.anomaly_score(x) for x in X])

    def compute_paths_all_tree(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        paths = np.zeros((n_samples, self.n_trees))

        for i, x in enumerate(X):
            for j, tree in enumerate(self.forest):
                paths[i, j] = int(tree.path_length(x))  # int değer
        return paths


