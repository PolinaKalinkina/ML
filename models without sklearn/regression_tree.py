
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, max_leaves=None, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.n_leaves = 0

    def fit(self, X, y):
        self.n_leaves = 0
        self.root = self._grow_tree(X, y, depth=0)

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape


        if self._stop_criteria(depth, n_samples, np.inf):
            leaf_value = self._calculate_leaf_value(y)
            self.n_leaves += 1
            return Node(value=leaf_value)

        best_feature, best_threshold, best_impurity = self._best_split(X, y)

        if best_feature is None or best_impurity < self.min_impurity_decrease:
            leaf_value = self._calculate_leaf_value(y)
            self.n_leaves += 1
            return Node(value=leaf_value)

        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold

        if (np.sum(left_idxs) < self.min_samples_leaf or
            np.sum(right_idxs) < self.min_samples_leaf):
            leaf_value = self._calculate_leaf_value(y)
            self.n_leaves += 1
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _stop_criteria(self, depth, n_samples, impurity_decrease):

        return (
            (self.max_depth is not None and depth >= self.max_depth) or
            (self.max_leaves is not None and self.n_leaves >= self.max_leaves) or
            (n_samples < 2 * self.min_samples_leaf)
        )

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _impurity_improvement(self, left_y, right_y, parent_impurity):
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        mse_left = self._mse(left_y)
        mse_right = self._mse(right_y)

        weighted_mse = (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
        return parent_impurity - weighted_mse

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_impurity = -np.inf
        parent_impurity = self._mse(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs = X[:, feature] <= threshold
                right_idxs = X[:, feature] > threshold

                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue

                impurity = self._impurity_improvement(
                    y[left_idxs], y[right_idxs], parent_impurity
                )

                if impurity > best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_impurity

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
