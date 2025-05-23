
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left          # Левое поддерево (значения <= threshold)
        self.right = right       # Правое поддерево (значения > threshold)
        self.value = value

    def is_leaf(self):
        return self.value is not None
# если узел конечный, то метод будет возвращать значение в этом узле

class DecisionTree:
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

    def _most_common_class(self, y):
        counter = Counter(y)  # количество каждого элемента
        most_common_items = counter.most_common(1)  #список из одного самого частого элемента
        most_common_pair = most_common_items[0]  #первый элемент
        most_common_value = most_common_pair[0]  #само значение
        return most_common_value

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (self._stop_criteria(depth, n_samples, n_classes, np.inf)):
            leaf_value = self._most_common_class(y)
            self.n_leaves += 1
            return Node(value=leaf_value)


        best_feature, best_threshold, best_impurity = self._best_split(X, y)


        if best_feature is None or best_impurity < self.min_impurity_decrease:
            leaf_value = self._most_common_class(y)
            self.n_leaves += 1
            return Node(value=leaf_value)

        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold

        if (np.sum(left_idxs) < self.min_samples_leaf or
            np.sum(right_idxs) < self.min_samples_leaf):
            leaf_value = self._most_common_class(y)
            self.n_leaves += 1
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _stop_criteria(self, depth, n_samples, n_classes, impurity_decrease):
        # Критерии останова:
        # 1. Достигнута максимальная глубина
        # 2. Достигнуто максимальное количество листьев
        # 3. Все образцы одного класса
        # 4. Слишком мало образцов
        return (
            (self.max_depth is not None and depth >= self.max_depth) or
            (self.max_leaves is not None and self.n_leaves >= self.max_leaves) or
            (n_classes == 1) or
            (n_samples < 2 * self.min_samples_leaf)
        )

    def _gini(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        sum_of_squares = 0.0
        for count in class_counts:
            probability = count / total_samples
            sum_of_squares += probability ** 2
        gini = 1 - sum_of_squares
        return gini

    def _impurity_improvement(self, left_y, right_y, parent_impurity):
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        gini_left = self._gini(left_y)
        gini_right = self._gini(right_y)


        weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return parent_impurity - weighted_gini
    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_impurity = -np.inf
        parent_impurity = self._gini(y)

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
