
class GridSearchCV:
    def __init__(self, estimator_class, param_grid, cv=5):
        self.estimator_class = estimator_class
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -1
        self.best_estimator_ = None

    def fit(self, X, y):
        param_combinations = self._generate_param_combinations()

        for params in param_combinations:
            scores = []
            fold_size = len(X) // self.cv

            for i in range(self.cv):

                val_start, val_end = i * fold_size, (i + 1) * fold_size
                X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                X_train = np.concatenate([X[:val_start], X[val_end:]])
                y_train = np.concatenate([y[:val_start], y[val_end:]])


                model = self.estimator_class(**params)
                model.fit(X_train, y_train)


                y_proba = model.predict_proba(X_val)[:, 1]
                score = self._roc_auc_score(y_val, y_proba)
                scores.append(score)

            mean_score = np.mean(scores)

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = model

    def _generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def _roc_auc_score(self, y_true, y_proba):
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]


        n_pos = np.sum(y_true == 1)
        n_neg = len(y_true) - n_pos


        tpr = []
        fpr = []
        current_tp = 0
        current_fp = 0

        for label in y_true_sorted:
            if label == 1:
                current_tp += 1
            else:
                current_fp += 1
            tpr.append(current_tp / n_pos)
            fpr.append(current_fp / n_neg)


        auc = 0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

        return auc
