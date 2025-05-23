

class Regularization:
    def __init__(self, alpha=1.0):
        self.alpha = alpha



class LassoRegularization(Regularization):
    def penalty(self, w):
        return self.alpha * np.sum(np.abs(w))

class RidgeRegularization(Regularization):
    def penalty(self, w):
        return self.alpha * np.sum(w**2)


class ElasticNetRegularization(Regularization):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super().__init__(alpha)
        self.l1_ratio = l1_ratio

    def penalty(self, w):
        l1_penalty = self.l1_ratio * np.sum(np.abs(w))
        l2_penalty = (1 - self.l1_ratio) * np.sum(w**2)
        return self.alpha * (l1_penalty + l2_penalty)

class LogisticRegression:
    def __init__(self, regularization=None, learning_rate=0.01, n_iterations=1000, class_weight=None):

        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.class_weight = class_weight
        self.weights = None
        self.bias = None



    def _sigmoid(self, z):
        # z: (w * X + b)
        return 1 / (1 + np.exp(-z))


        # X: матрица признаков (n_samples, n_features)   y: вектор целевых значений (n_samples,)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Рассчитываем веса классов
        if self.class_weight is not None:
            class_weights = np.array([self.class_weight[cls] for cls in y])
        else:
            class_weights = np.ones(n_samples)

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)


            #dw: градиент по весам (производная функции потерь по весам)
            #db: градиент по смещению (производная функции потерь по смещению)
            dw = (1 / n_samples) * np.dot(X.T, class_weights * (y_predicted - y))
            db = (1 / n_samples) * np.sum(class_weights * (y_predicted - y))


            #если reg не None, добавляем штраф к градиенту
            if self.regularization:
                dw += (self.regularization.penalty(self.weights) / n_samples)

            #обновляем веса и смещение с учетом градиентов
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        #порог 0.5
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

class_weights = {0: 1, 1: 10}
