import numpy as np

class LinearRegression():
    def __init__(self, is_intercept=True):
        self.is_intercept = is_intercept
        self.weights = None

    def _add_intercept(self, X):
        n_points, n_features = X.shape
        intercept = np.ones((n_points, 1))
        return np.hstack((intercept, X))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.is_intercept:
            X = self._add_intercept(X)

        XTX = X.T @ X
        XTy = X.T @ y

        self.weights = np.linalg.inv(XTX) @ XTy
        print(self.weights)

    def predict(self, X):
        X = np.array(X)
        if self.is_intercept:
            X = X.reshape(-1, self.weights.shape[0]-1)
            X = self._add_intercept(X)
        else:
            X = X.reshape(-1, self.weights.shape[0])
            X = self._add_intercept(X)
        y_pred = X @ self.weights
        return y_pred
    

class LinearRegressionSGD():
    def __init__(self, learning_rate=0.001, num_inter=1000, is_intercept=True, batch_size=2):
        self.learning_rate = learning_rate
        self.num_inter = num_inter
        self.is_intercept = is_intercept
        self.batch_size = batch_size
        self.weights = None

    def _add_intercept(self, X):
        n_points = X.shape[0]
        intercept = np.ones((n_points, 1))
        return np.hstack((X, intercept))

    def _create_batch(self, X, y):
        n_samples = X.shape[0]
        list_of_index = np.arange(n_samples)
        np.random.shuffle(list_of_index)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx+self.batch_size, n_samples)
            batch_idx =  list_of_index[start_idx:end_idx]
            yield X[batch_idx], y[batch_idx]


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.is_intercept:
            X = self._add_intercept(X)
        
        self.weights = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)

        for i in range(self.num_inter):
            for x_batch, y_batch in self._create_batch(X, y):
                n_batch = x_batch.shape[0]
                predictions = x_batch @ self.weights
                grad = (1 /  n_batch) * (x_batch.T @ (predictions - y_batch))
                self.weights -= (self.learning_rate  * grad)

    def predict(self, X):
        if self.is_intercept:
            X = X.reshape(-1, self.weights.shape[0]-1)
            X = self._add_intercept(X)
        else:
            X = X.reshape(-1, self.weights.shape[0])
        return X @ self.weights


if __name__ == "__main__":
    linear_regression = LinearRegressionSGD()
    np.random.seed(42)
    X = np.random.randn(5,2)
    y = np.random.rand(5)
    linear_regression.fit(X, y)
    print(linear_regression.weights)    
    new_X = np.random.randn(2)
    y_pred = linear_regression.predict(new_X)
    print(y_pred)