import numpy as np

class LogisticRegression():
    def __init__(self, n_steps=100, batch_size=2, is_intercept=True, regularization=None, learning_rate=0.001, lambda_=10):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.is_intercept = is_intercept
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.weights = None

    def _add_intercept(self, X):
        n_points = X.shape[0]
        return np.hstack((X, np.ones((n_points, 1))))


    def _create_batch(self, X, y):
        n_points = X.shape[0]
        list_of_index = np.arange(n_points)
        np.random.shuffle(list_of_index)

        for start_id in range(0, n_points, self.batch_size):
            end_id = min(start_id + self.batch_size, n_points)
            batch_idx = list_of_index[start_id:end_id]
            yield X[batch_idx], y[batch_idx]


    def sigmoid(self, array):
        return 1 / (1 + np.exp(-array))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)


        if self.is_intercept:
            X = self._add_intercept(X)

        self.weights = np.zeros((X.shape[1], 1))
        y = y.reshape(X.shape[0], -1)
        for i in range(self.n_steps):
            for x_data, y_data in self._create_batch(X, y):
                n_batch = x_data.shape[0]
                prediction = self.sigmoid(x_data @ self.weights)
                grad = (1/n_batch) * (x_data.T) @ (prediction - y_data)
                if self.regularization == "l1":
                    reg_term = self.lambda_ * np.sign(np.vstack(([[0]], self.weights[1:])))
                    grad += reg_term / n_batch
                elif self.regularization == "l2":
                    reg_term = self.lambda_ * np.vstack(([[0]], self.weights[1:]))
                    grad += reg_term / n_batch

                self.weights -= self.learning_rate * grad

        print(self.weights)

    def predict(self, X, thresh = 0.5):
        X = np.array(X)
        if self.is_intercept:
            X = X.reshape(-1, self.weights.shape[0] - 1)
            X = self._add_intercept(X)
        else:
            X = X.reshape(-1, self.weights.shape[0])

        pred = self.sigmoid(X @ self.weights)
        pred[pred > thresh] = 1
        pred[pred < thresh] = 0
        pred = pred.astype(np.int8) 
        return pred
    

if __name__ == "__main__":
    logistic_reg = LogisticRegression(regularization="l2")
    np.random.seed(42)
    X = np.random.randn(5,2)
    y = np.array([1,0,0,1,0])
    logistic_reg.fit(X,y)
    pred = logistic_reg.predict(np.random.randn(2))
    print(pred)


        
        
