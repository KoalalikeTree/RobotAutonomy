import numpy as np
import matplotlib.pyplot as plt


def train_test_split(X, y, test_ratio):
    n = len(X)
    idxs = np.arange(n)
    np.random.shuffle(idxs)

    n_test = int(test_ratio * n)
    idxs_test = idxs[:n_test]
    idxs_train = idxs[n_test:]

    X_train = X[idxs_train]  
    X_test = X[idxs_test]
    y_train = y[idxs_train]
    y_test = y[idxs_test]

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


class TwoClassLogisticRegressionClassifier:

    def __init__(self, n_features):
        self._W = np.zeros(n_features + 1)

    def _sigmoid(self, arr):
        ''' TODO(Q2.1): Implement the sigmoid function over an array of numbers
        '''
        return np.divide(1, 1+np.exp(-arr))

    def predict_prob(self, X):
        ''' TODO(Q2.2): Implement the predict_prob function which returns the predicted probabilities of a positive class.

        Note: You need to augment all features with an additional bias feature of value 1
        '''
        _X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        out = self._sigmoid(np.dot(self._W.T, _X.T))
        return out

    def predict(self, X):
        pred_prob = self.predict_prob(X)
        return 1 * (pred_prob > 0.5)

    def fit(self, X, y, learning_rate, n_iters):
        accs = []
        acc = accuracy(y, self.predict(X))
        accs.append(acc)
        num_samples = X.shape[0]
        num_features = X.shape[1]

        X_aug = np.c_[X, np.ones(len(X))]
        for _ in range(n_iters):
            ''' TODO(Q2.3): Implement the gradient ascent function for logistic regression
            '''

            out = y - self._sigmoid(np.dot(self._W.T, X_aug.T))
            for j in range(num_features+1):
                self._W[j] += learning_rate * (np.sum(out * X_aug[:, j]))

            acc = accuracy(y, self.predict(X))
            accs.append(acc)

        return accs


class MultiClassLogisticRegressionClassifier:

    def __init__(self, n_features, n_classes):
        self._n_classes = n_classes
        self._clfs = [TwoClassLogisticRegressionClassifier(n_features) for _ in range(n_classes)]

    def predict_prob(self, X):
        return np.array([clf.predict_prob(X) for clf in self._clfs])

    def predict(self, X):
        pred_prob = self.predict_prob(X)
        return np.argmax(pred_prob, axis=0)

    def fit(self, X, y, learning_rate, n_iters):
        accs = []
        for t in range(n_iters):
            acc = accuracy(y, self.predict(X))
            print('Iter {} | Train acc: {:.3f}'.format(t, acc))
            accs.append(acc)

            for c, clf in enumerate(self._clfs):
                ''' TODO(Q3.1): Implement multi-class classification by:

                - Creating a new set of labels where labels of the current class (c) has 1, and the rest 0
                - Fitting the current classifier (clf) on this new set of 0-1 labels

                Note: you should only call clf.fit with n_iters=1! 
                This is because we are iterating the outer loop with the actual n_iters
                '''

                binary_label = ( y == c )
                acc_c = clf.fit(X, binary_label, learning_rate, 1)
                print(c, ' class = ', acc_c)

        return accs


if __name__ == "__main__":
    data = np.load('data.npz')
    features, labels, names = data['features'], data['labels'], data['names']

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, 0.3)

    # Make model
    n_features = X_test.shape[1]
    n_classes = len(np.unique(y_test))
    multi_clf = MultiClassLogisticRegressionClassifier(n_features, n_classes)

    # Fit model
    learning_rate = 1e-1
    n_iters = 100
    accs = multi_clf.fit(X_train, y_train, learning_rate, n_iters)
    print('Achieved final training accuracy: {:.3f}'.format(accs[-1]))

    # Plot Training accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(accs)
    plt.xlim([0, n_iters - 1])
    plt.ylim([0, 1])
    plt.xlabel('Training iteration')
    plt.ylabel('Training Accuracy')
    plt.title('Accuracy on Training Set during Training')
    plt.savefig('p2_train.png')
    plt.show()

    # Test set accuracy
    y_test_preds = multi_clf.predict(X_test)
    test_acc = accuracy(y_test, y_test_preds)
    print('Achieved test accuracy: {:.3f}'.format(test_acc))
