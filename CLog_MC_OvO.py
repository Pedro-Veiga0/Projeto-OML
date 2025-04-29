from CLogDKPd_MGmB import CLogDKPd_MGmB
from CLog_MGmB import CLog_MGmB
from utils import plot_decision_boundary
import numpy as np

class CLog_OVO():
    def __init__(self, n_iter=1000, batch_size=None, kernel=1, step=0.01, mode='primal'):
        """
        Initialize the CLog_OVO classifier.
        Parameters:
            n_iter : int, optional (default=1000)
                Number of iterations for the optimization algorithm.
            batch_size : int, optional (default=None)
                Size of the mini-batch for stochastic gradient descent. if None, uses the entire dataset for each iteration.
            kernel : int, optional (default=1)
                Degree of the polynomial kernel.
            step : float, optional (default=0.01)
                Learning rate for the optimization algorithm.
            mode : str, optional (default='primal')
                Mode of the classifier. 'primal' for primal form, 'dual' for dual form.
        """
        self.n_iter = n_iter
        self.models = []
        self.classes_ = None
        self.batch_size = batch_size
        self.kernel = kernel if mode == 'dual' else 1
        self.step = step
        self.mode = mode
    
    def fit(self, X, y):
        """
        One-vs-One (OvO) classification using CLogDKPd_MGmB.
        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training data.
            y : array-like, shape (n_samples,)
                Target labels.
        """
        # get the number of classes
        self.classes_ = set(y)
        # create a list of models for each pair of classes
        self.models = []
        for i, class_i in enumerate(self.classes_):
            for j, class_j in enumerate(self.classes_):
                if i < j:
                    # create a binary classifier for the pair of classes
                    mask = (y == class_i) | (y == class_j)

                    if self.mode == 'primal':
                        model = CLog_MGmB(n_iter=self.n_iter, batch_size=self.batch_size, learning_rate=self.step)
                    else:
                        model = CLogDKPd_MGmB(n_iter=self.n_iter, batch_size=self.batch_size, kernel=self.kernel, learning_rate=self.step)

                    # fit the model on the binary data
                    X_bin = X[mask]
                    y_binary = y[mask]
                    # convert the labels to 0 and 1 - class_i = 1, class_j = 0
                    y_binary = (y_binary == class_i).astype(int)
                    # fit the model
                    model.fit(X_bin, y_binary)
                    self.models.append((model, class_i, class_j))
        return self
    
    def predict(self, X):
        """
        Predict the class labels for the input data.
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns:
        y_pred : array, shape (n_samples,)
            Predicted class labels.
        """
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes_)))  # (n_samples, n_classes)

        # Map class labels to indices
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}

        # For each model, get the predicted probabilities and update the votes
        for model, class_i, class_j in self.models:
            pred = model.predict_proba(X)  # (n_samples,)
            for i in range(n_samples):
                votes[i, class_to_index[class_i]] += pred[i]
                votes[i, class_to_index[class_j]] += 1 - pred[i]

        # Return the class with the most votes
        y_pred = votes.argmax(axis=1)
        return np.array(y_pred)
    
    def get_errors(self):
        """
        Get the errors of each model.
        Returns:
        errors : list
            List of errors for each model.
        """
        errors = []
        for model, class_i, class_j in self.models:
            errors.append((model.get_errors(), class_i, class_j))
        return errors
    

if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification, make_blobs, make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate a synthetic dataset
    # X, y = make_classification(n_samples=1000, n_features=784, n_classes=3, n_informative=500, random_state=42)
    # X, y = make_classification(n_samples=1000,
    #                         n_features=2,
    #                         n_informative=2,
    #                         n_redundant=0,
    #                         n_clusters_per_class=1,
    #                         class_sep=0.95,        # separação pequena → mais difícil
    #                         n_classes=4,
    #                         random_state=12)
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    train_std = np.std(X_train, axis=0)
    train_mean = np.mean(X_train, axis=0)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Create an instance of the CLog_OVO classifier
    clf = CLog_OVO(n_iter=5000, batch_size=128, kernel=3, step=0.001, mode='dual')
    
    # Fit the model on the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    #pribt probs
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")

    # Plot the decision boundary
    plot_decision_boundary(clf, X_test, y_test,500)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # # teste with sklearn classifier
    # from sklearn.linear_model import LogisticRegression
    # clf_sklearn = LogisticRegression(max_iter=5000)
    # clf_sklearn.fit(X_train, y_train)
    # y_pred_sklearn = clf_sklearn.predict(X_test)
    # accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    # print(f"Accuracy sklearn: {accuracy_sklearn:.2f}")