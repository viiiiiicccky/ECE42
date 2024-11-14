import numpy as np
import random
import collections
from numpy.random.mtrand import f
import util

# YOU ARE NOT ALLOWED TO USE sklearn or Pytorch in this assignment


class Optimizer:

    def __init__(
        self, name, lr=0.001, gama=0.9, beta_m=0.9, beta_v=0.999, epsilon=1e-8
    ):
        # self.lr will be set as the learning rate that we use upon creating the object, i.e., lr
        # e.g., by creating an object with Optimizer("sgd", lr=0.0001), the self.lr will be set as 0.0001
        self.lr = lr

        # Based on the name used for creating an Optimizer object,
        # we set the self.optimize to be the desiarable method.
        if name == "sgd":
            self.optimize = self.sgd
        elif name == "heavyball_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.heavyball_momentum
        elif name == "nestrov_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.nestrov_momentum
        elif name == "adam":
            # setting beta_m, beta_v, and epsilon
            # (read the handout to see what these parametrs are)
            self.beta_m = beta_m
            self.beta_v = beta_v
            self.epsilon = epsilon

            # setting the initial first momentum of the gradient
            # (read the handout for more info)
            self.v = 0

            # setting the initial second momentum of the gradient
            # (read the handout for more info)
            self.m = 0

            # initializing the iteration number
            self.t = 1

            self.optimize = self.adam

    def sgd(self, gradient):
        # update rule: w_t+1 = w_t - lr * gradient
        return -self.lr * gradient

    def heavyball_momentum(self, gradient):
        # adds heavyball momentum to sgd
        update = -self.lr * gradient + self.gama * self.v
        self.v = update # update the momentum
        return update

    def nestrov_momentum(self, gradient):
        return self.heavyball_momentum(gradient)

    def adam(self, gradient):
        # following the handout fomulas
        self.m = (1 - self.beta_m) * gradient + self.beta_m * self.m
        self.v = (1 - self.beta_v) * np.square(gradient) + self.beta_v * self.v
        m_hat_t_1 = self.m / (1 - pow(self.beta_m, self.t))
        v_hat_t_1 = self.v / (1 - pow(self.beta_v, self.t))
        self.t = self.t + 1
        update = -self.lr * m_hat_t_1 / (np.sqrt(v_hat_t_1) + self.epsilon)
        return update
        


class MultiClassLogisticRegression:
    def __init__(self, n_iter=10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres

    def fit(
        self,
        X,
        y,
        batch_size=64,
        lr=0.001,
        gama=0.9,
        beta_m=0.9,
        beta_v=0.999,
        epsilon=1e-8,
        rand_seed=4,
        verbose=False,
        optimizer="sgd",
    ):
        # setting the random state for consistency.
        np.random.seed(rand_seed)

        # find all classes in the train dataset.
        self.classes = self.unique_classes_(y)

        # assigning an integer value to each class, from 0 to (len(self.classes)-1)
        self.class_labels = self.class_labels_(self.classes)

        # one-hot-encode the labels.
        self.y_one_hot_encoded = self.one_hot(y)

        # add a column of 1 to the leftmost column.
        X = self.add_bias(X)

        # initialize the E_in list to keep track of E_in after each iteration.
        self.loss = []

        # initialize the weight parameters with a matrix of all zeros.
        # each row of self.weights contains the weights for one of the classes.
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))

        # create an instance of optimizer
        opt = Optimizer(
            optimizer, lr=lr, gama=gama, beta_m=beta_m, beta_v=beta_v, epsilon=epsilon
        )

        i, update = 0, 0
        while i < self.n_iter:
            self.loss.append(
                self.cross_entropy(self.y_one_hot_encoded, self.predict_with_X_aug_(X))
            )

            # sample a batch of data, X_batch and y_batch, with batch_size number of datapoint uniformly at random
            random_index = np.random.choice(X.shape[0], batch_size)
            X_batch = X[random_index]
            y_batch = self.y_one_hot_encoded[random_index]

            # find the gradient that should be inputed the optimization function.
            gradient = self.compute_grad(X_batch, y_batch, self.weights)

            update = opt.optimize(gradient) # update vector by using the optimizatio method
            self.weights += update # update self.weights

            # stopping criterion. check if norm infinity of the update vector is smaller than self.thres.
            # if so, break the while loop.
            if np.linalg.norm(update, np.inf) < self.thres:
              break

            if i % 1000 == 0 and verbose:
                print(
                    " Training Accuray at {} iterations is {}".format(
                        i, self.evaluate_(X, self.y_one_hot_encoded)
                    )
                )
            i += 1
        return self

    def add_bias(self, X):
        # inserts a column of 1's to the leftmost column of X
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def unique_classes_(self, y):
        # return a list of unique elements in y
        return np.unique(y)

    def class_labels_(self, classes):
        # create a dictionary where each class is mapped to a unique integer label
        class_labels_dict = {} # initialize an empty dictionary
        num = 0 
        for each_class in classes:
          class_labels_dict[each_class] = num # set the value of this class element to a unique number
          num += 1 # increment the number by 1 for the next class
        return class_labels_dict

    def one_hot(self, y):
        y_one_hot = np.zeros((len(y), len(self.class_labels))) # initialize matrix to 0
        for i, label in enumerate(y):
          class_index = self.class_labels[label] # get class index for the label
          y_one_hot[i][class_index] = 1
        return y_one_hot

    def softmax(self, z):
        # ref: https://en.wikipedia.org/wiki/Softmax_function
        z_exp = np.exp(z)
        sum_z_exp = np.sum(z_exp, axis=1) # compute the sum of exp along each row
        return z_exp / sum_z_exp.reshape(-1, 1)

    def predict_with_X_aug_(self, X_aug):
        return self.softmax(np.dot(X_aug, self.weights.T))

    def predict(self, X):
        # using the implemented methods add_bias() and predict_with_X_aug()
        X_aug = self.add_bias(X)
        return self.predict_with_X_aug_(X_aug)

    def predict_classes(self, X):
        # ref: https://numpy.org/doc/2.0/reference/generated/numpy.argmax.html
        # argmax returns the indices of the maximum values along an axis.
        predicted_prob = self.predict(X) # get predicted probabilities for each class
        predicted_classes_indices = np.argmax(predicted_prob, axis=1)
        return self.classes[predicted_classes_indices]

    def score(self, X, y):
        correct = 0 # track the number of correctly classified classes
        predicted_classes = self.predict_classes(X)
        for i in range(len(y)): # loop through predictions and true labels
          if predicted_classes[i] == y[i]:
            correct += 1
        return correct / len(y) # ratio of correct predictions

    def evaluate_(self, X_aug, y_one_hot_encoded):
        predicted_prob = self.predict_with_X_aug_(X_aug)
        predicted_classes = np.argmax(predicted_prob, axis=1)
        true_classes = np.argmax(y_one_hot_encoded, axis=1)

        correct = 0 # track the number of correct classifications
        for i in range(len(true_classes)):
          if predicted_classes[i] == true_classes[i]:
            correct += 1
        return correct / len(true_classes) # return the accuracy
          
    def cross_entropy(self, y_one_hot_encoded, probs):
        # CE(P,Q) = - sum(P(k) * log(Q(k))), where k is over all classes
        # compute the cross entropy error between one-hot encoded true labels and the predicted probabilities
        return -np.sum(y_one_hot_encoded * np.log(probs)) / y_one_hot_encoded.shape[0]

    def compute_grad(self, X_aug, y_one_hot_encoded, w):
        prob = self.predict_with_X_aug_(X_aug) # get the predicted probabilities
        err = prob - y_one_hot_encoded # get the difference between prediction and true labels
        gradient = np.dot(X_aug.T, err) / X_aug.shape[0] # compute the gradient of the loss function
        return gradient.T


def kmeans(examples, K, maxIters):
    """
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """

    centers = random.sample(examples, K)
    # centers = [c.copy() for c in centers]
    assignments = [-1] * len(examples)
    totalCost = 0

    for _ in range(maxIters):
        new_assignments = [-1] * len(examples)
        new_centers = [collections.Counter() for _ in range(K)]
        totalCost = 0  # reset totalCost for each iteration

        # precompute norms for centers to avoid redundant calculations
        center_norms = [np.linalg.norm(list(center.values())) for center in centers]

        for i, example in enumerate(examples):
            example_norm = np.linalg.norm(list(example.values()))
            best_center = 0
            best_similarity = -1

            for j, center in enumerate(centers):
                similarity = util.dotProduct(center, example) / (center_norms[j] * example_norm)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_center = j

            new_assignments[i] = best_center
            totalCost += 1 - best_similarity
            util.increment(new_centers[best_center], 1, example)

        if new_assignments == assignments:
            break

        assignments = new_assignments
        centers = new_centers

    return centers, assignments, totalCost


    
