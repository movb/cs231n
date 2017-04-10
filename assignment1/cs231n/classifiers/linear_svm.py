import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  h = 0.0001
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
    #print(W.shape)
    #print(W.T.dot(X[i]).shape)
    #print(W[:,y[i]].T.dot(X[i]).shape)
    #print(X[i].T.shape)
    grad = ((W.T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1.0) > 0) * X[i].reshape(X[i].shape[0],1)
    grad[:,y[i]] = -(np.sum((W.T.dot(X[i]) - W[:,y[i]].T.dot(X[i]) + 1.0) > 0) - 1.0)*X[i]
    #grad = ((X[i].dot(W) - X[i].dot(W[:,y[i]]) + 1) > 0)*X[i].reshape(X[i].shape[0],1)
    #grad[:,y[i]] = -(np.sum((X[i].dot(W) - X[i].dot(W[:,y[i]]) + 1) > 0) - 1)*X[i]
    dW += grad 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = scores[y]
  loss = (np.sum((scores - correct_class_score + 1) > 0) - num_train)/float(num_train)
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  print("W.shape = ", W.shape)
  print("W[:, y].shape = ", W[:, y].shape)
  print(((X.dot(W) - np.sum(X*W[:, y].T,axis=1).reshape(y.shape[0],1) + 1) > 0).shape)
  #print( ( (np.sum((X.dot(W) - np.sum(X*W[:, y].T,axis=1).reshape(y.shape[0],1) + 1) > 0, axis=1) - 1).T.dot(X) ).shape)
  #print( dW[np.arange(dW.shape[0]),y].shape )
  
  temp = ((X.dot(W) - np.sum(X*W[:, y].T,axis=1).reshape(y.shape[0],1) + 1) > 0)
  temp[np.arange(temp.shape[0]),y] = -(np.sum((X.dot(W) - np.sum(X*W[:, y].T,axis=1).reshape(y.shape[0],1) + 1) > 0, axis=1) - 1)
  dW = X.T.dot(temp)
  dW += reg*W
  #dW[np.arange(dW.shape[0]),y] = -np.sum((X.dot(W) - np.sum(X*W[:, y].T,axis=1).reshape(y.shape[0],1) + 1) > 0, axis=1).T.dot(X)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
