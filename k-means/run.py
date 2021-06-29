# -*- coding: utf-8 -*-
import os, pickle
import numpy as np
from PIL import Image
from IPython.display import display


def load_mnist():
  if not os.path.exists('mnist.pkl'):
    url = 'https://lab.djosix.com/mnist.pkl.gz'
    assert os.system(f'wget -O mnist.pkl.gz {url}') == 0
    assert os.system('gunzip mnist.pkl.gz') == 0

  with open('mnist.pkl', 'rb') as f:
    return pickle.load(f)


def sample_from_clusters(X, cluster_indices, num_clusters, k=32):
  """
  Sample k MNIST images from each cluster, one row per cluster, and form an image.
  """

  rows = []

  for i in range(num_clusters):
    row = X[np.where(cluster_indices == i)]
    row = row[np.random.choice(row.shape[0], size=k, replace=True)]
    row = row.reshape(-1, 28, 28) # each datapoint is an 28x28 image
    row = np.concatenate(row, 1)

    rows.append(row) # sampled images
    rows.append(np.full([1, row.shape[1]], 255, dtype=np.uint8)) # white line

  rows = np.concatenate(rows, 0)

  return Image.fromarray(rows)


def compute_mean_distance(X, cluster_indices, num_clusters):
  """
  Compute mean L2 distance from data points to their cluster centroids.
  """
  assert X.shape[0] == cluster_indices.shape[0], 'size not matched'

  total = np.zeros([num_clusters, X.shape[1]])
  count = np.zeros([num_clusters])

  for x, c in zip(X, cluster_indices):
    total[c] += x
    count[c] += 1

  count[count == 0] = 1 # avoid zero division when there is nothing in a cluster
  means = total / count[:, np.newaxis]

  distances = np.sqrt(np.power(X - means[cluster_indices], 2).sum(1))
  return distances.mean()


def your_score(mean_distance):
  r = (mean_distance - 1575) / (1700 - 1575)
  return round(min(max(100 - 50 * r, 0), 100))


"""## Implement your k-means"""
def sample(X,k):
    rand_arr = np.arange(X.shape[0])
    np.random.shuffle(rand_arr)
    print(X)
    return (X[rand_arr[0:k]])

def compute_new_centroid(X,guess,centroid,num_clusters):
  y=guess.reshape(guess.shape[0],1)
  Xy=np.append(X,y, axis=1)
  for i in range(num_clusters):
      cluster = (Xy[:,len(Xy[0])-1]==i)
      cluster_set = X[cluster]
      if len(cluster_set)==0:
        centroid=(np.max(X)-0) * np.random.random_sample(size=(k,X.shape[1])) + 0
      else: 
        centroid[i]=sum(cluster_set[:,])/len(cluster_set)
  print(centroid)
  return centroid
  
  
def assign_point(X,guess,centroid):
  guess=[nearst_centroid(_,centroid) for _ in X]
  return guess

def nearst_centroid(x,centroid):
  distance=[sum((x-c)*(x-c)) for c in centroid]
  nearst=distance.index(min(distance))
  return nearst

def kmeans(X, num_clusters):
  """
  Run K-means algorithm on X and yield assigned cluster indices at each steps.

  Args:
    X (np.ndarray):
      An array of size N*D, where N is the dataset size and D is the number of
      features. For MNIST, N is 70000 and D is 784 (28x28).
    num_clusters (int):
      Number of clusters.
  
  Yields:
    (np.ndarray)
      An integer array of cluster indices (start from 0) assigned to each
      data points, the array size should be N.
  """

  np.random.seed(1337)

  # Implement k-means here
  # initialize k centroid
  #max value 255
  k=num_clusters
  centroid=sample(X,k)
  #the label you guess
  guess=np.random.randint(1,size=X.shape[0])
  guess=assign_point(X,guess,centroid)
  i=0
  while True:
    # Take one k-means step
    #assign each point to its closest centroid
    newguess=assign_point(X,guess,centroid)
    #compute the new centroid mean of each cluster
    centroid=compute_new_centroid(X,guess,centroid,k)
    # Yield cluster index for each point in X assigned by this k-means step
    yield guess # change this
    i+=1
    # You should implement a stopping criterion
    if i==10:
      return
    guess=newguess

"""## Test your k-means"""

#load the dataset
X, _ = load_mnist()
num_clusters = 10
print(X)
for i, cluster_indices in enumerate(kmeans(X, num_clusters)):
  mean_distance = compute_mean_distance(X, cluster_indices, num_clusters)
  score = your_score(mean_distance)
  print(f'step: {i}, mean_distance: {mean_distance}, score: {score}')
  
  display(sample_from_clusters(X, cluster_indices, num_clusters))
  if score == 100:
    break

