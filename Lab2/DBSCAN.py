# Source: https://blog.dominodatalab.com/topology-and-density-based-clustering/

from sklearn.clusters import DBSCAN
import pandas as pd

data = pd.read_csv("Wholesale.csv")
# Drop non-continuous variables
data.drop(["Channel", "Region"], axis = 1, inplace = True)

data = data[["Grocery", "Milk"]]
data = data.as_matrix().astype("float32", copy = False)

stscaler = StandardScaler().fit(data)
data = stscaler.transform(data)

dbsc = DBSCAN(eps = .5, min_samples = 15).fit(data)

labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True

from sklearn.datasets import make_moons
# moons_X: Data, moon_y: Labels
moons_X, moon_y = make_moons(n_samples = 2000)

def add_noise(X,y, noise_level = 0.01):
  #The number of points we wish to make noisy
  amt_noise = int(noise_level*len(y))
  #Pick amt_noise points at random
  idx = np.random.choice(len(X), size = amt_noise)
  #Add random noise to these selected points
  noise = np.random.random((amt_noise, 2) ) -0.5
  X[idx,:] += noise
  return X

def knn_estimate(data, k, point):
  n,d = data.shape
  #Reshape the datapoint, so the cdist function will work
  point = point.reshape((1,2))
  #Find the distance to the kth nearest data point
  knn = sorted(reduce(lambda x,y: x+y,cdist(data, point).tolist()))[k+1]
  #Compute the density estimate using the mathematical formula
  estimate = float(k)/(n*np.power(knn, d)*np.pi)
  return estimate

def makeCraters(inner_rad = 4, outer_rad = 4.5, donut_len = 2, inner_pts = 1000, outer_pts = 500):
  #Make the inner core
  radius_core = inner_rad*np.random.random(inner_pts)
  direction_core = 2*np.pi*np.random.random(size = inner_pts)
  #Simulate inner core points
  core_x = radius_core*np.cos(direction_core)
  core_y = radius_core*np.sin(direction_core)
  crater_core = zip(core_x, core_y)
  #Make the outer ring
  radius_ring = outer_rad + donut_len*np.random.random(outer_pts)
  direction_ring = 2*np.pi*np.random.random(size = outer_pts)
  #Simulate ring points
  ring_x = radius_ring*np.cos(direction_ring)
  ring_y = radius_ring*np.sin(direction_ring)
  crater_ring = zip(ring_x, ring_y)
  return np.array(crater_core), np.array(crater_ring)
