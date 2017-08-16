import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# DataLoader class: need to customize according to your dataset
class DataLoader(object):
    """ Example data loader class to load and process dataset.

    Note: In this tutorial, we load the whole dataset to memory.
          You need to customize this class based on your dataset.
    """
    def __init__(self, data_dir, width, height, channel):
        # variable to hold the whole dataset
        self.dataset = read_data_sets(data_dir, one_hot=False)
        # basic stats of the dataset
        self.num = self.dataset.train.images.shape[0]
        self.h = height
        self.w = width
        self.c = channel
        # counter that indicates which image to load
        self._idx = 0
        
    def next_batch(self, batch_size):
        """ Load next batch of training data """
        images_batch = np.zeros((batch_size, self.h, self.w, self.c)) 
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            # when your dataset is huge, you may need to load images on the fly
            images_batch[i, ...] = self.dataset.train.images[self._idx].reshape((self.h, self.w, self.c))
            labels_batch[i, ...] = self.dataset.train.labels[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        return images_batch, labels_batch
    
    def load_test(self):
        """ Load testing data """
        return self.dataset.test.images.reshape((-1, self.h, self.w, self.c)), self.dataset.test.labels
