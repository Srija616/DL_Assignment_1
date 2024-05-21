import numpy as np

class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True, flatten=True) -> None:
        self.x = x if not flatten else x.reshape(x.shape[0], -1)
        self.x = self.x / 255.0
        self.y = np.eye(np.max(y) + 1)[y]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.n = self.x.shape[0]
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        batch_indices = self.indices[self.i:self.i+self.batch_size]
        batch_x, batch_y = self.x[batch_indices], self.y[batch_indices]
        self.i += self.batch_size
        return batch_x, batch_y