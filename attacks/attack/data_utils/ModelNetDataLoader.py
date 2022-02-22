import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


classes = {'airplane' : 0, 'bathtub' : 1, 'bed': 2, 'bench': 3, 'bookshelf':4, 'bottle':5, 'bowl' : 6, 'car':7, 'chair' : 8, 'cone' : 9, 'cup':10, 'curtain':11
           , 'desk':12, 'door':13, 'dresser':14, 'flower_pot':15, 'glass_box':16, 'guitar':17, 'keyboard':18, 'lamp':19, 'laptop':20, 'mantel':21, 'monitor':22,
           'nightstand':23, 'person':24, 'piano':25, 'plant':26, 'radio':27, 'range_hood':28, 'sink':29, 'sofa':30, 'stairs':31, 'stool':32, 'table':33, 'tent':34,
           'toilet':35, 'tv_stand':36, 'vase':37, 'wardrobe':38, 'xbox' : 39}

####load data

###to correctly run the code please pay attention to the folder structure and file names. you must first have the Modelnet40 train and test data.
def load_data(main_path, data_path, label_path, mode = 'train'):
    BASE_DIR = main_path
    DATA_dir = os.path.join(BASE_DIR, '')

    if mode == 'train':
        X_file = os.path.join(DATA_dir, data_path)
        y_file = os.path.join(DATA_dir, label_path)
    elif mode == 'test':
        X_file = os.path.join(DATA_dir, data_path)
        y_file = os.path.join(DATA_dir, label_path)

    return np.load(X_file), np.load(y_file).astype('int')


####some utility functions for sampling, rotating and adding noise to point clouds


##randomly choose #sample_num points
class PointSampler(object):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __call__(self, points):
        new_points = points[np.random.choice(1024, self.sample_num, replace=False)]

        return new_points


##always pick first #sample_num points (this is used in the authors implementation)
class PointCut(object):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __call__(self, points):
        new_points = points[:self.sample_num]

        return new_points

## normalize point cloud to unit sphere
class Normalize(object):
    def __call__(self, verts):
        normalized_points = verts - np.mean(verts, axis=0)
        max_norm = np.max(np.linalg.norm(normalized_points, axis=1))

        normalized_points = normalized_points / max_norm

        return normalized_points

## randomly rotate point cloud

class RandomRotate(object):

    def __call__(self, verts):
        theta = 2 * np.random.uniform() * np.pi
        rotation_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        rotated = np.matmul(verts, rotation_mat)

        return rotated

# add a random gaussian noise to point cloud
class RandomNoise(object):

    def __call__(self, verts):
        noise = np.random.normal(0, 0.01, verts.shape)
        noise = np.clip(noise, -0.05, 0.05)
        return verts + noise

class toTensor(object):

    def __call__(self, verts):
        return torch.from_numpy(verts)


##in test time we dont rotate and add noise



class ModelNetDataLoader(Dataset):
  def __init__(self, X, y, transform):
    self.X = X
    self.y = y

    self.transform = transform
  def __len__(self):
    return len(self.y)


  def __getitem__(self, idx):
    x = self.X[idx]
    x = self.transform(x)
    y = torch.tensor(self.y[idx])

    return x, y


if __name__ == '__main__':
    train_X, train_y = load_data()
    test_X, test_y = load_data(mode='test')

    default_transform = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            toTensor()]
    )

    train_transform = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            RandomNoise(),
            RandomRotate(),
            toTensor()]
    )

    trainset = PointnetDataset(train_X, train_y, transform=train_transform)
    testset = PointnetDataset(test_X, test_y, transform=default_transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)

    print(trainset.__len__(), testset.__len__())