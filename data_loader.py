import os
import pickle as pkl


def get_data(root='.'):
	with open(os.path.join(root, 'x_train.pkl'), 'rb') as f:
	    x_train = pkl.load(f)
	with open(os.path.join(root, 'y_train.pkl'), 'rb') as f:
	    y_train = pkl.load(f)
	with open(os.path.join(root, 'x_test.pkl'), 'rb') as f:
	    x_test = pkl.load(f)

	return x_train, y_train, x_test

