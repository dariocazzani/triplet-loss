from tensorflow.examples.tutorials.mnist import input_data # for data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
import numpy as np

def triplet_load_train(batch_size=128):
    batch_anchor_x, batch_anchor_y = mnist.train.next_batch(batch_size)
    batch_positive_x = []
    current = 0
    while len(batch_positive_x) < batch_size:
        x, y = mnist.train.next_batch(1)
        if batch_anchor_y[current] == y:
            # print("anchor label: {} - positive label: {}".format(batch_anchor_y[current], y))
            batch_positive_x.append(np.squeeze(x))
            current += 1

    batch_negative_x = []
    current = 0
    while len(batch_negative_x) < batch_size:
        x, y = mnist.train.next_batch(1)
        if batch_anchor_y[current] != y:
            # print("anchor label: {} - negative label: {}".format(batch_anchor_y[current], y))
            batch_negative_x.append(np.squeeze(x))
            current += 1

    return batch_anchor_x, np.array(batch_positive_x), np.array(batch_negative_x)

def triplet_load_validation(batch_size=128):
    batch_anchor_x, batch_anchor_y = mnist.validation.next_batch(batch_size)
    batch_positive_x = []
    current = 0
    while len(batch_positive_x) < batch_size:
        x, y = mnist.validation.next_batch(1)
        if batch_anchor_y[current] == y:
            # print("anchor label: {} - positive label: {}".format(batch_anchor_y[current], y))
            batch_positive_x.append(np.squeeze(x))
            current += 1

    batch_negative_x = []
    current = 0
    while len(batch_negative_x) < batch_size:
        x, y = mnist.validation.next_batch(1)
        if batch_anchor_y[current] != y:
            # print("anchor label: {} - negative label: {}".format(batch_anchor_y[current], y))
            batch_negative_x.append(np.squeeze(x))
            current += 1

    return batch_anchor_x, np.array(batch_positive_x), np.array(batch_negative_x)

def test_images_and_labels():
    return mnist.test.images, mnist.test.labels

def test_images():
    images, _ = test_images_and_labels()
    return images

if __name__ == '__main__':
    a, p, n = triplet_load()
    print(a.shape)
    print(p.shape)
    print(n.shape)
