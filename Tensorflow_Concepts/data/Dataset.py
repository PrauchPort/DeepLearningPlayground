import tensorflow as tf
import collections

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])

for element in dataset:
    print(element)

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(lambda x: x*2)

print(list(dataset.as_numpy_iterator()))


a = 1
b = 2.0
c = (1, 2)
d = {"a": (2, 2), "b": 3}
Point = collections.namedtuple("Point", ['x', 'y'])
e = Point(1, 2)
f = tf.data.Dataset.range(10)


dataset = tf.data.Dataset.range(100)


def dataset_fn(ds):
    return ds.filter(lambda x: x < 5)


dataset = dataset.apply(dataset_fn)

print(list(dataset.as_numpy_iterator()))
