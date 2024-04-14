import numpy as np

filepath = "AI_2qubits_training_data.txt"

with open(filepath, "r") as f:
  data = f.read()

def parse_line(line):
  a, b = line.split()

  return (list(map(int, a)), int(b))

x, y = zip(*(parse_line(line) for line in data.split("\n") if line))

x = np.asarray(x)
y = np.asarray(y) - 1




z = 2 * x[:,::2] + x[:,1::2]

# TODO: Vectorize
probs = np.zeros((z.shape[0], 4))
for i, row in enumerate(z):
  for j in range(4):
    probs[i,j] = np.sum(row == j) / len(row)


# z = 8 * x[:,::4] + 4 * x[:,1::4] + 2 * x[:,2::4] + x[:,3::4]

# # TODO: Vectorize
# probs2 = np.zeros((z.shape[0], 16))
# for i, row in enumerate(z):
#   for j in range(16):
#     probs2[i,j] = np.sum(row == j) / len(row)

x = np.concatenate((x, probs), axis=1)





# Neural network
import tensorflow as tf
import tensorflow.keras as K
from sklearn.model_selection import train_test_split

y = K.utils.to_categorical(y, num_classes=3)

tf.experimental.numpy.experimental_enable_numpy_behavior()
def shuffle_pairs(x, y):
  doubles = x.reshape(-1, 2)
  tf.random.shuffle(x)
  return (doubles.flatten(), y)
  # tf.random.shuffle(x)
  # return x, y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat(10).shuffle(100_000).map(shuffle_pairs).batch(10_000)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10_000)


model = K.models.Sequential([
  K.layers.Input((100 + 4,)),
  K.layers.Dense(64, activation="relu"),
  K.layers.Dense(3, activation="softmax")
])

# model = K.models.Sequential([
#   K.layers.Input((100,)),

#   K.layers.Reshape((100, 1)),
#   K.layers.Conv1D(8, 3, activation="relu"),
#   K.layers.Flatten(),

#   K.layers.Dense(16, activation="relu"),
#   K.layers.Dense(3, activation="softmax")
# ])

# model.summary()
# exit()

model.compile(
  optimizer="adam",
  loss="categorical_crossentropy",
  metrics=["accuracy"]
)

# model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
model.fit(train_ds, validation_data=test_ds, epochs=100)
# model.summary()









# import itertools
# import matplotlib.pyplot as plt


# filepath = "AI_2qubits_training_data.txt"

# with open(filepath, "r") as f:
#   data = f.read()

# def parse_line(line):
#   a, b = line.split()
#   return (a, int(b))

# x, y = zip(*(parse_line(line) for line in data.split("\n") if line))



# def count_pairs(s):
#   def chunks(arr, chunk_size):
#     return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

#   counts = [0] * 4
#   for chunk in chunks(s, 2):
#     counts[int(chunk, 2)] += 1

#   total = sum(counts)
#   return [x / total for x in counts]

# # def chunks(arr, chunk_size):
# #   return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

# # numbers = itertools.chain.from_iterable(chunks(line, 10) for line in x)
# # numbers = list(int(x, 2) for x in numbers)

# # import numpy as np
# # z = np.fft.fft(numbers)
# # print(numbers)

# # plt.plot(z)
# # plt.show()


# xs = []

# fig, axs = plt.subplots(1, 3)
# for i in range(3):
#   x_filtered = [x for x,y in zip(x,y) if y == i + 1]

#   def chunks(arr, chunk_size):
#     return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

#   numbers = itertools.chain.from_iterable(chunks(line, 2) for line in x_filtered)
#   numbers = list(int(x, 2) for x in numbers)

#   xs.append(numbers)

#   # from collections import Counter
#   # print(Counter(numbers).most_common(10))
#   print(count_pairs("".join(x_filtered)))

#   axs[i].scatter(numbers[:-1], numbers[1:], s=1)
#   axs[i].set_aspect(1)

# plt.show()

# # import numpy as np
# # print(np.corrcoef(xs[0][:-1], xs[0][1:]))
# # print(np.corrcoef(xs[1][:-1], xs[1][1:]))
# # print(np.corrcoef(xs[2][:-1], xs[2][1:]))

