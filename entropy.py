from math import log2
from collections import Counter

# Returns (shannon entropy, min entropy)
# Partial explanation: https://crypto.stackexchange.com/a/67408
def entropy(bits, n):
  def chunks(arr, chunk_size):
    return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

  arr = chunks(bits, n)
  counts = Counter(arr)
  total = len(arr)

  shannon_entropy = 0
  for count in counts.values():
    prob = count / total
    shannon_entropy -= prob * log2(prob)

  min_entropy = -max(log2(count / total) for count in counts.values())
  return (shannon_entropy, min_entropy)



import os
z = []

for filename in os.listdir("data3"):
  filepath = f"data3/{filename}"

  with open(filepath, "r") as f:
    data = f.read()

  # for line in lines:
  #   print(line, entropy(line.replace(" ", ""), 1))

  n = len(data.split("\n", 1)[0].split())
  ent = entropy("".join(data.split()), 1)

  z.append((n, ent[0], ent[1]))

  print(filepath, ent)


import matplotlib.pyplot as plt

z.sort(key = lambda x: x[0])
plt.plot([x[0] for x in z], [x[1] for x in z])
plt.plot([x[0] for x in z], [x[2] for x in z])
plt.show()
