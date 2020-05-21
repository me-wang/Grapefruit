import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Basic Graph
x = [0, 1, 2, 3, 4]
y = [0, 2, 4, 6, 8]

# Resize your Graph
plt.figure(figsize=(6, 4.5), dpi=100)

# Shorthand notation
# fmt = '[color][marker][line]'
plt.plot(x, y, 'r*--', label='2x')

x2 = np.arange(0, 4, 0.5)
print(x2)
# Plot part of the graph as line
plt.plot(x2[:5], x2[:5]**2, 'b', label='x^2')
# Plot remainder of graph as a dot
plt.plot(x2[4:], x2[4:]**2, 'b--')

# Add a title (specify font parameters with fontdict)
plt.title('Basic Graph!', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})
plt.xlabel('X Axis', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 10})
plt.ylabel('y Axis', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 10})

plt.xticks([0, 1, 2, 3, 4])
plt.yticks([0, 2, 4, 6, 6.5, 8])
# Add a legend
plt.legend()
# Save figure
plt.savefig('./GraphClass/mygraph.png', dpi=100)
# show plot
plt.show()
