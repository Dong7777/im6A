import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker, cm
df = pd.read_excel('热图.xlsx')
x_axis = ['h_b','h_l','h_k','m_b','m_h','m_k','m_l','m_t','r_b','r_k','r_t']
y_axis = ['h_b','h_k','h_l','m_b','m_h','m_k','m_l','m_t','r_b','r_k','r_t']
a=np.array(df)
b=a[:,1:]
b=np.array(b).astype(float)

fig, ax = plt.subplots()
im = ax.imshow(b,cmap=cm.Purples_r,vmin=40, vmax=90)

ax.set_xticks(np.arange(len(x_axis)))
ax.set_yticks(np.arange(len(y_axis)))
# ... and label them with the respective list entries
ax.set_xticklabels(x_axis)
ax.set_yticklabels(y_axis)
plt.setp(ax.get_yticklabels(),fontsize='13')
plt.setp(ax.get_xticklabels(),fontsize='13')

# 添加每个热力块的具体数值
# Loop over data dimensions and create text annotations.
for i in range(len(x_axis)):
    for j in range(len(y_axis)):
        text = ax.text(j, i, b[i, j],
                       ha="center", va="center", color="k")

fig.tight_layout()

plt.colorbar(im)
plt.show()


