from sklearn.datasets import load_digits

dataset=load_digits()

data=dataset.data
target=dataset.target
imgs=dataset.images

print(data.shape,imgs.shape,target.shape)

from matplotlib import pyplot as plt

plt.imshow(imgs[0],cmap='gray')
plt.show()
