from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import pickle
import numpy as np
from tensorflow.python.keras.backend import expand_dims
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array
import matplotlib.pyplot as plt
import os
from time import sleep
import math

from tensorflow.python.ops import math_ops

def get_extract_model():
    vgg16_model = VGG16(weights= 'imagenet')
    return Model(inputs = vgg16_model.inputs, outputs = vgg16_model.get_layer('fc1').output)

def image_processing(img):
    img = img.resize((224, 224))
    img = img.convert('RGB')
    x = image.img_to_array(img)
    x = expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    img = Image.open(image_path)
    img = image_processing(img)
    # vectorize
    vector = model.predict(img)
    # normalize
    vector = vector/np.linalg.norm(vector)
    return vector

# searched_image
search_image = 'panda10.jpg'
model = get_extract_model()
search_vector = extract_vector(model, search_image)

# vectorize images in datasets and store them in a .pkl file
vectors = []
paths = []
for image_path in os.listdir('pyimagesearch/datasets/animals/'):
    image_path_full = 'pyimagesearch/datasets/animals/' + image_path
    vectors.append(extract_vector(model=model,image_path=image_path_full))
    paths.append(image_path_full)

# pickle.dump(vectors, open('vectors.pkl', 'wb'))
# pickle.dump(paths, open('paths.pkl', 'wb'))

# Euclidian distances
distances = [np.linalg.norm(vector - search_vector) for vector in vectors]

#search engine
K = int(input("[INPUT] Number of wanted images:"))
ids = np.argsort(distances)[:K]

#output
nearest_images = [(paths[id], distances[id]) for id in ids]
#plot
axes = []
grid_size = 4
fig = plt.figure(figsize=(10,5))

for id in range(K):
    axes.append(fig.add_subplot(grid_size, grid_size, id+1))
    axes[-1].set_title(nearest_images[id][1])
    plt.imshow(Image.open(nearest_images[id][0]))

fig.tight_layout()
plt.show()





