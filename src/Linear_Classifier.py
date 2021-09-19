import numpy as np
import cv2

# initialize the class labels and set the seeds of pseudorandom
# number of generator so we can reproduce our results 
labels = ['dogs', 'cat', 'panda']
np.random.seed(1)
#print(labels)

W = np.random.randn(3, 3702)
b = np.random.randn(3)

# load our example, resize it
# and then vectorize it
orig = cv2.imread('pyimagesearch/datasets/animals/dog.jpg')
image = cv2.resize(orig, (32,32))

#scores = W.dot(image) + b

#for (label, score) in zip(labels, scores):
#    print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score in image as our prediction
#cv2.putText(orig, 'Label: {}'.format(labels[np.argmax(scores)]), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
#            0.9, (0, 255, 0), 2)

cv2.imshow('Image', orig)
cv2.waitkey(0)


