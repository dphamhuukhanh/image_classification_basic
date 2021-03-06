from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagsearch.preprocessing import SimplePreprocessor
from pyimagsearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
#import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset', 
                required=True, help='path to input dataset')
ap.add_argument('-k', '--neighbors',
                type=int, default=1, help='# of nearest neighbors for classification')
ap.add_argument("-j", "--jobs",
                type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


# grab the list of images
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

# initialize the image preprocessor, load the dataset from disk, resize images
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of images
print('[INFO] features matrix: {:.1f}MB'.format(data.nbytes / (1024*1000.0)))


# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier
print('[INFO] evaluating k-NN classifier...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['job'])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))