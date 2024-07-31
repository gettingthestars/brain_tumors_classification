# import libraries
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix

# mount the google drive to colab
from google.colab import drive
drive.mount('/content/drive')

# train and test datasets directories
train_no_dir = "brain_tumors/train/no_tumor/"
train_pituitary_dir = "brain_tumors/train/pituitary_tumor/"
train_meningioma_dir = "brain_tumors/train/meningioma_tumor/"
train_glioma_dir = "brain_tumors/train/glioma_tumor/"
test_no_dir = "brain_tumors/test/no_tumor/"
test_pituitary_dir = "brain_tumors/test/pituitary_tumor/"
test_meningioma_dir = "brain_tumors/test/meningioma_tumor/"
test_glioma_dir = "brain_tumors/test/glioma_tumor/"

# parent directory 
parent_dir = "/content/drive/MyDrive/Machine Learning/Machine Learning Datasets/"
  
# full paths
path_train_no = os.path.join(parent_dir, train_no_dir)#join the paths together
path_train_pituitary = os.path.join(parent_dir, train_pituitary_dir)
path_train_meningioma = os.path.join(parent_dir, train_meningioma_dir)
path_train_glioma = os.path.join(parent_dir, train_glioma_dir)
path_test_no = os.path.join(parent_dir, test_no_dir)
path_test_pituitary = os.path.join(parent_dir, test_pituitary_dir)
path_test_meningioma = os.path.join(parent_dir, test_meningioma_dir)
path_test_glioma = os.path.join(parent_dir, test_glioma_dir)

# create a list of filenames for each folder
# the glob module finds all the pathnames matching a specified pattern according to the rules
flist_train_no = glob.glob(path_train_no + '*.jpg')
flist_train_pituitary = glob.glob(path_train_pituitary + '*.jpg')
flist_train_meningioma = glob.glob(path_train_meningioma + '*.jpg')
flist_train_glioma = glob.glob(path_train_glioma + '*.jpg')
flist_test_no = glob.glob(path_test_no + '*.jpg')
flist_test_pituitary = glob.glob(path_test_pituitary + '*.jpg')
flist_test_meningioma = glob.glob(path_test_meningioma + '*.jpg')
flist_test_glioma = glob.glob(path_test_glioma + '*.jpg')


# define a function to loop through all images in a folder, resize them (make images have 
#the same size to train models,total pixels should be the same), and add them to a single numpy array(convert image 
# into numpy array)
def images_to_array(filelist):
  # create an empty list to contain images
  images = list()
  for fname in filelist:
    # load an image
    image = load_img(fname, color_mode = "grayscale")
    # resize the image
    image = image.resize((250, 250))# if the size is too small, then u will lose substantial information
    # convert the image to a numpy array
    image = img_to_array(image)
    # use the squeeze() function to reduce dimensions from (250, 250, 1) to (250, 250),'1' denotes grayscale
    # if it is colorful image, then it should be (250,250,3)
    image = image.squeeze()
    # append to the images list
    images.append(image)
  # convert the images list to a numpy array
  images = np.array(images)
  return images


# perform the above defined function to process images in each folder
images_train_no = images_to_array(flist_train_no)
images_train_pituitary = images_to_array(flist_train_pituitary)
images_train_meningioma = images_to_array(flist_train_meningioma)
images_train_glioma = images_to_array(flist_train_glioma)
images_test_no = images_to_array(flist_test_no)
images_test_pituitary = images_to_array(flist_test_pituitary)
images_test_meningioma = images_to_array(flist_test_meningioma)
images_test_glioma = images_to_array(flist_test_glioma)


# concatenate arrays along the rows to form train and test features
X_train = np.concatenate((images_train_no, images_train_pituitary, images_train_meningioma, images_train_glioma), axis = 0)
X_test = np.concatenate((images_test_no, images_test_pituitary, images_test_meningioma, images_test_glioma), axis = 0)

# take a look at the shapes of X_train and X_test
X_train.shape
print('\n')
X_test.shape

# save X_train and X_test arrays so that we won't have to process the images
np.save(parent_dir + 'brain_tumors/X_train.npy', X_train)
np.save(parent_dir + 'brain_tumors/X_test.npy', X_test)

# load the saved X_train and X_test numpy arrays
X_train = np.load(parent_dir + 'brain_tumors/X_train.npy')
X_test = np.load(parent_dir + 'brain_tumors/X_test.npy')

# reshape the above arrays from (obs, 250, 250) to (obs, 62500),meaning you have 62500 features
#because sklearn models usually only accept data with 2 dimensions (i.e., matrix)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# show the reshaped X_train and X_test shapes
# now we have a large number of features (62500)!
X_train.shape
print('\n')
X_test.shape

# concatenate arrays along the rows to form train and test targets
# 0 - no tumor
# 1 - pituitary tumor
# 2 - meningioma tumor
# 3 - glioma tumor
y_train = np.concatenate((np.full(len(flist_train_no), 0), 
                         (np.full(len(flist_train_pituitary), 1)),
                         (np.full(len(flist_train_meningioma), 2)),
                         (np.full(len(flist_train_glioma), 3))),
                         axis = 0)
y_test = np.concatenate((np.full(len(flist_test_no), 0), 
                         (np.full(len(flist_test_pituitary), 1)),
                         (np.full(len(flist_test_meningioma), 2)),
                         (np.full(len(flist_test_glioma), 3))),
                         axis = 0)

# apply standardization on train and test sets
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)#fit_transform only on the training set
X_test = scalar.transform(X_test)#no fit because we don't want the train set to impact the test set


# train a binary classifier
y_train_any_tumor = (y_train > 0) # create a binary target for any brain tumor (0: no tumor; 1: any tumor type)
sgd_clf = SGDClassifier() # stochastic gradient descent (sgd) classifier
sgd_clf.fit(X_train, y_train_any_tumor) # fit the model

# dichotomize the test targets
y_test_any_tumor = (y_test > 0) # create a binary target for any brain tumor (0: no tumor; 1: any tumor type)


# do a prediction with the fitted model
some_img = X_test[0] # get the first instance from the train set
plt.imshow(some_img.reshape(250, 250), cmap = 'binary') # plot the first instance
plt.axis('off')
print('\n')
print('true label: ', y_test_any_tumor[0]) # true label
y_pred_some_img = sgd_clf.predict([some_img]) # predict the label for the instance
print('\n')
print('predicted label: ', y_pred_some_img[0])

# predicted tumor rate in the test set
y_test_pred_any_tumor = sgd_clf.predict(X_test)
sum(y_test_pred_any_tumor) / len(X_test)

# compare to the baseline tumor rate in the test set
sum(y_test_any_tumor) / len(y_test_any_tumor)

# plot the confusion matrix
plot_confusion_matrix(sgd_clf, X_test, y_test_any_tumor)

# report precision, recall, and f1 score
precision_score(y_test_any_tumor, y_test_pred_any_tumor) # precision = TP / (TP + FP)
print('\n')
recall_score(y_test_any_tumor, y_test_pred_any_tumor) # recall = TP / (TP + FN)
print('\n')
f1_score(y_test_any_tumor, y_test_pred_any_tumor) # f1 score = 2 * precision * recall / (precision + recall)


# decision scores and default threshold
y_score = sgd_clf.decision_function([some_img])
y_score
print('\n')
threshold = 0 # default threshold
y_some_img_pred = (y_score > threshold)
y_some_img_pred

# reset threshold
threshold = -70000000
y_some_img_pred = (y_score > threshold)
y_some_img_pred

# obtain all decision scores (default method is to predict class, not decision function scores)
y_scores = sgd_clf.decision_function(X_test)

# obtain the precision recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test_any_tumor, y_scores)

# construct a function to plot precision recall vs threshold curve
def plot_precision_recall_vs_threshold(precisons, recalls, thresholds):
  plt.plot(thresholds, precisions[:-1], 'b--', label = 'Precision')
  plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall')
  plt.legend()
  plt.xlabel('Threshold')
plot_precision_recall_vs_threshold(precisions, recalls, thresholds) # use the function to plot the curve

# plot precision vs recall
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')

# obtain the roc curve
# fpr: false positive rate. fpr = FP / (FP + TN) = 1 - specificity
# tpr: true positive rate. tpr = TP / (TP + FN) = recall = sensitivity
fpr, tpr, thresholds = roc_curve(y_test_any_tumor, y_scores)

# construct a function to plot roc curve
def plot_roc_curve(fpr, tpr, label = None):
  plt.plot(fpr, tpr, linewidth = 2, label = label)
  plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
  plt.xlabel('False positive rate (1 - specificity)')
  plt.ylabel('True positive rate (recall or sensitivity)')
  plt.legend()
plot_roc_curve(fpr, tpr)

# obtain the auc (area under the curve)
roc_auc_score(y_test_any_tumor, y_scores)

# train a random-forest classifier and compare modeling results
forest_clf = RandomForestClassifier()
forest_clf.fit(X_train, y_train_any_tumor)
y_probas_forest = forest_clf.predict_proba(X_test)


# y_probas_forest has two columns, proba for negative class and proba for postive class
y_probas_forest
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class


# plot random forest againt sgd in one roc curve graph
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test_any_tumor, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label = 'SDG')
plot_roc_curve(fpr_forest, tpr_forest, label = 'Random forest')
plt.legend(loc = 'lower right')


# obtain the auc for the random forest model (area under the curve)
roc_auc_score(y_test_any_tumor, y_scores_forest)

# multiclass classification
svm_clf = SVC() # use support vector machine classifier
svm_clf.fit(X_train, y_train)
print('\n')
svm_clf.predict([some_img])

# take a look at the decision scores for each class
# the predicted label correponds to the label with the highest decision score
some_img_scores = svm_clf.decision_function([some_img])
some_img_scores

# the index for the highest score
print(np.argmax(some_img_scores))
print('\n')

# predicted label
svm_clf.predict([some_img])[0]

# multilabel classification (each label is binary)
# meningioma and pituitary are often benign
# glioma is often malicious
y_train_any_tumor = (y_train > 0) # an indicator for any tumor
y_train_benign = (y_train < 3) # an indicator for benign tumor (including no tumor)
y_multilabel = np.c_[y_train_any_tumor, y_train_benign] # column combine the two labels
y_multilabel[:5]

# create multilabel test labels
y_test_any_tumor = (y_test > 0) # an indicator for any tumor
y_test_benign = (y_test < 3) # an indicator for benign tumor (including no tumor)
y_multilabel_test = np.c_[y_test_any_tumor, y_test_benign] # column combine the two labels

# train a k-nearest neighbor classifier
knn_clf = KNeighborsClassifier(n_neighbors = 3)
knn_clf.fit(X_train, y_multilabel)
print('\n')
knn_clf.predict([some_img])

# compute the average f1 score cross labels
y_test_knn_pred = knn_clf.predict(X_test)
f1_score(y_multilabel_test, y_test_knn_pred, average = 'macro') # simple average
print('\n')
f1_score(y_multilabel_test, y_test_knn_pred, average = 'weighted') # weighted average (weighted by instances with each target label)





















