'''
How good is your segmentation?

As discussed in lecture, there are a variety of ways to determine the performance of a segmentation algorithm.
A simple way is to compute a confusion matrix, and statistics like accuracy, sensitivity, and specificity.

In this week’s lab materials on Moodle you will find a ground truth mask, groundTruth.png.
This mask indicates which pixels should be part of the correct segmentation, and which are not.

Let's load it and compare it side-by-side with your segmentation mask:
'''

from thresholding import label_img_filtered

from skimage import io, img_as_float,img_as_ubyte
import matplotlib.pyplot as plt
from skimage import measure, color

result = img_as_float(label_img_filtered > 0)
gt = io.imread('GroundTruth.png')
gt = img_as_float(color.rgb2gray(gt))

fig, ax = plt.subplots(ncols=2, figsize=(18, 6))
ax[0].imshow(gt, cmap='gray')
ax[0].axis('off'), ax[0].set_title('Ground-truth')
ax[1].imshow(result, cmap='gray')
ax[1].axis('off'), ax[1].set_title('Our result')
fig.tight_layout()
plt.show();


'''
Task 2.1: Computing classification-based metrics

Use this image to compute the true positives, true negatives, false positives, and false negatives of your segmentation.
If you’re unsure about their definition, consult the lecture slides.

Once you have these, determine the accuracy, sensitivity, and specificity of your segmentation.

Sensitivity =  76.695777006819
Specificity =  99.97291244111759
Accuracy =  98.75469571750564
'''

##############################################################################
# TODO: Compute classification-based metrics                                 #
##############################################################################
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

preds_list = gt.reshape(-1)
labels_list = result.reshape(-1)

threshold=0.5
preds_list = preds_list >= threshold
tn, fp, fn, tp = confusion_matrix(labels_list, preds_list).ravel()


sensitivity = tp/ (tp + fn)
print("Sensitivity = ",sensitivity*100)

specificity = tn/ (tn + fp)
print("Specificity = ",specificity*100)

accuracy = accuracy_score(labels_list, preds_list)
print("Accuracy = ",accuracy*100)

# Sensitivity =  76.695777006819
# Specificity =  99.97291244111759
# Accuracy =  98.75469571750564
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################


'''
Task 2.2: Computing the Dice-Sørensen Coefficient (DSC)

Another very important metric is the DSC coefficient. Compute it using the previously determined variables.

Dice-Sørensen Coefficient =  86.57079197893458
'''

##############################################################################
# TODO: Compute DSC                                                          #
##############################################################################
dsc = 2*tp/ (2*tp + fp + fn)
print("DSC coefficient = ",dsc*100)
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
