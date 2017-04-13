from Antipasti.io.preprocessing import as_function_over_axes
# import tensorflow as tf
from sklearn.metrics import adjusted_rand_score
from scipy.ndimage import watershed_ift, label


def _adjusted_rand_loss(prediction, target, threshold=0.5):
    def adjusted_rand_loss_i(prediction, target, threshold=0.5):
        # Prediction.shape = (nb, row, col, nc)
        thresholded_prediction = (prediction > threshold)
        # connected components, background must be 0
        connected_components = label(thresholded_prediction)[0].astype(np.int)

        # get rid of boundaries
        print type(connected_components), connected_components.dtype, type(
            thresholded_prediction), thresholded_prediction.dtype
        predicted_segmentation = watershed_ift(thresholded_prediction.astype(np.uint8),
                                               connected_components)
        return adjusted_rand_score(target.ravel(), predicted_segmentation.ravel())
    adj = 0
    for pred_sice, target_slice in zip(prediction, target):
        adj += abs(adjusted_rand_loss_i(pred_sice, target_slice))
    return adj


def adjusted_rand_loss(prediction, target, threshold=0.5):
    # Prediction.shape = (nb, row, col, nc)
    thresholded_prediction = (prediction > threshold)

if __name__ == '__main__':
    import numpy as np
    a = np.ones((1, 10,10, 1))
    a[0, 5, :] = 0
    target = np.zeros_like(a)
    a[0, :5, :] = 1
    print adjusted_rand_loss(a, target.astype(np.uint16))
