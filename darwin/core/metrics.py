import matplotlib
from matplotlib import pyplot as plt
import scipy.sparse as sparse

# from Antipasti.io.preprocessing import as_function_over_axes
# import tensorflow as tf
from sklearn.metrics import adjusted_rand_score
from scipy.ndimage import watershed_ift, gaussian_filter
from scipy.ndimage.measurements import  label
import kruskals_ws as kw
import numpy as np


matplotlib.rcParams.update({'font.size': 5})
np.random.seed(1236)
fixed_rand = np.random.rand(10000, 3)
fixed_rand[0, :] = 0
rand_cm = matplotlib.colors.ListedColormap(fixed_rand)


def edges_to_rand_i(edge_pred, gt_label):
    ws = kw.Watershredder(edge_pred.shape[-3:-1])
    ws.execute_kruskal_updates(edge_pred)
    a = ws.get_display_label_image()
    rand = adapted_rand(a, gt_label)
    print 'rand', rand
    return a, ws


# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py
def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    if np.any(seg == 0):
        print 'waarning zeros in seg, treat as background'
    if np.any(gt == 0):
        print 'waarning zeros in gt, 0 labels will be ignored'

    if np.all(seg == 0) or np.all(gt == 0):
        print 'all labels 0,  fake rand this should not be here. Check in segmentation script'
        if all_stats:
            return (0, 0, 1)
        else:
            return 0

    # boundaries = dp.segmenation_to_membrane_core(gt.squeeze())[0]
    # boundaries = binary_dilation(boundaries, iterations=1)
    # gt[boundaries, 0] = 0

    # print 'after gt', np.sum(gt == 0)

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)

    # mask to foreground in A
    mask = (segA > 0)
    segA = segA[mask]
    segB = segB[mask]
    n = segA.size  # number of nonzero pixels in original segA

    # print 'n', n
    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                             shape=(n_labels_A, n_labels_B),
                             dtype=np.uint64)

    # print 'pij'
    # print p_ij.todense()

    # In the paper where adapted rand is proposed, they treat each background
    # pixel in segB as a different value (i.e., unique label for each pixel).
    # To do this, we sum them differently than others

    B_nonzero = p_ij[:, 1:]             # ind (label_gt, label_seg), so ignore 0 seg labels
    B_zero = p_ij[:, 0]

    # this is a count
    num_B_zero = B_zero.sum()

    # sum of the joint distribution
    #   separate sum of B>0 and B=0 parts
    sum_p_ij = (B_nonzero).power(2).sum() + num_B_zero

    # print 'sum pij', sum_p_ij

    # these are marginal probabilities
    a_i = p_ij.sum(1)           # sum over all seg labels overlapping one gt label (except 0 labels)
    b_i = B_nonzero.sum(0)
    # print 'ai', a_i
    # print 'bi', b_i

    sum_a = np.power(a_i, 2).sum()
    sum_b = np.power(b_i, 2).sum() + num_B_zero

    precision = float(sum_p_ij) / sum_b
    recall = float(sum_p_ij) / sum_a

    fScore = 2.0 * precision * recall / (precision + recall)

    if all_stats:
        return (fScore, precision, recall)
    else:
        return fScore



if __name__ == '__main__':
    el = 20
    a = np.ones((el, el, 3), dtype=np.float32)

    a[:, :, 2] = 0
    a[el/2, :, 2] = 10.
    a[:, :, 1] = np.random.random((el, el)) / 1000.

    a[el/10, el/10, 0] = 0
    a[-el/10, -el/10, 0] = 0
    target = np.zeros_like(a[..., 2])
    target[:el/2, :] = 1
    target += 1

    # a[0, :5, :] = 1
    # for a_i in a
    for i in [1, 2]:
        a[:, :, i] = gaussian_filter(a[:, :, i], 5)

    f, ax = plt.subplots(3,3)
    ax[0, 0].imshow(a[..., 0], interpolation='None', cmap='gray')
    ax[0, 1].imshow(a[..., 1], interpolation='None', cmap='gray')
    ax[0, 2].imshow(a[..., 2], interpolation='None', cmap='gray')
    b, ws= edges_to_rand_i(a, target)
    ax[1,0].imshow(b, interpolation='None', cmap=rand_cm)
    ax[1,1].imshow(target, interpolation='None', cmap=rand_cm)
    plt.show()

    exit()
    import numpy as np



    print adjusted_rand_loss(a, target.astype(np.uint16))













