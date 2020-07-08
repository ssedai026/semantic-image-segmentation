import numpy as np

def dice_coefficient(pred, gt):
    """
    Computes dice coefficients between two masks
    :param pred: predicted masks - [0 ,1]
    :param gt: ground truth  masks - [0 ,1]
    :return: dice coefficient
    """
    d = (2 * np.sum(pred * gt) + 1) / ((np.sum(pred) + np.sum(gt)) + 1)

    return d


def dice_coefficient_batch(pred, gt, eer_thresh=0.5):
    dice_all = []
    n = pred.shape[0]
    for i in range(n):
        seg = pred[i, :, :]
        seg = (seg >= eer_thresh).astype(np.uint8)
        gtd = gt[i, :, :]

        d = dice_coefficient(seg, gtd)
        dice_all.append(d)

    return dice_all




def compute_dice_metric( preds, labels,  eval_class_indices=None):

    """
    Evaluates the segmentation by computing dice coefficient
    :param preds: NxHxWxC prediction masks  where pixel values are between [0,1] or
    :param labels: NxHxWxC ground truth masks where pixel values are between [0,1]
    :param eval_class_indices, indices of classes to be evaluated, if None, all indices will be evaluated
    """

    if(eval_class_indices is None):
        eval_class_indices = range(preds.shape[3])

    evals = [ dice_coefficient_batch(preds[:,:,:,i], labels[:,:,:,i])for i in eval_class_indices]
    evals = [np.expand_dims(np.asanyarray(e), -1) for e in evals]
    dices = np.concatenate(evals, axis=1) #( N,C) matrix
    dices_mean  = np.mean(dices, axis=0) #(C,)

    res_verbose = ''
    for c,ev in zip(eval_class_indices, dices_mean):
        res_verbose += 'Class ' + str(c) + ' DC='+str(ev)+ '\n'


    return dices_mean, dices, res_verbose

if (__name__=='__main__'):


    labels = np.random.random((10,50,50,9))
    preds = np.random.random((10, 50, 50, 9))
    dices_mean, dices, res_verbose = compute_dice_metric(preds, labels)
    print(res_verbose)
