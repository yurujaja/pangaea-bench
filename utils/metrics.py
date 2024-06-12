# -*- coding: utf-8 -*-
''' 
Adapted from: https://github.com/gkakogeorgiou/mados
Modifications: minimal modifications
Authors: Yuru Jia, Valerio Marsocci
'''

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score, hamming_loss, label_ranking_loss, coverage_error
import sklearn.metrics as metr
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Evaluation for Pixel-level semantic segmentation
def Evaluation(y_predicted, y_true):

    micro_prec = precision_score(y_true, y_predicted, average='micro')
    macro_prec = precision_score(y_true, y_predicted, average='macro')
    weight_prec = precision_score(y_true, y_predicted, average='weighted')
    
    micro_rec = recall_score(y_true, y_predicted, average='micro')
    macro_rec = recall_score(y_true, y_predicted, average='macro')
    weight_rec = recall_score(y_true, y_predicted, average='weighted')
        
    macro_f1 = f1_score(y_true, y_predicted, average="macro")
    micro_f1 = f1_score(y_true, y_predicted, average="micro")
    weight_f1 = f1_score(y_true, y_predicted, average="weighted")
        
    subset_acc = accuracy_score(y_true, y_predicted)
    
    iou_acc = jaccard_score(y_true, y_predicted, average='macro')

    info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "weightPrec" : weight_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "weightRec" : weight_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "weightF1" : weight_f1,
            "subsetAcc" : subset_acc,
            "IoU": iou_acc
            }
    
    return info

# Evaluation for Multi-Label classification
def Evaluation_ML(y_predicted, predicted_probs, y_true):

    micro_prec = precision_score(y_true, y_predicted, average='micro')
    macro_prec = precision_score(y_true, y_predicted, average='macro')
    sample_prec = precision_score(y_true, y_predicted, average='samples')
    
    micro_rec = recall_score(y_true, y_predicted, average='micro')
    macro_rec = recall_score(y_true, y_predicted, average='macro')
    sample_rec = recall_score(y_true, y_predicted, average='samples')
        
    macro_f1 = f1_score(y_true, y_predicted, average="macro")
    micro_f1 = f1_score(y_true, y_predicted, average="micro")
    sample_f1 = f1_score(y_true, y_predicted, average="samples")
        
    subset_acc = accuracy_score(y_true, y_predicted)

    hamming = hamming_loss(y_true, y_predicted)
    coverage = coverage_error(y_true, y_predicted)
    rank_loss = label_ranking_loss(y_true, y_predicted)

    info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "samplePrec" : sample_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "sampleRec" : sample_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "sampleF1" : sample_f1,
            "HammingLoss" : hamming,
            "subsetAcc" : subset_acc,
            "coverageError" : coverage,
            "rankLoss" : rank_loss
            }
    return info

def confusion_matrix(y_gt, y_pred, labels, percentage = False):

    # compute metrics
    cm      = metr.confusion_matrix  (y_gt, y_pred)
    f1_macro= metr.f1_score          (y_gt, y_pred, average='macro')
    mRec      = metr.recall_score      (y_gt, y_pred, average='macro')
    OA      = metr.accuracy_score    (y_gt, y_pred)
    UA      = metr.precision_score   (y_gt, y_pred, average=None)
    Rec      = metr.recall_score      (y_gt, y_pred, average=None)
    f1      = metr.f1_score          (y_gt, y_pred, average=None)
    IoC     = metr.jaccard_score     (y_gt, y_pred, average=None)
    mIoC     = metr.jaccard_score    (y_gt, y_pred, average='macro')
      
    # confusion matrix
    sz1, sz2 = cm.shape
    cm_with_stats             = np.zeros((sz1+4,sz2+2))
    cm_with_stats[0:-4, 0:-2] = cm
    cm_with_stats[-3  , 0:-2] = np.round(100*IoC,1)
    cm_with_stats[-2  , 0:-2] = np.round(100*UA,1)
    cm_with_stats[-1  , 0:-2] = np.round(100*f1,1)
    cm_with_stats[0:-4,   -1] = np.round(100*Rec,1)
    
    cm_with_stats[-4  , 0:-2] = np.sum(cm, axis=0) 
    cm_with_stats[0:-4,   -2] = np.sum(cm, axis=1)
    
    # convert to list
    cm_list = cm_with_stats.tolist()
    
    # first row
    first_row = []
    first_row.extend (labels)
    first_row.append ('Sum')
    first_row.append ('Recall')
    
    # first col
    first_col = []
    first_col.extend(labels)
    first_col.append ('Sum')
    first_col.append ('IoU')
    first_col.append ('Precision')
    first_col.append ('F1-score')
    
    # fill rest of the text 
    idx = 0
    for sublist in cm_list:
        if   idx == sz1:
            sublist[-2]  = 'mRec:'
            sublist[-1]  = round(100*mRec,1)
            cm_list[idx] = sublist
        elif   idx == sz1+1:
            sublist[-2]  = 'mIoU:'
            sublist[-1]  = round(100*mIoC,1)
            cm_list[idx] = sublist
            
        elif idx == sz1+2:
            sublist[-2]  = 'OA:'
            sublist[-1]  = round(100*OA,1)
            cm_list[idx] = sublist
            
        elif idx == sz1+3:
            cm_list[idx] = sublist
            sublist[-2]  = 'F1-macro:'
            sublist[-1]  = round(100*f1_macro,1)    
        idx +=1
    
    conf_array = np.array(cm_list)
    if percentage:
        temp = conf_array[:-4,:].astype(float)
        conf_array[:-4,:-2] = (100*temp[:,:-2]/temp[:,-2].reshape(-1,1)).round(1).astype(str)
        
    
    # Convert to data frame
    df = pd.DataFrame(conf_array)
    df.columns = first_row
    df.index = first_col
    
    return df

def print_confusion_matrix_ML(confusion_matrix, class_label, ind_names, col_names):

    df_cm = pd.DataFrame(confusion_matrix, index=ind_names, columns=col_names)
    
    df_cm.index.name = class_label
    return df_cm