import numpy as np
from utils.rectangle_metrics import rectangle_dice, rectangle_mask, average_precision_50, count_correlation


def record(gt_boxes, pred_boxes, scores, gt_labels=None, pred_labels=None):
    return {'gt_boxes': np.array(gt_boxes).reshape(-1,4), 'pred_boxes':np.array(pred_boxes).reshape(-1,4),
            'gt_labels':np.array(gt_labels if gt_labels is not None else [0]*len(gt_boxes)),
            'pred_labels':np.array(pred_labels if pred_labels is not None else [0]*len(pred_boxes)),
            'pred_scores':np.array(scores)}


def test_dice_union_clipping_empty_and_partial_overlap():
    box=[[0,0,2,2]]
    assert rectangle_dice(box,box*2,4,4)['dice']==1
    assert rectangle_dice(box,[[1,0,3,2]],4,4)['dice']==.5
    assert rectangle_dice(box,[],4,4)['dice']==0
    assert np.isnan(rectangle_dice([],[],4,4)['dice'])
    assert rectangle_mask([[-2,-2,2,2],[3,3,3,4]],4,4).sum()==4


def test_ap_confidence_order_duplicates_and_absent_classes():
    box=[0,0,2,2]; fp=[5,5,7,7]
    rows,mean=average_precision_50([record([box],[box,box],[.9,.8])],[0,1])
    assert mean==1 and rows[0]['tp50']==1 and rows[0]['fp50']==1
    assert np.isnan(rows[1]['ap50'])
    rows,mean=average_precision_50([record([box],[fp,box],[.9,.8])],[0])
    assert mean==.5
    _,mean=average_precision_50([record([box],[],[])],[0])
    assert mean==0
    _,mean=average_precision_50([record([box],[box],[.9],pred_labels=[1])],[0,1])
    assert mean==0
    # An empty-GT image's high-score false positive must lower AP.
    _,mean=average_precision_50([record([], [box],[.95]),record([box],[box],[.9])],[0])
    assert mean==.5


def test_ap_iou_threshold_and_image_isolation():
    box=[0,0,3,2]
    _,mean=average_precision_50([record([box],[[1,0,4,2]],[.9])],[0])
    assert mean==1  # IoU exactly 0.5
    _,mean=average_precision_50([record([box],[],[]),record([], [box],[.9])],[0])
    assert mean==0


def test_pcc_constant_single_and_systematic_bias():
    assert np.isnan(count_correlation([1],[1])['pcc'])
    assert np.isnan(count_correlation([1,1],[1,2])['pcc'])
    result=count_correlation([1,2,3],[2,4,6])
    assert abs(result['pcc']-1)<1e-10 and result['count_bias']==2


def test_cps_formula_cap_zero_denominator_and_grade_boundaries():
    from utils.rectangle_metrics import cps_from_flat5_counts, cps_grade, paired_score_correlation
    assert cps_from_flat5_counts([90, 999, 10, 5, 999])['cps'] == 15
    capped = cps_from_flat5_counts([0, 0, 2, 4, 0])
    assert capped['cps'] == 100 and capped['cps_raw'] == 300
    assert np.isnan(cps_from_flat5_counts([0, 1, 0, 10, 1])['cps'])
    assert [cps_grade(v, [1, 5, 10]) for v in [0, 1, 4.9, 5, 9.9, 10, 100]] == [0, 1, 1, 2, 2, 3, 3]
    assert np.isnan(cps_grade(float('nan'), [1, 5, 10]))
    result = paired_score_correlation([0, 1, 2], [0, float('nan'), 2])
    assert result['n_valid_pairs'] == 2 and result['n_invalid_pred'] == 1
    assert abs(result['pcc']-1) < 1e-10
