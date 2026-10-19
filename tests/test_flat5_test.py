import json
import numpy as np
from PIL import Image
from utils.flat5_test import load_test_folder, calculate_dice, calculate_map, calculate_pcc


def test_folder_includes_empty_gt_and_fixed_four_crops(tmp_path):
    nested = tmp_path/'nested'; nested.mkdir()
    image = np.zeros((1024,1024,3), dtype=np.uint8)
    for i,(y,x) in enumerate([(0,0),(0,512),(512,0),(512,512)]):
        image[y:y+512,x:x+512] = 30*(i+1)
    Image.fromarray(image).save(nested/'example.png')
    (nested/'example.json').write_text(json.dumps({'cells': []}))
    dataset = load_test_folder(tmp_path)
    assert len(dataset)==4
    for i in range(4):
        a, labels, _, _ = dataset[i]
        b, _, _, _ = dataset[i]
        assert a.shape==(3,512,512) and not len(labels)
        assert np.all(a.numpy()==30*(i+1)) and np.array_equal(a,b)


def test_metrics_keep_empty_gt_and_other():
    box=np.array([[0,0,2,2]])
    def record(name,gt,pred,score):
        return dict(patch=name,gt_boxes=box if gt else np.empty((0,4)),gt_labels=np.array([4] if gt else [],dtype=int),pred_boxes=box if pred else np.empty((0,4)),pred_labels=np.array([4] if pred else [],dtype=int),pred_scores=np.array([score] if pred else []))
    records=[record('empty',False,True,.99),record('other',True,True,.9)]
    dice,table=calculate_dice(records)
    assert len(table)==2 and np.isclose(dice,2/3)
    score,ap=calculate_map(records)
    assert score==.5 and ap.loc[ap.class_id==4,'fp50'].iloc[0]==1
    stats,cps=calculate_pcc(records)
    assert len(cps)==2 and stats['n_pairs']==2 and stats['n_valid_pairs']==0
