import json
import tempfile
from pathlib import Path
import numpy as np
from utils.flat5_eval_dataset import Flat5EvaluationDataset, select_fixed_test_records


def test_native_names_are_not_seven_class_remapped():
    names=['Negative Tumor(Epithelial)','Negative Immune','Positive Tumor(Epithelial)','Positive Immune','Other']
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'labels.json'
        cells=[{'class_name':name,'class_id':99,'local_coordinates':[[10,10],[30,30]]} for name in names]
        path.write_text(json.dumps({'cells':cells}))
        labels=Flat5EvaluationDataset.load_labels(path,image_size=(100,100))
        assert labels[:,0].tolist()==[0,1,2,3,4]
        assert np.allclose(labels[:,1:],[[.2,.2,.2,.2]]*5)
        cells[0]['class_name']='unknown'
        path.write_text(json.dumps({'cells':cells}))
        try:Flat5EvaluationDataset.load_labels(path,image_size=(100,100))
        except ValueError:pass
        else:raise AssertionError('Unknown classes must not disappear silently')


def test_old_names_merge_into_same_five_classes():
    names=['Negative Epithelial','Negative Lymphocyte','Negative Macrophage',
           'Positive Epithelial','Positive Lymphocyte','Positive Macrophage','other']
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'labels.json'
        path.write_text(json.dumps({'cells':[{'class_name':name,'local_coordinates':[[0,0],[20,20]]} for name in names]}))
        assert Flat5EvaluationDataset.load_labels(path,image_size=(100,100))[:,0].tolist()==[0,1,1,2,3,3,4]


def test_fixed_manifest_excludes_extra_records_and_rejects_missing():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        records=[(root/'new'/'patch_1.jpeg',root/'new'/'patch_1.json'),
                 (root/'old'/'patch_1.jpeg',root/'old'/'patch_1.json')]
        manifest=root/'manifest.json'
        manifest.write_text(json.dumps({'expected_count':1,'image_paths':['new/patch_1.jpeg'],
                                        'excluded_previous_slides':['old']}))
        assert select_fixed_test_records(records,root,manifest)==records[:1]
        try:select_fixed_test_records(records[1:],root,manifest)
        except FileNotFoundError:pass
        else:raise AssertionError('Missing records must fail')
