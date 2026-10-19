import json
import numpy as np
from detection_support.helpers import PredictionStore, save_predictions_json, average_precision_50


def test_disk_predictions_preserve_ap_and_empty_gt(tmp_path):
    records = [
        dict(patch='empty', gt_boxes=np.empty((0, 4)), gt_labels=np.array([], dtype=int),
             pred_boxes=np.array([[0, 0, 2, 2]]), pred_labels=np.array([0]), pred_scores=np.array([.99])),
        dict(patch='positive', gt_boxes=np.array([[0, 0, 2, 2]]), gt_labels=np.array([0]),
             pred_boxes=np.array([[0, 0, 2, 2]]), pred_labels=np.array([0]), pred_scores=np.array([.9])),
    ]
    store = PredictionStore(tmp_path/'predictions', create=True)
    for record in records:
        store.append(record)
    assert len(store) == 2
    for _ in range(2):
        for actual, expected in zip(store, records):
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
    assert average_precision_50(store, [0])[1] == average_precision_50(records, [0])[1] == .5
    reopened = PredictionStore(tmp_path/'predictions')
    assert len(reopened) == 2
    save_predictions_json(reopened, tmp_path/'predictions.json')
    exported = json.loads((tmp_path/'predictions.json').read_text())
    assert len(exported) == 2 and exported[0]['gt_boxes'] == []
