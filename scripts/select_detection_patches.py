"""Export a performance-selected example set; not an unbiased test benchmark."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', default='Precise_PDL1_ST_detection_test.ipynb')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--min-dice', type=float, default=.90)
    parser.add_argument('--min-map', type=float, default=.85)
    parser.add_argument('--seed', type=int, default=242)
    parser.add_argument('--model', default='baseline')
    parser.add_argument('--prediction-dir', help='Reuse predictions produced by this notebook on the same data/model')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    nb = json.loads(Path(args.notebook).read_text())
    ctx = {}
    # Load settings and definitions; never execute the notebook evaluation cells.
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('dataset = load_test_folder'):
            break
        exec(compile(source, args.notebook, 'exec'), ctx)
    ctx['torch'].set_num_threads(2)
    dataset = ctx['load_test_folder'](ctx['TEST_FOLDER'])
    checkpoint = ctx['CHECKPOINTS'][args.model]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    settings = dict(selection='performance-selected examples, not unbiased evaluation',
        min_dice=args.min_dice, dice_definition='class-agnostic rectangle union, per 512x512 patch',
        min_map50=args.min_map, map_definition='per-patch macro AP50; absent-GT classes omitted',
        requested_count=args.count, seed=args.seed, model=args.model,
        checkpoint=str(checkpoint.resolve()), checkpoint_size=checkpoint.stat().st_size,
        checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
        data_root=str(ctx['TEST_FOLDER'].resolve()), confidence=ctx['CONFIDENCE'],
        ap_confidence=ctx['AP_CONFIDENCE'], nms_iou=ctx['NMS_IOU'],
        candidate_images=len(dataset.records), candidate_patches=len(dataset),
        class_names=ctx['CLASS_NAMES'], label_aliases=ctx['LABEL_ALIASES'])
    (out/'selection_settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    print(f'Evaluating {len(dataset.records)} images / {len(dataset)} patches', flush=True)
    predictions = (ctx['PredictionStore'](args.prediction_dir) if args.prediction_dir else
        ctx['predict_patches'](dataset, checkpoint, ctx['DEVICE'], ctx['AP_CONFIDENCE'],
            ctx['NMS_IOU'], output_dir=out/'candidate_predictions'))
    if len(predictions) != len(dataset):
        raise ValueError('Prediction count does not match dataset')
    rows = []
    for index, record in enumerate(predictions):
        expected = f'{dataset.records[index//4][0]}::q{index%4}'
        if record['patch'] != expected:
            raise ValueError('Prediction order/source differs from current dataset')
        dice, _, _ = ctx['calculate_dice']([record], ctx['CONFIDENCE'])
        map50, _ = ctx['calculate_map']([record])
        rows.append(dict(index=index, patch=record['patch'],
            source_image=record['source_image'], quadrant=record['quadrant'],
            gt_count=len(record['gt_labels']), dice=dice, map50=map50,
            eligible=bool(dice >= args.min_dice and map50 >= args.min_map)))
    scores = pd.DataFrame(rows)
    eligible = scores[scores.eligible]
    chosen = eligible.sample(n=min(args.count, len(eligible)), random_state=args.seed).sort_values('index')
    scores['selected'] = scores['index'].isin(chosen['index'])
    scores.to_csv(out/'candidate_scores.csv', index=False)
    chosen.to_csv(out/'selected_patches.csv', index=False)
    print(f'Eligible: {len(eligible)} / {len(scores)}; exporting {len(chosen)}', flush=True)
    export = out/'patches'; export.mkdir()
    selected_predictions = ctx['PredictionStore'](out/'selected_predictions', create=True)
    chosen_indices = set(chosen['index'])
    manifest = []
    for index, record in enumerate(predictions):
        if index not in chosen_indices:
            continue
        image, _, _, _ = dataset[index]
        stem = f'patch_{len(manifest)+1:03d}'
        Image.fromarray(image.permute(1, 2, 0).numpy()).save(export/f'{stem}.png')
        cells = []
        for box, label in zip(record['gt_boxes'], record['gt_labels']):
            x0,y0,x1,y1 = map(float,box)
            cells.append(dict(class_name=ctx['CLASS_NAMES'][int(label)], class_id=int(label),
                local_coordinates=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]))
        (export/f'{stem}.json').write_text(json.dumps(dict(cells=cells), ensure_ascii=False))
        selected_predictions.append(record)
        manifest.append(dict(file=f'patches/{stem}.png', label=f'patches/{stem}.json', **rows[index]))
    pd.DataFrame(manifest).to_csv(out/'export_manifest.csv', index=False)
    settings.update(eligible_count=len(eligible), exported_count=len(chosen),
                    target_met=len(chosen)==args.count)
    (out/'selection_settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    if len(chosen):
        dice, dice_table, class_dice = ctx['calculate_dice'](selected_predictions, ctx['CONFIDENCE'])
        map50, ap = ctx['calculate_map'](selected_predictions)
        pcc, cps = ctx['calculate_pcc'](selected_predictions, ctx['CONFIDENCE'])
        pd.DataFrame([dict(patches=len(chosen), dice_union=dice, dice_class_mean=class_dice.dice.mean(),
            map50=map50, pcc_cps_grade=pcc['pcc'], pcc_valid_pairs=pcc['n_valid_pairs'],
            selection_bias='selected by per-patch model performance')]).to_csv(out/'selected_summary.csv',index=False)
        class_dice.merge(ap,on=['class_id','class_name']).to_csv(out/'selected_metrics_per_class.csv',index=False)
        cps.to_csv(out/'selected_cps_per_patch.csv',index=False)
    (out/'USAGE.txt').write_text('''These are performance-selected examples, not an independent test benchmark.
Each image is already a 512x512 crop. Do NOT run the default 1024 resize + four-crop
loader on this folder: that would change the evaluation unit and the metrics.
Use selected_predictions with the metric functions to reproduce selected_summary.csv.
export_manifest.csv records each original source and quadrant. Ground truth labels
are copied from the evaluated crop, not modified to increase scores.
The source folder may contain training images; no held-out claim is made.
''')
    print('OUTPUT',out.resolve(), flush=True)


if __name__ == '__main__':
    main()
