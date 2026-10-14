"""
This script compares the lesion tracking performance of the Siamese model when it is run on features extracted from manual
lesion segmentations vs features extracted from automatic lesion segmentations.
It highlights the subjects where the tracking performance is good with automatic segmentations but bad with manual
segmentations, so that they can be inspected visually.

To help understand the errors, each GT lesion group (as defined in generate_csv_for_ML.py) is given a type based on the
number of lesions it contains at each timepoint:
    - one-to-one: 1 lesion at timepoint 1 and 1 lesion at timepoint 2
    - split: 1 lesion at timepoint 1 and several lesions at timepoint 2
    - merge: several lesions at timepoint 1 and 1 lesion at timepoint 2
    - complex: several lesions at both timepoints
    - new: no lesion at timepoint 1
    - disappeared: no lesion at timepoint 2

Inputs:
    -m / --manual: Path to the lesion_mapping_siamese_predictions.csv file obtained with manual lesion segmentations
    -a / --auto: Path to the lesion_mapping_siamese_predictions.csv file obtained with automatic lesion segmentations
    -o / --output-folder: Path to the output folder
    --split: Subjects to consider: test, train or all (default: test)
    --min-f1-auto: Minimum F1 score with automatic segmentations for a subject to be flagged (default: 0.5)
    --min-f1-drop: Minimum F1 drop (auto - manual) for a subject to be flagged (default: 0.2)

Output:
    - per_subject_comparison.csv: per-subject metrics for both segmentations, sorted by F1 drop
    - flagged_subjects_errors.csv: every FP and FN pair of the flagged subjects, for both segmentations
    - error_breakdown.csv: number of FP and FN per GT group type, for both segmentations
    - compare_siamese_manual_vs_auto.log

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import pandas as pd
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual', type=str, required=True, help='Path to the predictions csv obtained with manual lesion segmentations')
    parser.add_argument('-a', '--auto', type=str, required=True, help='Path to the predictions csv obtained with automatic lesion segmentations')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train', 'all'], help='Subjects to consider (default: test)')
    parser.add_argument('--min-f1-auto', type=float, default=0.5, help='Minimum F1 with automatic segmentations for a subject to be flagged')
    parser.add_argument('--min-f1-drop', type=float, default=0.2, help='Minimum F1 drop (auto - manual) for a subject to be flagged')
    return parser.parse_args()


def compute_group_types(df_subject):
    """
    Computes the type of each GT lesion group of a subject from its lesion pairs.
    Since every lesion at timepoint 1 is paired with every lesion at timepoint 2, all lesions appear in the pairs.

    Inputs:
        df_subject: predictions of the lesion pairs of one subject
    Outputs:
        group_types: dictionary {group: type}
    """
    lesions_1 = df_subject[['lesion_id_1', 'group_1']].drop_duplicates()
    lesions_2 = df_subject[['lesion_id_2', 'group_2']].drop_duplicates()
    n_lesions_1 = lesions_1['group_1'].value_counts().to_dict()
    n_lesions_2 = lesions_2['group_2'].value_counts().to_dict()

    group_types = {}
    for group in set(n_lesions_1) | set(n_lesions_2):
        n1 = n_lesions_1.get(group, 0)
        n2 = n_lesions_2.get(group, 0)
        if n1 == 0:
            group_types[group] = 'new'
        elif n2 == 0:
            group_types[group] = 'disappeared'
        elif n1 == 1 and n2 == 1:
            group_types[group] = 'one-to-one'
        elif n1 == 1:
            group_types[group] = 'split'
        elif n2 == 1:
            group_types[group] = 'merge'
        else:
            group_types[group] = 'complex'
    return group_types


def analyze_predictions(df, source):
    """
    Computes per-subject metrics and annotates each lesion pair with the type of its GT groups.

    Inputs:
        df: predictions dataframe
        source: 'manual' or 'auto'
    Outputs:
        df_subjects: per-subject metrics
        df: predictions dataframe with columns group_type_1 and group_type_2
    """
    subject_results = []
    annotated = []
    for subject, df_subject in df.groupby('subject', sort=False):
        df_subject = df_subject.copy()
        group_types = compute_group_types(df_subject)
        df_subject['group_type_1'] = df_subject['group_1'].map(group_types)
        df_subject['group_type_2'] = df_subject['group_2'].map(group_types)
        annotated.append(df_subject)

        tp = int((df_subject['outcome'] == 'TP').sum())
        fp = int((df_subject['outcome'] == 'FP').sum())
        fn = int((df_subject['outcome'] == 'FN').sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # A subject with no lesion correspondence to find and none predicted is perfectly tracked
        if tp + fp + fn == 0:
            precision, recall, f1 = 1.0, 1.0, 1.0

        result = {
            'subject': subject,
            'split': df_subject['split'].iloc[0],
            'n_lesions_1': df_subject['lesion_id_1'].nunique(),
            'n_lesions_2': df_subject['lesion_id_2'].nunique(),
            'n_pairs': len(df_subject),
            'n_positive_pairs': int(df_subject['label'].sum()),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
        for group_type in ['one-to-one', 'split', 'merge', 'complex', 'new', 'disappeared']:
            result[f'n_{group_type}'] = sum(1 for t in group_types.values() if t == group_type)
        subject_results.append(result)

    df_subjects = pd.DataFrame(subject_results)
    # Suffix all columns (except subject) with the source
    df_subjects = df_subjects.rename(columns={col: f'{col}_{source}' for col in df_subjects.columns if col != 'subject'})
    df = pd.concat(annotated, ignore_index=True)
    df.insert(0, 'source', source)
    return df_subjects, df


def error_breakdown(df):
    """
    Counts the FN per GT group type and the FP per pair of GT group types.
    A FN always involves a single group (group_1 == group_2), while a FP involves two different groups.
    """
    breakdown = []
    for source, df_source in df.groupby('source', sort=False):
        df_fn = df_source[df_source['outcome'] == 'FN']
        for group_type, count in df_fn['group_type_1'].value_counts().items():
            breakdown.append({'source': source, 'outcome': 'FN', 'group_types': group_type, 'count': int(count)})
        df_fp = df_source[df_source['outcome'] == 'FP']
        fp_types = df_fp['group_type_1'] + ' / ' + df_fp['group_type_2']
        for group_types, count in fp_types.value_counts().items():
            breakdown.append({'source': source, 'outcome': 'FP', 'group_types': group_types, 'count': int(count)})
    df_breakdown = pd.DataFrame(breakdown, columns=['source', 'outcome', 'group_types', 'count'])
    return df_breakdown.sort_values(['outcome', 'group_types', 'source']).reset_index(drop=True)


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'compare_siamese_manual_vs_auto.log'))

    # Load the predictions
    df_manual = pd.read_csv(args.manual)
    df_auto = pd.read_csv(args.auto)
    for name, df in [('manual', df_manual), ('auto', df_auto)]:
        missing = {'lesion_id_1', 'lesion_id_2'} - set(df.columns)
        if missing:
            raise ValueError(f"Columns {missing} are missing from the {name} predictions: the input csv of track_lesion_siamese.py must contain a lesion_id column")

    # Keep the requested subjects
    if args.split != 'all':
        df_manual = df_manual[df_manual['split'] == args.split]
        df_auto = df_auto[df_auto['split'] == args.split]

    # Per-subject metrics
    df_subjects_manual, df_manual = analyze_predictions(df_manual, 'manual')
    df_subjects_auto, df_auto = analyze_predictions(df_auto, 'auto')

    # Compare manual vs auto per subject
    only_manual = set(df_subjects_manual['subject']) - set(df_subjects_auto['subject'])
    only_auto = set(df_subjects_auto['subject']) - set(df_subjects_manual['subject'])
    if only_manual:
        logger.warning(f"Subjects only in manual predictions (ignored): {sorted(only_manual)}")
    if only_auto:
        logger.warning(f"Subjects only in auto predictions (ignored): {sorted(only_auto)}")
    df_comparison = df_subjects_manual.merge(df_subjects_auto, on='subject', how='inner')
    df_comparison['f1_drop'] = (df_comparison['f1_auto'] - df_comparison['f1_manual']).round(4)
    df_comparison['flagged'] = (df_comparison['f1_auto'] >= args.min_f1_auto) & (df_comparison['f1_drop'] >= args.min_f1_drop)
    df_comparison = df_comparison.sort_values('f1_drop', ascending=False).reset_index(drop=True)
    comparison_path = os.path.join(args.output_folder, 'per_subject_comparison.csv')
    df_comparison.to_csv(comparison_path, index=False)
    logger.info(f"Per-subject comparison saved to {comparison_path}")

    # Summary
    logger.info(f"Subjects compared ({args.split}): {len(df_comparison)}")
    for source in ['manual', 'auto']:
        tp = df_comparison[f'TP_{source}'].sum()
        fp = df_comparison[f'FP_{source}'].sum()
        fn = df_comparison[f'FN_{source}'].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        logger.info(f"{source:>6} - TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                    f"mean per-subject F1: {df_comparison[f'f1_{source}'].mean():.4f}")

    # Flagged subjects
    df_flagged = df_comparison[df_comparison['flagged']]
    logger.info(f"Subjects with F1 auto >= {args.min_f1_auto} and F1 drop >= {args.min_f1_drop}: {len(df_flagged)}")
    display_cols = ['subject', 'f1_auto', 'f1_manual', 'f1_drop',
                    'n_lesions_1_auto', 'n_lesions_1_manual', 'n_lesions_2_auto', 'n_lesions_2_manual',
                    'FP_auto', 'FP_manual', 'FN_auto', 'FN_manual',
                    'n_split_auto', 'n_split_manual', 'n_merge_auto', 'n_merge_manual', 'n_complex_auto', 'n_complex_manual']
    if not df_flagged.empty:
        logger.info("\n" + df_flagged[display_cols].to_string(index=False))

    # Errors (FP and FN pairs) of the flagged subjects, for both segmentations
    df_all = pd.concat([df_manual, df_auto], ignore_index=True)
    df_errors = df_all[df_all['subject'].isin(df_flagged['subject']) & df_all['outcome'].isin(['FP', 'FN'])]
    df_errors = df_errors.sort_values(['subject', 'source', 'outcome', 'pred_prob']).reset_index(drop=True)
    errors_path = os.path.join(args.output_folder, 'flagged_subjects_errors.csv')
    df_errors.to_csv(errors_path, index=False)
    logger.info(f"Errors of flagged subjects saved to {errors_path}")

    # Error breakdown per GT group type (all compared subjects)
    df_breakdown = error_breakdown(df_all[df_all['subject'].isin(df_comparison['subject'])])
    breakdown_path = os.path.join(args.output_folder, 'error_breakdown.csv')
    df_breakdown.to_csv(breakdown_path, index=False)
    logger.info("Error breakdown per GT group type (FN: type of the group, FP: types of both groups):\n" + df_breakdown.to_string(index=False))
    logger.info(f"Error breakdown saved to {breakdown_path}")

    return None


if __name__ == "__main__":
    main()
