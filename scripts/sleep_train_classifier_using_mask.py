import numpy as np
import pandas as pd
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier
import argparse


WAKE_STATE   = 0
SLEEP_STATE  = 1
RANDOM_STATE = 42
N_ESTIMATORS = 100
GRID_ROWS    = 4
GRID_COLS    = 6
SCORES_FILENAME = "scores.txt"


def compute_zone_features(epoch_array, grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
    H, W = epoch_array.shape
    zone_h = H // grid_rows
    zone_w = W // grid_cols
    features = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            zone = epoch_array[
                r * zone_h:(r + 1) * zone_h,
                c * zone_w:(c + 1) * zone_w
            ]
            features.append(float(zone.mean()))
    return np.array(features)


def get_feature_names():
    return [f"zone_r{r}_c{c}" for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def load_scores(npy_dir):
    scores_path = os.path.join(npy_dir, SCORES_FILENAME)
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"No scores.txt found in {npy_dir}")
    scores = {}
    with open(scores_path, 'r') as f:
        for epoch_idx, line in enumerate(f):
            line = line.strip()
            if line:
                scores[epoch_idx] = int(line)
    return scores


def load_epochs_from_dir(npy_dir, video_id):
    scores  = load_scores(npy_dir)
    pattern = re.compile(rf"video_{video_id}_epoch_(\d+)\.npy")
    entries = []
    for fname in sorted(os.listdir(npy_dir)):
        match = pattern.match(fname)
        if not match:
            continue
        epoch_idx = int(match.group(1))
        if epoch_idx not in scores:
            print(f"[Warning] No score for epoch {epoch_idx} in {npy_dir}, skipping.")
            continue
        arr      = np.load(os.path.join(npy_dir, fname))
        features = compute_zone_features(arr)
        stage    = scores[epoch_idx]
        entries.append((epoch_idx, features, stage))
    return entries


def load_dataset(args):
    feature_names = get_feature_names()
    rows = []

    epoch_dirs = sorted([
        d for d in os.listdir(args.npy_root)
        if os.path.isdir(os.path.join(args.npy_root, d))
        and re.match(r'video_\d+_.+_epochs', d)
    ])

    if not epoch_dirs:
        raise RuntimeError(f"No video_*_epochs directories found in {args.npy_root}")

    for epoch_dir in epoch_dirs:
        video_id = int(re.match(r'video_(\d+)_', epoch_dir).group(1))
        npy_dir  = os.path.join(args.npy_root, epoch_dir)
        print(f"[Video {video_id}] Loading from {epoch_dir}")
        try:
            entries = load_epochs_from_dir(npy_dir, video_id)
        except FileNotFoundError as e:
            print(f"[Warning] {e} — skipping.")
            continue
        for epoch_idx, feature_vec, stage in entries:
            row = {
                'video_id':        video_id,
                'epoch_id':        epoch_idx,
                'unique_epoch_id': f"{video_id}_{epoch_idx:04d}",
                'stage':           stage,
            }
            for fname, fval in zip(feature_names, feature_vec):
                row[fname] = fval
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nLoaded {len(df)} labeled epochs across {df['video_id'].nunique()} videos.")
    print(f"Class distribution:\n{df['stage'].value_counts().sort_index()}\n")
    return df


def get_balanced_data(df, feature_names):
    counts = df['stage'].value_counts()
    n      = int(counts.min())
    wake   = df[df['stage'] == WAKE_STATE].sample(n=n, random_state=RANDOM_STATE)
    sleep  = df[df['stage'] == SLEEP_STATE].sample(n=n, random_state=RANDOM_STATE)
    return pd.concat([wake, sleep]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_classifier(classifier):
    if classifier == 'rf':
        return RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    elif classifier == 'xgb':
        return XGBClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1,
                             eval_metric='logloss')


def get_transition_matrix(df_train):
    """
    Compute 2x2 transition probability matrix from label sequence.
    Returns (2, 2) numpy array where T[i, j] = P(state j | state i).
    """
    stages = df_train.sort_values(['video_id', 'epoch_id'])['stage'].values

    T = np.zeros((2, 2))
    for i in range(len(stages) - 1):
        T[stages[i], stages[i + 1]] += 1

    # Normalize rows, handle zero rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums

    print(f"\nTransition matrix (Wake=0, Sleep=1):")
    print(f"  W→W: {T[0,0]:.3f}  W→S: {T[0,1]:.3f}")
    print(f"  S→W: {T[1,0]:.3f}  S→S: {T[1,1]:.3f}\n")

    return T


def get_emission_matrix(Y_true, Y_pred, n_states=2):
    """
    Compute emission matrix from training predictions.
    E[i, j] = P(predicted j | true state i).
    """
    E = np.zeros((n_states, n_states))
    for true, pred in zip(Y_true, Y_pred):
        E[true, pred] += 1

    row_sums = E.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    E = E / row_sums

    print(f"Emission matrix (rows=true, cols=predicted):")
    print(f"  P(pred W | true W): {E[0,0]:.3f}  P(pred S | true W): {E[0,1]:.3f}")
    print(f"  P(pred W | true S): {E[1,0]:.3f}  P(pred S | true S): {E[1,1]:.3f}\n")

    return E


def apply_hmm(Y_pred_test, Y_pred_train, Y_train, T, E):
    """
    Use Viterbi algorithm to smooth classifier predictions.

    T: (2,2) transition matrix
    E: (2,2) emission matrix
    Y_pred_test: raw classifier predictions on test set

    Returns smoothed predictions.
    """
    n_states = 2

    # Initial state distribution from training labels
    wake_frac  = np.mean(Y_train == WAKE_STATE)
    sleep_frac = 1.0 - wake_frac
    pi = np.array([wake_frac, sleep_frac])

    observations = Y_pred_test
    n_obs = len(observations)

    # Viterbi
    viterbi  = np.zeros((n_obs, n_states))
    backptr  = np.zeros((n_obs, n_states), dtype=int)

    # Initialise
    viterbi[0] = np.log(pi + 1e-10) + np.log(E[:, observations[0]] + 1e-10)

    # Recurse
    for t in range(1, n_obs):
        for s in range(n_states):
            trans_prob = viterbi[t - 1] + np.log(T[:, s] + 1e-10)
            backptr[t, s] = np.argmax(trans_prob)
            viterbi[t, s] = np.max(trans_prob) + np.log(E[s, observations[t]] + 1e-10)

    # Backtrack
    smoothed = np.zeros(n_obs, dtype=int)
    smoothed[-1] = np.argmax(viterbi[-1])
    for t in range(n_obs - 2, -1, -1):
        smoothed[t] = backptr[t + 1, smoothed[t + 1]]

    return smoothed


def run_leave_one_out_cv(df, feature_names, imputer, args):
    logo   = LeaveOneGroupOut()
    X      = df[feature_names].values
    y      = df['stage'].values
    groups = df['video_id'].values

    print(f"Running Leave-One-Group-Out CV over {len(set(groups))} videos\n")

    results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_id  = groups[test_idx][0]
        df_train = df.iloc[train_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        print(f"{'='*50}")
        print(f"Fold {fold} | Test video: video_id={test_id}")

        # Impute on full train fold
        X_train_full = imputer.fit_transform(df_train[feature_names].values)
        X_test       = imputer.transform(df_test[feature_names].values)
        Y_test       = df_test['stage'].values

        # Update df_train with imputed values then balance
        df_train[feature_names] = X_train_full
        df_train_full = df_train.copy()  # full unbalanced, imputed
        df_train      = get_balanced_data(df_train, feature_names)
        X_train       = df_train[feature_names].values
        Y_train       = df_train['stage'].values

        print(f"Train epochs: {len(df_train)} (balanced) / {len(df_train_full)} (full)  "
              f"|  Test epochs: {len(df_test)}")
        dist = {int(k): int(v) for k, v in pd.Series(Y_test).value_counts().sort_index().items()}
        print(f"Test class distribution: {dist}\n")

        # Train classifier
        clf    = build_classifier(args.classifier)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        base_acc = metrics.accuracy_score(Y_test, Y_pred)
        print("=== Base classifier ===")
        print(classification_report(Y_test, Y_pred, target_names=['Wake', 'Sleep']))
        print(f"Accuracy: {base_acc:.3f}\n")

        if args.add_hmm:
            # Build T from full unbalanced train label sequence
            T = get_transition_matrix(df_train_full)

            # Build E from full unbalanced train predictions
            Y_pred_train_full = clf.predict(X_train_full)
            Y_train_full      = df_train_full['stage'].values
            E = get_emission_matrix(Y_train_full, Y_pred_train_full)

            # Viterbi smoothing on test predictions
            Y_pred_smoothed = apply_hmm(Y_pred, Y_pred_train_full, Y_train_full, T, E)

            hmm_acc = metrics.accuracy_score(Y_test, Y_pred_smoothed)
            print(f"=== After HMM (accuracy {base_acc:.3f} → {hmm_acc:.3f}) ===")
            print(classification_report(Y_test, Y_pred_smoothed, target_names=['Wake', 'Sleep']))
            print(f"Accuracy: {hmm_acc:.3f}\n")

            Y_pred_final = Y_pred_smoothed
            final_acc    = hmm_acc
        else:
            Y_pred_final = Y_pred
            final_acc    = base_acc

        results.append({
            'test_video_id':  test_id,
            'base_accuracy':  base_acc,
            'final_accuracy': final_acc,
            'hmm_applied':    args.add_hmm,
            'n_train':        len(df_train),
            'n_test':         len(df_test),
        })

        if args.save_predictions:
            pd.DataFrame({
                'unique_epoch_id': df_test['unique_epoch_id'].values,
                'label':           Y_test,
                'prediction':      Y_pred_final,
            }).to_csv(f"predictions_test_video_{test_id}.csv", index=False)

    results_df = pd.DataFrame(results)
    print(f"{'='*50}")
    print("=== Leave One Out Cross Validation Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nMean final accuracy:  {results_df['final_accuracy'].mean():.3f} "
          f"± {results_df['final_accuracy'].std():.3f}")
    if args.add_hmm:
        print(f"Mean base accuracy:   {results_df['base_accuracy'].mean():.3f} "
              f"± {results_df['base_accuracy'].std():.3f}")
        delta = results_df['final_accuracy'].mean() - results_df['base_accuracy'].mean()
        print(f"Mean HMM improvement: {delta:+.3f}")

    return results_df


def train(args):
    df            = load_dataset(args)
    feature_names = get_feature_names()
    imputer       = SimpleImputer(strategy='constant', fill_value=0)
    run_leave_one_out_cv(df, feature_names, imputer, args)


def main():
    parser = argparse.ArgumentParser(
        description='Sleep/wake classifier using per-epoch spatial movement arrays',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--npy_root',         required=True,
                        help='Root dir containing video_*_epochs/ folders')
    parser.add_argument('--classifier',       required=True, choices=['rf', 'xgb'],
                        help='Classifier to use')
    parser.add_argument('--add_hmm',          action='store_true',
                        help='Apply Viterbi HMM smoothing after classification')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save per-epoch predictions for each fold')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()