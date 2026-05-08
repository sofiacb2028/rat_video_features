import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier
from hmm_filter.hmm_filter import HMMFilter
import argparse


WAKE_STATE  = 0
SLEEP_STATE = 1

NON_FEATURE_COLS = {'unique_epoch_id', 'time_sec', 'stage', 'video_id'}

RANDOM_STATE = 42
N_ESTIMATORS = 100


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def get_balanced_data(df):
    counts = df['stage'].value_counts()
    n = int(counts.min())
    wake  = df[df['stage'] == WAKE_STATE].sample(n=n, random_state=RANDOM_STATE)
    sleep = df[df['stage'] == SLEEP_STATE].sample(n=n, random_state=RANDOM_STATE)
    return pd.concat([wake, sleep]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_classifier(classifier):
    if classifier == 'rf':
        return RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    elif classifier == 'xgb':
        return XGBClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=1, eval_metric='logloss')


def get_t_matrix(df):
    """
    Compute 2x2 transition probability matrix from label sequence.
    Returns dict keyed by (from_state, to_state).
    """
    df_temp = df[['unique_epoch_id', 'video_id', 'stage']].drop_duplicates().copy()
    df_temp['previous_stage'] = df_temp['stage'].shift()
    df_temp = df_temp.reset_index(drop=True)
    df_temp.loc[0, 'previous_stage'] = df_temp.loc[0, 'stage']
    df_temp['previous_stage'] = df_temp['previous_stage'].astype(int)

    w2w = len(df_temp[(df_temp['stage'] == WAKE_STATE)  & (df_temp['previous_stage'] == WAKE_STATE)])
    w2s = len(df_temp[(df_temp['stage'] == SLEEP_STATE) & (df_temp['previous_stage'] == WAKE_STATE)])
    s2w = len(df_temp[(df_temp['stage'] == WAKE_STATE)  & (df_temp['previous_stage'] == SLEEP_STATE)])
    s2s = len(df_temp[(df_temp['stage'] == SLEEP_STATE) & (df_temp['previous_stage'] == SLEEP_STATE)])

    from_w = w2w + w2s
    from_s = s2w + s2s

    t = {
        (WAKE_STATE,  WAKE_STATE):  w2w / from_w,
        (WAKE_STATE,  SLEEP_STATE): w2s / from_w,
        (SLEEP_STATE, WAKE_STATE):  s2w / from_s,
        (SLEEP_STATE, SLEEP_STATE): s2s / from_s,
    }

    print(f"\nTransition matrix (Wake=0, Sleep=1):")
    print(f"  W→W: {t[(0,0)]:.3f}  W→S: {t[(0,1)]:.3f}")
    print(f"  S→W: {t[(1,0)]:.3f}  S→S: {t[(1,1)]:.3f}\n")

    return t


def apply_hmm(clf, t_matrix, X_train_full, Y_train_full,
              X_test, Y_test, df_train_full, df_test, features):
    """
    Fit HMM on full unbalanced training fold, apply to test fold.
    Manually sets transition matrix to bypass hmm_filter.fit() bug
    with binary classes (ValueError: Length of values (1) does not
    match length of index (2)).
    """
    y_pred_train_full = clf.predict(X_train_full)
    train_data        = pd.DataFrame(Y_train_full, columns=['Stage'])
    train_data['predict_stage'] = y_pred_train_full
    train_data['video']         = df_train_full['video_id'].values

    hmmfilter   = HMMFilter()
    hmmfilter.A = t_matrix

    # Manually build emission matrix from training predictions
    # instead of calling hmmfilter.fit(), which has a pandas bug
    # with binary (2-class) groupby in extract_probabs.
    true_labels = train_data['Stage'].values
    pred_labels = train_data['predict_stage'].values

    emission = {}
    for true_state in [WAKE_STATE, SLEEP_STATE]:
        mask        = true_labels == true_state
        preds       = pred_labels[mask]
        n           = len(preds)
        if n == 0:
            # Fallback: assume perfect emission
            emission[(true_state, true_state)]                                   = 1.0
            emission[(true_state, SLEEP_STATE if true_state == WAKE_STATE else WAKE_STATE)] = 0.0
        else:
            for pred_state in [WAKE_STATE, SLEEP_STATE]:
                emission[(true_state, pred_state)] = (preds == pred_state).sum() / n

    hmmfilter.B = emission

    proba_records = pd.DataFrame.from_records(
        clf.predict_proba(X_test), columns=clf.classes_
    ).to_dict(orient='records')

    test_data = pd.DataFrame(Y_test, columns=['Stage'])
    test_data['predict_stage'] = clf.predict(X_test)
    test_data['probabs']       = [{k: v for k, v in r.items() if v > 0} for r in proba_records]
    test_data['index']         = pd.RangeIndex(len(test_data))
    test_data['video']         = df_test['video_id'].values

    df_hmm = hmmfilter.predict(
        test_data,
        session_column='video',
        probabs_column='probabs',
        prediction_column='predict_stage'
    )
    df_hmm = df_hmm.set_index('index').sort_index()

    pre_hmm_acc = metrics.accuracy_score(Y_test, test_data['predict_stage'])
    hmm_acc     = metrics.accuracy_score(Y_test, df_hmm['predict_stage'])

    return pre_hmm_acc, hmm_acc, df_hmm


def run_leave_one_out_cv(df, features, imputer, args):
    logo   = LeaveOneGroupOut()
    X      = df[features].values
    y      = df['stage'].values
    groups = df['video_id'].values

    print(f"Running Leave-One-Group-Out CV over {len(set(groups))} mice\n")

    results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_id  = groups[test_idx][0]
        df_train = df.iloc[train_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        print(f"{'='*50}")
        print(f"Fold {fold} | Test mouse: video_id={test_id}")

        # Impute on full train fold
        X_train_full = imputer.fit_transform(df_train[features].values)
        X_test       = imputer.transform(df_test[features].values)
        Y_train_full = df_train['stage'].values
        Y_test       = df_test['stage'].values

        # Store full imputed train fold for HMM
        df_train_full = df_train.copy()
        df_train_full[features] = X_train_full

        # Balance for classifier
        df_train_balanced = get_balanced_data(df_train_full)
        X_train = df_train_balanced[features].values
        Y_train = df_train_balanced['stage'].values

        print(f"Train epochs: {len(df_train_balanced)} (balanced) / {len(df_train_full)} (full)  "
              f"|  Test epochs: {len(df_test)}")
        dist = {int(k): int(v) for k, v in pd.Series(Y_test).value_counts().sort_index().items()}
        print(f"Test class distribution: {dist}\n")

        clf    = build_classifier(args.classifier)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        base_acc = metrics.accuracy_score(Y_test, Y_pred)
        print("=== Base classifier ===")
        print(classification_report(Y_test, Y_pred, target_names=['Wake', 'Sleep']))
        print(f"Accuracy: {base_acc:.3f}\n")

        if args.add_hmm:
            t_matrix = get_t_matrix(df_train_full)

            pre_hmm_acc, hmm_acc, hmm_df = apply_hmm(
                clf, t_matrix,
                X_train_full, Y_train_full,
                X_test, Y_test,
                df_train_full, df_test, features
            )
            Y_pred_final = hmm_df['predict_stage'].values
            print(f"=== After HMM (accuracy {pre_hmm_acc:.3f} → {hmm_acc:.3f}) ===")
            print(classification_report(Y_test, Y_pred_final, target_names=['Wake', 'Sleep']))
            print(f"Accuracy: {hmm_acc:.3f}\n")
            final_acc = hmm_acc
        else:
            Y_pred_final = Y_pred
            final_acc    = base_acc

        results.append({
            'test_video_id':  test_id,
            'base_accuracy':  base_acc,
            'final_accuracy': final_acc,
            'hmm_applied':    args.add_hmm,
            'n_train':        len(df_train_balanced),
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
    df = pd.read_csv(args.train_dataset)

    features = get_feature_cols(df)
    print(f"Features detected: {len(features)}")
    print(f"Total epochs:      {len(df)}")
    print(f"Class distribution:\n{df['stage'].value_counts().sort_index()}\n")

    imputer = SimpleImputer(strategy='constant', fill_value=0)
    run_leave_one_out_cv(df, features, imputer, args)


def main():
    parser = argparse.ArgumentParser(
        description='Sleep/wake classifier for rat tracking data — leave-one-out CV by mouse',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--train_dataset',    required=True,
                        help='Path to CSV with features + stage column')
    parser.add_argument('--classifier',       required=True, choices=['rf', 'xgb'],
                        help='Classifier to use')
    parser.add_argument('--add_hmm',          action='store_true',
                        help='Apply HMM filter after classification')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save per-epoch predictions for each fold')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()