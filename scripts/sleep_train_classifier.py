import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier
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


def run_leave_one_out_cv(df, features, imputer, args):
    logo = LeaveOneGroupOut()

    X = df[features].values
    y = df['stage'].values
    groups = df['video_id'].values

    print(f"Running Leave-One-Group-Out CV over {len(set(groups))} mice\n")

    results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_id = groups[test_idx][0]

        print(f"{'='*50}")
        print(f"Fold {fold} | Test mouse: video_id={test_id}")

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = y[train_idx], y[test_idx]

        df_train = df.iloc[train_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        df_train[features] = X_train
        df_train = get_balanced_data(df_train)

        X_train = df_train[features].values
        Y_train = df_train['stage'].values

        print(f"Train epochs: {len(df_train)}  |  Test epochs: {len(df_test)}")
        print(f"Test class distribution: {dict(pd.Series(Y_test).value_counts().sort_index())}\n")

        clf = build_classifier(args.classifier)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        acc = metrics.accuracy_score(Y_test, Y_pred)

        print(classification_report(Y_test, Y_pred, target_names=['Wake', 'Sleep']))
        print(f"Accuracy: {acc:.3f}\n")

        results.append({
            'test_video_id': test_id,
            'accuracy': acc,
            'n_train': len(df_train),
            'n_test': len(df_test),
        })

        if args.save_predictions:
            out_fname = f"predictions_test_video_{test_id}.csv"
            pd.DataFrame({
                'unique_epoch_id': df_test['unique_epoch_id'].values,
                'label': Y_test,
                'prediction': Y_pred,
            }).to_csv(out_fname, index=False)

    results_df = pd.DataFrame(results)

    print(f"{'='*50}")
    print("=== Leave One Out Cross Validation Summary ===")
    print(results_df.to_string(index=False))
    print(f"\nMean accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")

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
    parser.add_argument('--train_dataset',    required=True, help='Path to CSV with features + stage column')
    parser.add_argument('--classifier',       required=True, choices=['rf', 'xgb'], help='Classifier to use')
    parser.add_argument('--save_predictions', action='store_true', help='Save per-epoch predictions for each fold')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
