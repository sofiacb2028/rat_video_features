import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import classification_report
import argparse
import os

WAKE_STATE  = 0
SLEEP_STATE = 1

NON_FEATURE_COLS = {'unique_epoch_id', 'time_sec', 'stage'}


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def get_balanced_data(df):
    counts = df['stage'].value_counts()
    print("Class counts before balancing:")
    print(counts.sort_index())
    n = int(counts.min())

    wake  = df[df['stage'] == WAKE_STATE].sample(n=n, random_state=42)
    sleep = df[df['stage'] == SLEEP_STATE].sample(n=n, random_state=42)

    balanced = pd.concat([wake, sleep]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nBalanced to {n} epochs per class ({n * 2} total)\n")
    return balanced


def random_forest(X_train, X_test, Y_train, Y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, Y_train)

    Y_pred_test = clf.predict(X_test)

    print("=== Test performance ===")
    print(classification_report(Y_test, Y_pred_test, target_names=['Wake', 'Sleep']))
    print(f"Test accuracy: {metrics.accuracy_score(Y_test, Y_pred_test):.3f}")

    return clf, metrics.accuracy_score(Y_test, Y_pred_test)


def train(args):
    df = pd.read_csv(args.train_dataset)

    features = get_feature_cols(df)
    print(f"Features detected: {len(features)}")
    print(f"Total epochs:      {len(df)}")
    print(f"Class distribution:\n{df['stage'].value_counts().sort_index()}\n")

    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df[features] = imputer.fit_transform(df[features])

    df_balanced = get_balanced_data(df)

    df_train, df_test = train_test_split(
        df_balanced, test_size=0.2, random_state=42,
        stratify=df_balanced['stage']
    )

    X_train = df_train[features].values
    X_test  = df_test[features].values
    Y_train = df_train['stage'].values
    Y_test  = df_test['stage'].values

    print(f"Train size: {len(df_train)}  |  Test size: {len(df_test)}\n")

    if args.classifier == 'rf':
        clf, test_acc = random_forest(X_train, X_test, Y_train, Y_test)

    importances = pd.Series(clf.feature_importances_, index=features)
    print("\n=== Top 15 most important features ===")
    print(importances.sort_values(ascending=False).head(15).to_string())

    if args.predict_dataset is not None:
        df_infer = pd.read_csv(args.predict_dataset)
        df_infer[features] = imputer.transform(df_infer[features])

        Y_infer      = df_infer['stage'].values
        Y_infer_pred = clf.predict(df_infer[features].values)

        print("\n=== Inference dataset performance ===")
        print(classification_report(Y_infer, Y_infer_pred, target_names=['Wake', 'Sleep']))

        out_fname = os.path.splitext(args.predict_dataset)[0] + '_predictions.csv'
        pd.DataFrame({
            'unique_epoch_id': df_infer['unique_epoch_id'].values,
            'label':           Y_infer,
            'prediction':      Y_infer_pred,
        }).to_csv(out_fname, index=False)
        print(f"Predictions saved to: {out_fname}")


def main():
    parser = argparse.ArgumentParser(
        description='Sleep/wake classifier for rat tracking data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--train_dataset',   required=True, help='Path to CSV with features + stage column')
    parser.add_argument('--classifier',      required=True, choices=['rf'], help='Classifier to use')
    parser.add_argument('--predict_dataset', default=None,  help='Optional separate CSV to predict on')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
