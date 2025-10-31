import argparse
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_csv_data, create_csv_submission
from implementations import ridge_regression



# --- metrics ---
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

def preprocess_data(tx, txt, y):
    """
    Full preprocessing pipeline (Salma version)
    """
    print("\n🔧 Starting preprocessing...")

    # Hyperparameters
    MISSING_THRESH = 50.0     
    MAX_FEATS = 200           
    VAR_THRESH = 1e-10        
    CORR_THRESH = 0.5         
    MIN_CORR_Y = 0.3          
    TOPK_YCORR = 100          

    # Split bias and features
    b_tr  = tx[:, :1]
    X_tr  = tx[:, 1:]
    b_te  = txt[:, :1]
    X_te  = txt[:, 1:]

    steps = []
    counts = [X_tr.shape[1]]

    # A) Drop high-missing features 
    miss_pct = np.isnan(X_tr).mean(axis=0) * 100.0
    var_tr0  = np.nanvar(X_tr, axis=0)

    keep = miss_pct < MISSING_THRESH
    idx_keep = np.where(keep)[0]

    if MAX_FEATS is not None:
        order = np.lexsort((-var_tr0[idx_keep], miss_pct[idx_keep]))
        idx_keep = idx_keep[order[:min(MAX_FEATS, len(idx_keep))]]

    keep_mask = np.zeros(X_tr.shape[1], dtype=bool)
    keep_mask[idx_keep] = True

    X_tr = X_tr[:, keep_mask]
    X_te = X_te[:, keep_mask]

    steps.append("A) Missing filter")
    counts.append(X_tr.shape[1])

    # B) Impute missing values 
    col_means = np.nanmean(X_tr, axis=0)
    inds = np.where(np.isnan(X_tr))
    if inds[0].size:
        X_tr[inds] = np.take(col_means, inds[1])
    inds_t = np.where(np.isnan(X_te))
    if inds_t[0].size:
        X_te[inds_t] = np.take(col_means, inds_t[1])

    steps.append("B) Imputation")
    counts.append(X_tr.shape[1])

    # C) Drop near-constant features 
    var_tr = np.var(X_tr, axis=0)
    keep_var = var_tr > VAR_THRESH
    X_tr = X_tr[:, keep_var]
    X_te = X_te[:, keep_var]

    steps.append("C) Variance filter")
    counts.append(X_tr.shape[1])

    # D) Remove highly correlated features 
    Z = (X_tr - X_tr.mean(axis=0)) / (X_tr.std(axis=0) + 1e-12)
    C = np.corrcoef(Z, rowvar=False)
    p = C.shape[0]
    keep_corr = np.ones(p, dtype=bool)
    for i in range(p):
        if keep_corr[i]:
            keep_corr[(i+1):] &= np.abs(C[i, (i+1):]) <= CORR_THRESH

    X_tr = X_tr[:, keep_corr]
    X_te = X_te[:, keep_corr]

    steps.append("D) Corr filter")
    counts.append(X_tr.shape[1])

    # E) Keep only features correlated with y 
    y_centered = y - y.mean()
    stdy = y_centered.std()
    stdx = X_tr.std(axis=0)
    stdx[stdx == 0] = 1.0
    corr_y = np.abs((y_centered @ (X_tr - X_tr.mean(axis=0))) / (len(y) * stdx * stdy))

    if TOPK_YCORR is not None:
        k = min(TOPK_YCORR, X_tr.shape[1])
        idx = np.argsort(-corr_y)[:k]
        keep_y = np.zeros_like(corr_y, dtype=bool)
        keep_y[idx] = True
    else:
        keep_y = corr_y > MIN_CORR_Y

    if not np.any(keep_y):
        keep_y[np.argmax(corr_y)] = True

    X_tr = X_tr[:, keep_y]
    X_te = X_te[:, keep_y]

    steps.append("E) Corr(y,x) filter")
    counts.append(X_tr.shape[1])

    # F) Standardize 
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    # Reattach bias
    tx  = np.c_[b_tr, X_tr]
    txt = np.c_[b_te, X_te]

    # Optional 0/1 labels
    y01 = (y > 0).astype(np.int8)

    print(f" Final feature count: {tx.shape[1]-1}")
    print("tx/txt shapes:", tx.shape, txt.shape)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(['Start'] + steps, counts, marker='o')
    plt.title('Feature Count After Each Preprocessing Step')
    plt.xlabel('Step')
    plt.ylabel('Number of Features')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('feature_reduction_steps.png', dpi=300)
    plt.close()

    return tx, txt, y01


def main(args):
    print("🚀 Starting Ridge Regression pipeline...")

    # 1. Load raw data
    y_train, tx_train, tx_test, train_ids, test_ids = load_csv_data(args.data_path)
    print(f"Loaded data: train={tx_train.shape}, test={tx_test.shape}")
    # --- Ensure arrays are 2D ---
    if tx_train.ndim == 1:
        tx_train = tx_train.reshape(-1, 1)
    if tx_test.ndim == 1:
        tx_test = tx_test.reshape(-1, 1)
    if y_train.ndim > 1:
        y_train = y_train.squeeze()

    print("✅ Shapes before preprocessing:")
    print("x_train:", tx_train.shape)
    print("x_test:", tx_test.shape)
    print("y_train:", y_train.shape)

    # 2. Preprocess
    tx_train, tx_test, y_train = preprocess_data(tx_train, tx_test, y_train)

    # 3. Train Ridge Regression
    print("\n🎯 Training Ridge Regression model...")
    w, loss = ridge_regression(y_train, tx_train, args.lambda_)
    print(f" Training done. Loss = {loss:.4f}")

    # 4. Evaluate
    y_pred_train = np.sign(tx_train.dot(w))
    acc = compute_accuracy(y_train, y_pred_train)
    f1 = compute_f1_score(y_train, y_pred_train)
    print(f" Training Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

    # 5. Predict test
    print("\n Predicting on test set...")
    y_pred_test = np.sign(tx_test.dot(w))

    # 6. Save submission
    print(" Creating CSV submission...")
    create_csv_submission(test_ids, y_pred_test, args.output_path)
    print(f" Submission saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ridge Regression with custom preprocessing")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to data folder")
    parser.add_argument("--lambda_", type=float, default=1e-5, help="Regularization strength (lambda)")
    parser.add_argument("--output_path", type=str, default="ridge_submission.csv", help="Output CSV file")
    parser.add_argument("--threshold", type=float, default=-0.58, help="Decision threshold for predictions")

    args = parser.parse_args()

    main(args)
