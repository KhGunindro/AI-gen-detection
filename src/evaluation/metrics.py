import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt


def evaluate_model(model,
                   test_ds,
                   class_names,
                   run_dirs,
                   cfg,
                   logger):
    
    logger.info("🔍 Starting evaluation...")
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    y_true = []
    y_pred = []
    y_prob = []

    logger.info("Generating predictions...")

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0).flatten()
        preds = (probabilities > 0.5).astype(int)

        y_true.extend(labels.numpy().astype(int))
        y_pred.extend(preds)
        y_prob.extend(probabilities)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    logger.info("Saving confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

    cm_path = os.path.join(run_dirs["plots"], "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    logger.info(f"Confusion matrix saved at: {cm_path}")

    logger.info("Saving classification report...")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    report_path = os.path.join(run_dirs["reports"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Classification report saved at: {report_path}")

    logger.info("Saving ROC curve...")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    roc_path = os.path.join(run_dirs["plots"], "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    logger.info(f"ROC curve saved at: {roc_path}")

    logger.info("✅ Evaluation Completed.")

    return {
        "loss": loss,
        "accuracy": accuracy,
        "confusion_matrix": cm_path,
        "classification_report": report_path,
        "roc_curve": roc_path
    }
