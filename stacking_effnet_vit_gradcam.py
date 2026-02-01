# stacking_effnet_vit_gradcam.py
# ==========================
# This script implements a stacking-based ensemble using EfficientNetV2-S and ViT-B/32 as base learners,
# and a Logistic Regression model as the meta-learner. The ensemble is evaluated on a holdout test set.
# Grad-CAM visualisations are generated for each base model and a strict-fusion map highlighting
# agreement regions is also created.
# ==========================

import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

from scipy.stats import ttest_rel
import numpy as np

import seaborn as sns

import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc   
from sklearn.preprocessing import label_binarize     


from statsmodels.stats.contingency_tables import mcnemar

# ---------------------
# Global Configuration  
# --------------------
train_dir = "data/brain_tumor/train"   # training dataset directory
test_dir  = "data/brain_tumor/test"    # test dataset directory
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available, otherwise CPU
batch_size = 16
epochs = 30                  # maximum training epochs (early stopping used to terminate earlier if needed)
lr_eff, lr_vit = 1e-4, 1e-4  # learning rates for EfficientNet and ViT
weight_decay = 1e-2          # L2 regularisation
holdout_ratio = 0.2          # proportion of training data reserved for meta-learner (stacking holdout)
patience = 3                 # early stopping patience (epochs without improvement)
seed = 42                    # reproducibility seed
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

checkpoint_dir = "checkpoints"  # save directory for best checkpoints
os.makedirs(checkpoint_dir, exist_ok=True)

results_dir = "liam_results_figures"
os.makedirs(results_dir, exist_ok=True)

print(f"[INFO] Using device: {device}")
print(f"[INFO] Max Epochs: {epochs}, Batch size: {batch_size}, Holdout ratio: {holdout_ratio}")

# --------------------
# Data Transforms  
# --------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])

# --------------------
# Dataset Creation & Train/Stack Split 
# --------------------
full_train = datasets.ImageFolder(train_dir, transform=train_tf)
test_ds    = datasets.ImageFolder(test_dir,  transform=test_tf)
num_classes = len(full_train.classes)

n_total = len(full_train)
n_stack = int(n_total * holdout_ratio)
n_base  = n_total - n_stack
base_ds, stack_ds = random_split(full_train, [n_base, n_stack], generator=torch.Generator().manual_seed(seed))

print(f"[INFO] Total training samples: {n_total}")
print(f"[INFO] Base-train samples: {n_base}, Stack-train samples: {n_stack}")
print(f"[INFO] Test samples: {len(test_ds)}")
print(f"[INFO] Classes: {full_train.classes}")

# --------------------
# DataLoaders  
# --------------------
base_loader  = DataLoader(base_ds,  batch_size=batch_size, shuffle=True)
stack_loader = DataLoader(stack_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# --------------------
# Base Model Builders  
# --------------------
def build_eff():
    print("[INFO] Building EfficientNetV2-S")
    return timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes).to(device)

def build_vit():
    print("[INFO] Building ViT-B/32")
    return timm.create_model("vit_base_patch32_224", pretrained=True, num_classes=num_classes).to(device)

# --------------------
# Evaluation Helper 
# --------------------
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

# --------------------
# Training Function 
# --------------------
def train_model(model, train_loader, val_loader, epochs, lr, name="Model", ckpt_path=None):
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[{name}] Loaded checkpoint from {ckpt_path}")
        return model

    crit = nn.CrossEntropyLoss()
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    patience_counter = 0

    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        train_acc = evaluate_model(model, train_loader)
        val_acc   = evaluate_model(model, val_loader)

        print(f"[{name}] Epoch {ep}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict()
            if ckpt_path:
                torch.save(best_state, ckpt_path)
                print(f"[{name}] Saved new best checkpoint (val acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{name}] Early stopping at epoch {ep} (best val acc={best_acc:.4f})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model

# --------------------
# Logit Collection 
# --------------------
@torch.no_grad()
def collect_logits(model, loader, name="Model"):
    model.eval()
    X, y = [], []
    for x, t in loader:
        x = x.to(device)
        out = model(x)
        X.append(out.cpu().numpy()); y.append(t.numpy())
    X = np.concatenate(X, 0)
    y = np.concatenate(y, 0)
    print(f"[INFO] Collected logits from {name}: {X.shape}")
    return X, y

# --------------------
# Base Model Training/Loading 
# --------------------
eff = build_eff()
eff_ckpt = os.path.join(checkpoint_dir, "eff_best.pth")
print("[INFO] Training/Loading EfficientNet...")
eff = train_model(eff, base_loader, stack_loader, epochs, lr_eff, name="EffNet", ckpt_path=eff_ckpt)

vit = build_vit()
vit_ckpt = os.path.join(checkpoint_dir, "vit_best.pth")
print("[INFO] Training/Loading ViT...")
vit = train_model(vit, base_loader, stack_loader, epochs, lr_vit, name="ViT", ckpt_path=vit_ckpt)

# --------------------
# Base-Model Evaluation 
# --------------------
eff_acc = evaluate_model(eff, test_loader)
vit_acc = evaluate_model(vit, test_loader)
print(f"\n[BASE] EfficientNet Test Accuracy: {eff_acc:.4f}")
print(f"[BASE] ViT Test Accuracy: {vit_acc:.4f}")

# --------------------
# Meta-Learner Training
# --------------------
print("\n[INFO] Collecting stack-train logits...")
X_eff_stack, y_stack = collect_logits(eff, stack_loader, name="EffNet")
X_vit_stack, _       = collect_logits(vit, stack_loader, name="ViT")
X_meta_stack = np.concatenate([X_eff_stack, X_vit_stack], axis=1)
print(f"[INFO] Meta-train feature matrix shape: {X_meta_stack.shape}")

print("[INFO] Training Logistic Regression meta-learner...")
meta = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=1000)
meta.fit(X_meta_stack, y_stack)
print("[INFO] Meta-learner training complete.")

# Print coefficients and intercepts of logistic regression meta-learner
print("\n[LOGISTIC REGRESSION COEFFICIENTS]")
for i, cls in enumerate(full_train.classes):
    print(f"Class {cls}:")
    print(f"  Intercept: {meta.intercept_[i]:.4f}")
    print(f"  Coefficients (first 10 shown): {meta.coef_[i][:10]}")


# --------------------
# Ensemble Evaluation
# --------------------
print("\n[INFO] Evaluating ensemble on test set...")
X_eff_test, y_test = collect_logits(eff, test_loader, name="EffNet")
X_vit_test, _      = collect_logits(vit, test_loader, name="ViT")
X_meta_test = np.concatenate([X_eff_test, X_vit_test], axis=1)

y_pred = meta.predict(X_meta_test)
ens_acc = accuracy_score(y_test, y_pred)
correct = (y_pred == y_test).sum()
total = y_test.shape[0]

print(f"\n[STACK-HOLDOUT] Ensemble Test Accuracy: {ens_acc:.4f}")
print(f"[STACK-HOLDOUT] Correct/Total: {correct}/{total} ({correct/total:.4f})")
print("\n[STACK-HOLDOUT] Classification report:")
print(classification_report(y_test, y_pred, target_names=full_train.classes))

# Additional Performance Metrics
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("\n[STACK-HOLDOUT] Ensemble Performance Metrics:")
print(f"Accuracy : {ens_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# ------ ROC-AUC (multiclass) ------
# Uses predicted probabilities from the meta-learner
y_prob = meta.predict_proba(X_meta_test)
try:
    roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr")
    print(f"ROC-AUC (OvR): {roc_auc_ovr:.4f}")
    # OVO for reference
    roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo")
    print(f"ROC-AUC (OvO): {roc_auc_ovo:.4f}")
except ValueError as e:
    print(f"[WARN] Could not compute ROC-AUC: {e}")

# Per-class ROC curves plot
try:
    y_bin = label_binarize(y_test, classes=np.arange(num_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6, 6))
    for i, cls in enumerate(full_train.classes):
        plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Ensemble Meta-Learner")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ensemble_roc_auc.png"), dpi=300, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"[WARN] ROC curve plotting skipped: {e}")

cm = confusion_matrix(y_test, y_pred)
print("\n[STACK-HOLDOUT] Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_train.classes)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap="Blues", ax=ax, colorbar=False)
plt.title("Confusion Matrix – Ensemble Model")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ensemble_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# --------------------
# Explainability (Grad-CAM) 
# --------------------

def find_last_conv(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

eff_target_layers = [find_last_conv(eff)]
vit_target_layers = [vit.blocks[-1].norm1]


def vit32_reshape_transform(tensor):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), 7, 7, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

os.makedirs("saved_heatmaps", exist_ok=True)

print(f"\n[INFO] Generating Grad-CAM visualizations (with strict fusion)...")
print(f"[INFO] EffNet target layer: {eff_target_layers[0]}")
print(f"[INFO] ViT target layer:   {vit_target_layers[0]}")

test_filepaths = [fp for (fp, _) in test_ds.samples]
test_labels    = [lbl for (_, lbl) in test_ds.samples]


with GradCAM(model=eff, target_layers=eff_target_layers) as eff_cam, \
     GradCAM(model=vit, target_layers=vit_target_layers, reshape_transform=vit32_reshape_transform) as vit_cam:

    n_to_draw = min(5, len(test_filepaths))
    for i in range(n_to_draw):
        idx = np.random.randint(len(test_filepaths))
        img_path = test_filepaths[idx]
        true_idx = test_labels[idx]

        img = Image.open(img_path).convert("RGB")
        rgb_img = np.array(img.resize((224, 224))) / 255.0
        input_tensor = test_tf(img).unsqueeze(0).to(device)

        eff_logits = eff(input_tensor)
        vit_logits = vit(input_tensor)

        eff_probs = torch.softmax(eff_logits, dim=1)
        vit_probs = torch.softmax(vit_logits, dim=1)
        eff_conf, eff_idx = eff_probs.max(dim=1)
        vit_conf, vit_idx = vit_probs.max(dim=1)

        eff_targets = [ClassifierOutputTarget(int(eff_idx.item()))]
        vit_targets = [ClassifierOutputTarget(int(vit_idx.item()))]

        gray_eff = eff_cam(input_tensor=input_tensor, targets=eff_targets, aug_smooth=True)[0]
        gray_vit = vit_cam(input_tensor=input_tensor, targets=vit_targets)[0]

        vis_eff = show_cam_on_image(rgb_img, gray_eff, use_rgb=True)
        vis_vit = show_cam_on_image(rgb_img, gray_vit, use_rgb=True)

        fused = np.minimum(gray_eff, gray_vit)
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

        vis_fused = show_cam_on_image(rgb_img, fused, use_rgb=True)

        ens_probs = meta.predict_proba(
            np.concatenate([eff_logits.cpu().detach().numpy(),
                            vit_logits.cpu().detach().numpy()], axis=1)
        )[0]
        ens_idx = np.argmax(ens_probs)

        true_cls     = test_ds.classes[true_idx]
        ens_cls  = test_ds.classes[ens_idx]
        eff_cls      = test_ds.classes[int(eff_idx.item())]
        vit_cls      = test_ds.classes[int(vit_idx.item())]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(
            f"Original\nTrue: {true_cls}\n"
            f"EffNet: {eff_cls}\n"
            f"ViT: {vit_cls}\n"
            f"Ensemble: {ens_cls}\n"
        )
        axes[1].imshow(vis_eff);   axes[1].set_title("EffNet Grad-CAM")
        axes[2].imshow(vis_vit);   axes[2].set_title("ViT Grad-CAM")
        axes[3].imshow(vis_fused); axes[3].set_title("Fusion")

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        save_name = f"heatmap_{i}_{true_cls}_Eff{eff_cls}_ViT{vit_cls}.png"
        save_path = os.path.join("saved_heatmaps", save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap: {save_path}")


# --------------------
# Prediction Helper for base models 
# --------------------
@torch.no_grad()
def get_predictions_and_probs(model, loader):
    model.eval()
    preds, probs, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds.append(out.argmax(1).cpu().numpy())
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(preds),
            np.concatenate(probs),
            np.concatenate(labels))


# --------------------
# McNemar’s Test 
# --------------------
def mcnemar_test(y_true, base_preds, ens_preds, base_name):
    # 2x2 contingency table:
    #   b = base correct, ens wrong
    #   c = base wrong, ens correct
    both_correct = np.sum((base_preds == y_true) & (ens_preds == y_true))
    b = np.sum((base_preds == y_true) & (ens_preds != y_true))  # base only
    c = np.sum((base_preds != y_true) & (ens_preds == y_true))  # ensemble only
    both_wrong   = np.sum((base_preds != y_true) & (ens_preds != y_true))

    table = [[both_correct, b],
             [c,            both_wrong]]

    # Run McNemar’s test
    result = mcnemar(table, exact=True)

    # Effect size metrics
    N = len(y_true)
    delta_p = (c - b) / N  # net proportion difference

    print(f"\n[MCNEMAR’S TEST] Ensemble vs {base_name}")
    print(f"Statistic = {result.statistic}, p = {result.pvalue:.4e}")
    print(f"Proportion difference (Δp) = {delta_p:.4%}")
    
    
    if result.pvalue < 0.05:
        print(f"Ensemble significantly differs from {base_name} (p<0.05)")
    else:
        print(f"No significant difference detected vs {base_name}")

# Get predictions for base models
eff_preds, _, _ = get_predictions_and_probs(eff, test_loader)
vit_preds, _, _ = get_predictions_and_probs(vit, test_loader)

# Run tests
mcnemar_test(y_test, eff_preds, y_pred, "EfficientNet")
mcnemar_test(y_test, vit_preds, y_pred, "ViT")