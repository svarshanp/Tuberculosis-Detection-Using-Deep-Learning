"""
Model Evaluation for Tuberculosis Detection (PyTorch) - 128px optimized
"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMG_SIZE = 128; BATCH_SIZE = 32
CLASS_NAMES = ['Normal', 'TB']
DEVICE = torch.device('cpu')

plt.rcParams.update({'figure.facecolor': '#0e1117', 'axes.facecolor': '#1a1a2e',
                      'text.color': 'white', 'axes.labelcolor': 'white',
                      'xtick.color': 'white', 'ytick.color': 'white',
                      'figure.dpi': 120, 'savefig.bbox': 'tight'})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_training import CustomCNN, build_transfer_model


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0), ds


def load_model(name):
    if name == 'Custom_CNN':
        model = CustomCNN()
    else:
        model = build_transfer_model(name)
    best = os.path.join(MODELS_DIR, f'{name}_best.pth')
    path = os.path.join(MODELS_DIR, f'{name}.pth')
    p = best if os.path.exists(best) else path
    if not os.path.exists(p): return None
    model.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    model.to(DEVICE); model.eval()
    return model


def get_predictions(model, loader):
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(DEVICE))
            probs = torch.softmax(out, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, annot_kws={'size': 18}, linewidths=1, linecolor='white')
    ax.set_xlabel('Predicted', fontsize=13); ax.set_ylabel('Actual', fontsize=13)
    ax.set_title(f'{name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f'{name}_confusion_matrix.png')); plt.close()


def plot_roc(y_true, y_prob, name):
    fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#00d4aa', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.15, color='#00d4aa')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{name} - ROC Curve', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f'{name}_roc_curve.png')); plt.close()
    return roc_auc


def plot_training_curves(name):
    hp = os.path.join(RESULTS_DIR, f'{name}_history.json')
    if not os.path.exists(hp): return
    with open(hp) as f: h = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(h['accuracy'], color='#00d4aa', lw=2, label='Train')
    axes[0].plot(h['val_accuracy'], color='#ff6b6b', lw=2, label='Val')
    axes[0].set_title(f'{name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(h['loss'], color='#00d4aa', lw=2, label='Train')
    axes[1].plot(h['val_loss'], color='#ff6b6b', lw=2, label='Val')
    axes[1].set_title(f'{name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f'{name}_training_curves.png')); plt.close()


def plot_all_roc(results):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#00d4aa', '#ff6b6b', '#ffd93d', '#6c5ce7']
    for (n, r), c in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
        ax.plot(fpr, tpr, color=c, lw=2.5, label=f'{n} (AUC = {auc(fpr, tpr):.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('All Models - ROC Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'all_roc_curves.png')); plt.close()


def plot_model_comparison(results):
    names, accs, f1s, aucs = [], [], [], []
    for n, r in results.items():
        names.append(n); accs.append(r['test_accuracy']*100)
        f1s.append(r['report']['weighted avg']['f1-score']*100)
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob']); aucs.append(auc(fpr, tpr)*100)
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(14, 7))
    b1 = ax.bar(x-w, accs, w, label='Accuracy', color='#00d4aa', edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x, f1s, w, label='F1-Score', color='#ff6b6b', edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x+w, aucs, w, label='AUC', color='#ffd93d', edgecolor='white', linewidth=0.5)
    for bars in [b1,b2,b3]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
    ax.set_ylabel('Score (%)'); ax.set_title('Model Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names); ax.legend(); ax.set_ylim(0, 110)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png')); plt.close()


def generate_gradcam(model, name, loader, n=4):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(2, n, figsize=(20, 8))
    for i in range(min(n, len(images))):
        img_t = images[i].unsqueeze(0).requires_grad_(True)
        img_np = images[i].numpy().transpose(1,2,0)
        img_np = np.clip((img_np * [0.229,0.224,0.225] + [0.485,0.456,0.406]) * 255, 0, 255).astype(np.uint8)
        out = model(img_t); pc = out.argmax(1).item(); tc = labels[i].item()
        axes[0,i].imshow(img_np); axes[0,i].axis('off')
        color = '#00d4aa' if pc == tc else '#ff6b6b'
        prob = torch.softmax(out, dim=1)[0, pc].item()
        axes[0,i].set_title(f'True:{CLASS_NAMES[tc]}\nPred:{CLASS_NAMES[pc]} ({prob:.2f})', fontsize=10, color=color)
        try:
            out[0, pc].backward(retain_graph=True)
            grad = img_t.grad.data.abs().squeeze().mean(0).numpy()
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
            hm = cv2.applyColorMap(np.uint8(255*cv2.resize(grad,(128,128))), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            sup = cv2.addWeighted(cv2.resize(img_np,(128,128)), 0.6, hm, 0.4, 0)
            axes[1,i].imshow(sup)
        except: axes[1,i].imshow(img_np)
        axes[1,i].axis('off'); axes[1,i].set_title('Grad-CAM', fontsize=10)
    fig.suptitle(f'{name} - Grad-CAM', fontsize=18, fontweight='bold', color='white')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f'{name}_gradcam.png')); plt.close()


def main():
    print("="*60, flush=True)
    print("TUBERCULOSIS DETECTION - MODEL EVALUATION", flush=True)
    print("="*60, flush=True)

    test_loader, _ = get_test_loader()
    model_names = ['Custom_CNN', 'ResNet50', 'EfficientNetB0']
    all_results = {}

    for name in model_names:
        model = load_model(name)
        if model is None: print(f"  {name}: not found, skip", flush=True); continue

        print(f"\nEvaluating {name}...", flush=True)
        y_true, y_pred, y_prob = get_predictions(model, test_loader)
        acc = (y_true == y_pred).mean()
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES), flush=True)

        res = {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
               'test_accuracy': float(acc), 'report': report}
        plot_confusion_matrix(y_true, y_pred, name); print(f"  [OK] Confusion matrix", flush=True)
        res['auc'] = plot_roc(y_true, y_prob, name); print(f"  [OK] ROC curve (AUC={res['auc']:.4f})", flush=True)
        plot_training_curves(name); print(f"  [OK] Training curves", flush=True)
        try: generate_gradcam(model, name, test_loader); print(f"  [OK] Grad-CAM", flush=True)
        except Exception as e: print(f"  [FAIL] Grad-CAM failed: {e}", flush=True)
        all_results[name] = res; del model

    if all_results:
        plot_all_roc(all_results); print("\n[OK] All ROC curves", flush=True)
        plot_model_comparison(all_results); print("[OK] Model comparison chart", flush=True)

        summary = {}
        for n, r in all_results.items():
            summary[n] = {
                'test_accuracy': r['test_accuracy'], 'test_loss': 0, 'auc': r.get('auc', 0),
                'precision_normal': r['report']['Normal']['precision'],
                'recall_normal': r['report']['Normal']['recall'],
                'f1_normal': r['report']['Normal']['f1-score'],
                'precision_tb': r['report']['TB']['precision'],
                'recall_tb': r['report']['TB']['recall'],
                'f1_tb': r['report']['TB']['f1-score'],
                'weighted_f1': r['report']['weighted avg']['f1-score'],
            }
        with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'Model':<20} {'Accuracy':>10} {'AUC':>10} {'F1':>10}", flush=True)
        print("-"*52, flush=True)
        for n, s in summary.items():
            print(f"{n:<20} {s['test_accuracy']*100:>9.2f}% {s['auc']*100:>9.2f}% {s['weighted_f1']*100:>9.2f}%", flush=True)
        best = max(summary.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\nBest Model: {best[0]} ({best[1]['test_accuracy']*100:.2f}%)", flush=True)

    print("\nEVALUATION COMPLETE!", flush=True)

if __name__ == '__main__':
    main()
