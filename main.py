import os
import time
import gc
import cv2
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                           precision_score, recall_score, f1_score,
                           roc_auc_score, accuracy_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from model import train_and_test_lightweight_cnn, train_and_test_vgg_like_cnn
from obrabotka import preprocess_for_lightweight_cnn


# ============================================================
# ЗАГРУЗКА ДАТАСЕТА
# ============================================================
def load_dataset_from_folder(data_path):
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    X = []
    y = []

    for class_name in class_names:
        class_path = os.path.join(data_path, 'images', class_name)
        if not os.path.exists(class_path):
            print(f"Папка не найдена: {class_path}")
            continue

        for img_file in os.listdir(class_path):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = preprocess_for_lightweight_cnn(img_path, target_size=(128, 128))
                    X.append(img)
                    y.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"Ошибка: {img_path} - {e}")

    return np.array(X), np.array(y), class_names


# ============================================================
# МЕТРИКИ
# ============================================================
def calculate_multiclass_metrics(model, X_test, y_test, class_names, device='cpu'):
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    per_class_precision = precision_score(y_test, y_pred, average=None)
    per_class_recall = recall_score(y_test, y_pred, average=None)
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    class_report = classification_report(y_test, y_pred, target_names=class_names)
    conf_matrix = confusion_matrix(y_test, y_pred)

    try:
        roc_auc_ovr = roc_auc_score(y_test, y_probs, multi_class='ovr', average='macro')
        roc_auc_ovo = roc_auc_score(y_test, y_probs, multi_class='ovo', average='macro')
    except Exception:
        roc_auc_ovr = None
        roc_auc_ovo = None

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'roc_auc_ovr': roc_auc_ovr,
        'roc_auc_ovo': roc_auc_ovo,
        'y_pred': y_pred,
        'y_probs': y_probs
    }

    return metrics


# ============================================================
# ЗАМЕР СКОРОСТИ
# ============================================================
def measure_inference_speed(model, X_test_tensor, device='cpu', n_runs=3):
    model.eval()
    model.to(device)
    X_test_tensor = X_test_tensor.to(device)

    with torch.no_grad():
        _ = model(X_test_tensor[:min(32, len(X_test_tensor))])

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(X_test_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    per_image_ms = (avg_time / len(X_test_tensor)) * 1000
    return avg_time, per_image_ms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models_speed(models_dict, X_test_tensor, device='cpu'):
    print("\n" + "="*70)
    print("СРАВНЕНИЕ СКОРОСТИ И РАЗМЕРА МОДЕЛЕЙ")
    print("="*70)
    print(f"{'Модель':<22} {'Параметры':<15} {'Время (с)':<15} {'мс/изобр':<15}")
    print("-"*70)

    results = {}
    for name, model in models_dict.items():
        avg_time, per_img = measure_inference_speed(model, X_test_tensor, device, n_runs=3)
        n_params = count_parameters(model)
        results[name] = {'time': avg_time, 'per_image_ms': per_img, 'params': n_params}
        print(f"{name:<22} {n_params:<15,} {avg_time:<15.4f} {per_img:<15.4f}")

    fastest = min(results.items(), key=lambda x: x[1]['time'])
    print("-"*70)
    print(f"Самая быстрая модель: {fastest[0]} ({fastest[1]['per_image_ms']:.4f} мс/изобр)")
    return results


# ============================================================
# ВИЗУАЛИЗАЦИЯ: ПО КАЖДОЙ МОДЕЛИ
# ============================================================
def plot_confusion_matrix(conf_matrix, class_names, save_path, model_name=''):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    title = 'Confusion Matrix'
    if model_name:
        title += f' — {model_name}'
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Матрица ошибок сохранена как {save_path}")


def plot_metrics_comparison(per_class_metrics, class_names, save_path, model_name=''):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.25

    bars1 = ax.bar(x - width, per_class_metrics['precision'], width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, per_class_metrics['recall'], width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, per_class_metrics['f1'], width, label='F1-Score', color='salmon')

    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    title = 'Per-Class Metrics Comparison'
    if model_name:
        title += f' — {model_name}'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График метрик сохранён как {save_path}")


# ============================================================
# ВИЗУАЛИЗАЦИЯ: СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================
def plot_models_accuracy_comparison(model_names, accuracies, params,
                                    save_path='models_accuracy_comparison.png'):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    width = 0.35

    bars = ax1.bar(x, accuracies, width, color=['#4C72B0', '#DD8452'], label='Точность, %')
    ax1.set_ylabel('Точность, %', fontsize=12)
    ax1.set_title('Сравнение точности моделей', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, acc, p in zip(bars, accuracies, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{acc:.2f}%\n({p:,} пар.)', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сравнения точности сохранён как {save_path}")


def plot_models_speed_comparison(model_names, times_total, times_per_image,
                                 save_path='models_speed_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(model_names))
    width = 0.5
    colors = ['#4C72B0', '#DD8452']

    bars1 = axes[0].bar(x, times_total, width, color=colors)
    axes[0].set_ylabel('Время, с', fontsize=12)
    axes[0].set_title('Общее время инференса на тестовой выборке', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars1, times_total):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{t:.4f} с', ha='center', va='bottom', fontsize=9)

    bars2 = axes[1].bar(x, times_per_image, width, color=colors)
    axes[1].set_ylabel('Время, мс/изобр', fontsize=12)
    axes[1].set_title('Время инференса на одно изображение', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars2, times_per_image):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{t:.4f} мс', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сравнения скорости сохранён как {save_path}")


def plot_models_full_comparison(model_names, accuracies, params,
                                times_total, times_per_image,
                                save_path='models_full_comparison.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(model_names))
    width = 0.5
    colors = ['#4C72B0', '#DD8452']

    bars1 = axes[0].bar(x, accuracies, width, color=colors)
    axes[0].set_ylabel('Точность, %', fontsize=12)
    axes[0].set_title('Точность моделей', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)

    bars2 = axes[1].bar(x, params, width, color=colors)
    axes[1].set_ylabel('Число параметров', fontsize=12)
    axes[1].set_title('Размер моделей', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, p in zip(bars2, params):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{p:,}', ha='center', va='bottom', fontsize=9)

    bars3 = axes[2].bar(x, times_per_image, width, color=colors)
    axes[2].set_ylabel('мс/изобр', fontsize=12)
    axes[2].set_title('Время инференса на изображение', fontsize=13)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, fontsize=11)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars3, times_per_image):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{t:.4f} мс', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Комплексный график сравнения сохранён как {save_path}")


# ============================================================
# ПЕЧАТЬ МЕТРИК
# ============================================================
def format_metrics_text(metrics, class_names, model_name):
    lines = []
    lines.append("=" * 70)
    lines.append(f"МЕТРИКИ МОДЕЛИ: {model_name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    lines.append("")
    lines.append("Macro Average:")
    lines.append(f"  Precision: {metrics['precision_macro']:.4f}")
    lines.append(f"  Recall:    {metrics['recall_macro']:.4f}")
    lines.append(f"  F1-Score:  {metrics['f1_macro']:.4f}")
    lines.append("")
    lines.append("Weighted Average:")
    lines.append(f"  Precision: {metrics['precision_weighted']:.4f}")
    lines.append(f"  Recall:    {metrics['recall_weighted']:.4f}")
    lines.append(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
    lines.append("")
    if metrics['roc_auc_ovr'] is not None:
        lines.append(f"ROC-AUC OvR (macro): {metrics['roc_auc_ovr']:.4f}")
        lines.append(f"ROC-AUC OvO (macro): {metrics['roc_auc_ovo']:.4f}")
        lines.append("")
    lines.append("Метрики по классам:")
    lines.append("-" * 70)
    lines.append(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    lines.append("-" * 70)
    for i, class_name in enumerate(class_names):
        lines.append(f"{class_name:<20} {metrics['per_class_precision'][i]:<12.4f} "
                     f"{metrics['per_class_recall'][i]:<12.4f} "
                     f"{metrics['per_class_f1'][i]:<12.4f}")
    lines.append("")
    lines.append("Classification Report:")
    lines.append("-" * 70)
    lines.append(metrics['classification_report'])
    lines.append("")
    return "\n".join(lines)


def print_all_metrics(metrics, class_names, model_name=''):
    header = "МЕТРИКИ МУЛЬТИКЛАССОВОЙ КЛАССИФИКАЦИИ"
    if model_name:
        header += f" — {model_name}"

    print("\n" + "="*70)
    print(header)
    print("="*70)

    print("\nОСНОВНЫЕ МЕТРИКИ:")
    print(f"  • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\n  • Macro Average:")
    print(f"    - Precision: {metrics['precision_macro']:.4f}")
    print(f"    - Recall:    {metrics['recall_macro']:.4f}")
    print(f"    - F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"\n  • Weighted Average:")
    print(f"    - Precision: {metrics['precision_weighted']:.4f}")
    print(f"    - Recall:    {metrics['recall_weighted']:.4f}")
    print(f"    - F1-Score:  {metrics['f1_weighted']:.4f}")

    if metrics['roc_auc_ovr'] is not None:
        print(f"\nROC-AUC SCORES:")
        print(f"  • One-vs-Rest (macro):  {metrics['roc_auc_ovr']:.4f}")
        print(f"  • One-vs-One (macro):   {metrics['roc_auc_ovo']:.4f}")

    print("\nМЕТРИКИ ПО КАЖДОМУ КЛАССУ:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {metrics['per_class_precision'][i]:<12.4f} "
              f"{metrics['per_class_recall'][i]:<12.4f} "
              f"{metrics['per_class_f1'][i]:<12.4f}")

    print("\nCLASSIFICATION REPORT:")
    print("-" * 70)
    print(metrics['classification_report'])

    cm = metrics['confusion_matrix']
    print("\nАНАЛИЗ МАТРИЦЫ ОШИБОК:")
    print(f"  • True Positives (диагональ): {np.trace(cm)}/{np.sum(cm)} "
          f"({np.trace(cm)/np.sum(cm)*100:.2f}%)")

    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((class_names[i], class_names[j], cm[i, j]))

    if errors:
        errors.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  • Топ-5 самых частых ошибок:")
        for i, (true_class, pred_class, count) in enumerate(errors[:5]):
            print(f"    {i+1}. {true_class} → {pred_class}: {count} ошибок")


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_path = "NEU Surface Defect Database/train"
    val_path = "NEU Surface Defect Database/validation"

    print("Загрузка тренировочных данных...")
    X_train, y_train, class_names = load_dataset_from_folder(train_path)
    print(f"Загружено {len(X_train)} изображений")

    print("\nЗагрузка валидационных данных...")
    X_val, y_val, _ = load_dataset_from_folder(val_path)
    print(f"Загружено {len(X_val)} изображений")

    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)

    print(f"\nВсего изображений: {len(X_combined)}")
    print(f"Форма изображений: {X_combined.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # --------------------------------------------------------
    # МОДЕЛЬ 1: Lightweight CNN
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("МОДЕЛЬ 1: LIGHTWEIGHT CNN")
    print("="*60)

    model_lw, history_lw, acc_lw = train_and_test_lightweight_cnn(
        X=X_combined, y=y_combined,
        test_size=0.2, epochs=13, batch_size=32, lr=0.001
    )
    free_memory()

    # --------------------------------------------------------
    # МОДЕЛЬ 2: VGG-like CNN
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("МОДЕЛЬ 2: VGG-LIKE CNN")
    print("="*60)

    model_vgg, history_vgg, acc_vgg = train_and_test_vgg_like_cnn(
        X=X_combined, y=y_combined,
        test_size=0.2, epochs=13, batch_size=32, lr=0.001
    )
    free_memory()

    # --------------------------------------------------------
    # ЕДИНАЯ ТЕСТОВАЯ ВЫБОРКА
    # --------------------------------------------------------
    _, X_test, _, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    X_test_tensor = torch.FloatTensor(np.transpose(X_test, (0, 3, 1, 2)))

    # --------------------------------------------------------
    # МЕТРИКИ ПО КАЖДОЙ МОДЕЛИ
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("МЕТРИКИ: LIGHTWEIGHT CNN")
    print("="*60)
    metrics_lw = calculate_multiclass_metrics(
        model_lw, X_test_tensor, y_test, class_names, device
    )
    print_all_metrics(metrics_lw, class_names, model_name="LightweightCNN")
    plot_confusion_matrix(
        metrics_lw['confusion_matrix'], class_names,
        save_path='confusion_matrix_lightweight.png',
        model_name='LightweightCNN'
    )
    plot_metrics_comparison(
        {
            'precision': metrics_lw['per_class_precision'],
            'recall': metrics_lw['per_class_recall'],
            'f1': metrics_lw['per_class_f1']
        },
        class_names,
        save_path='per_class_metrics_lightweight.png',
        model_name='LightweightCNN'
    )
    with open('metrics_lightweight.txt', 'w', encoding='utf-8') as f:
        f.write(format_metrics_text(metrics_lw, class_names, "LightweightCNN"))

    print("\n" + "="*60)
    print("МЕТРИКИ: VGG-LIKE CNN")
    print("="*60)
    metrics_vgg = calculate_multiclass_metrics(
        model_vgg, X_test_tensor, y_test, class_names, device
    )
    print_all_metrics(metrics_vgg, class_names, model_name="VGGLikeCNN")
    plot_confusion_matrix(
        metrics_vgg['confusion_matrix'], class_names,
        save_path='confusion_matrix_vgg.png',
        model_name='VGGLikeCNN'
    )
    plot_metrics_comparison(
        {
            'precision': metrics_vgg['per_class_precision'],
            'recall': metrics_vgg['per_class_recall'],
            'f1': metrics_vgg['per_class_f1']
        },
        class_names,
        save_path='per_class_metrics_vgg.png',
        model_name='VGGLikeCNN'
    )
    with open('metrics_vgg.txt', 'w', encoding='utf-8') as f:
        f.write(format_metrics_text(metrics_vgg, class_names, "VGGLikeCNN"))

    # --------------------------------------------------------
    # СРАВНЕНИЕ СКОРОСТИ
    # --------------------------------------------------------
    speed_results = compare_models_speed(
        {'LightweightCNN': model_lw, 'VGGLikeCNN': model_vgg},
        X_test_tensor, device
    )

    # --------------------------------------------------------
    # СРАВНЕНИЕ ТОЧНОСТИ
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("СРАВНЕНИЕ ТОЧНОСТИ")
    print("="*70)
    print(f"{'Модель':<22} {'Точность на тесте':<20} {'Параметры':<15}")
    print("-"*70)
    print(f"{'LightweightCNN':<22} {acc_lw:<20.2f}% {speed_results['LightweightCNN']['params']:<15,}")
    print(f"{'VGGLikeCNN':<22} {acc_vgg:<20.2f}% {speed_results['VGGLikeCNN']['params']:<15,}")

    if acc_lw >= acc_vgg:
        best_name, best_acc = "LightweightCNN", acc_lw
    else:
        best_name, best_acc = "VGGLikeCNN", acc_vgg

    print("\n" + "-"*70)
    print(f"Лучшая модель по точности: {best_name} ({best_acc:.2f}%)")

    # --------------------------------------------------------
    # ГРАФИЧЕСКОЕ СРАВНЕНИЕ МОДЕЛЕЙ
    # --------------------------------------------------------
    model_names = ['LightweightCNN', 'VGGLikeCNN']
    accuracies = [acc_lw, acc_vgg]
    params = [
        speed_results['LightweightCNN']['params'],
        speed_results['VGGLikeCNN']['params']
    ]
    times_total = [
        speed_results['LightweightCNN']['time'],
        speed_results['VGGLikeCNN']['time']
    ]
    times_per_image = [
        speed_results['LightweightCNN']['per_image_ms'],
        speed_results['VGGLikeCNN']['per_image_ms']
    ]

    plot_models_accuracy_comparison(model_names, accuracies, params)
    plot_models_speed_comparison(model_names, times_total, times_per_image)
    plot_models_full_comparison(model_names, accuracies, params, times_total, times_per_image)

    # --------------------------------------------------------
    # ОБЩИЙ ОТЧЁТ
    # --------------------------------------------------------
    with open('classification_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("СРАВНЕНИЕ МОДЕЛЕЙ\n")
        f.write("="*70 + "\n\n")
        f.write(f"Устройство: {device}\n\n")

        f.write("СКОРОСТЬ ИНФЕРЕНСА:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Модель':<22} {'Параметры':<15} {'Время (с)':<15} {'мс/изобр':<15}\n")
        for name, r in speed_results.items():
            f.write(f"{name:<22} {r['params']:<15,} {r['time']:<15.4f} {r['per_image_ms']:<15.4f}\n")
        f.write("\n")

        f.write("ТОЧНОСТЬ:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Модель':<22} {'Точность':<20}\n")
        f.write(f"{'LightweightCNN':<22} {acc_lw:<20.2f}%\n")
        f.write(f"{'VGGLikeCNN':<22} {acc_vgg:<20.2f}%\n\n")

        f.write(f"Лучшая модель по точности: {best_name} ({best_acc:.2f}%)\n\n")

        f.write(format_metrics_text(metrics_lw, class_names, "LightweightCNN"))
        f.write("\n\n")
        f.write(format_metrics_text(metrics_vgg, class_names, "VGGLikeCNN"))