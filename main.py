import os
import cv2
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score,
                           roc_auc_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_and_test_lightweight_cnn
from obrabotka import preprocess_for_lightweight_cnn
import torch

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
                    # Используем нашу функцию предобработки
                    img = preprocess_for_lightweight_cnn(img_path, target_size=(128, 128))
                    X.append(img)
                    y.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"Ошибка: {img_path} - {e}")
    
    return np.array(X), np.array(y), class_names

def calculate_multiclass_metrics(model, X_test, y_test, class_names, device='cpu'):
   
    import torch
    import torch.nn.functional as F
    
    model.eval()
    model.to(device)
    
    # Получение предсказаний
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        # Обрабатываем батчами, чтобы не перегружать память
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
    
    # 1. Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # 2. Precision, Recall, F1-score (macro и weighted)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # 3. Per-class metrics
    per_class_precision = precision_score(y_test, y_pred, average=None)
    per_class_recall = recall_score(y_test, y_pred, average=None)
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    
    # 4. Classification report
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    
    # 5. Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 6. ROC-AUC (One-vs-Rest)
    try:
        roc_auc_ovr = roc_auc_score(y_test, y_probs, multi_class='ovr', average='macro')
        roc_auc_ovo = roc_auc_score(y_test, y_probs, multi_class='ovo', average='macro')
    except:
        roc_auc_ovr = None
        roc_auc_ovo = None
    
    # Собираем все метрики в словарь
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

def plot_confusion_matrix(conf_matrix, class_names, save_path='confusion_matrix.png'):
    """
    Визуализация матрицы ошибок
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Матрица ошибок сохранена как {save_path}")

def plot_metrics_comparison(per_class_metrics, class_names, save_path='per_class_metrics.png'):
    """
    Визуализация метрик по каждому классу
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, per_class_metrics['precision'], width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, per_class_metrics['recall'], width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, per_class_metrics['f1'], width, label='F1-Score', color='salmon')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График метрик сохранен как {save_path}")

def print_all_metrics(metrics, class_names):
    """
    Красивое отображение всех метрик
    """
    print("\n" + "="*70)
    print("МЕТРИКИ МУЛЬТИКЛАССОВОЙ КЛАССИФИКАЦИИ")
    print("="*70)
    
    # Основные метрики
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
    
    # ROC-AUC
    if metrics['roc_auc_ovr'] is not None:
        print(f"\nROC-AUC SCORES:")
        print(f"  • One-vs-Rest (macro):  {metrics['roc_auc_ovr']:.4f}")
        print(f"  • One-vs-One (macro):   {metrics['roc_auc_ovo']:.4f}")
    
    # Per-class метрики
    print("\nМЕТРИКИ ПО КАЖДОМУ КЛАССУ:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {metrics['per_class_precision'][i]:<12.4f} "
              f"{metrics['per_class_recall'][i]:<12.4f} {metrics['per_class_f1'][i]:<12.4f}")
    
    # Полный classification report
    print("\nCLASSIFICATION REPORT:")
    print("-" * 70)
    print(metrics['classification_report'])
    
    # Статистика по матрице ошибок
    cm = metrics['confusion_matrix']
    print("\nАНАЛИЗ МАТРИЦЫ ОШИБОК:")
    print(f"  • True Positives (диагональ): {np.trace(cm)}/{np.sum(cm)} ({np.trace(cm)/np.sum(cm)*100:.2f}%)")
    
    # Находим самые частые ошибки
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

# Главная функция
if __name__ == "__main__":
    # Путь к папке с датасетом
    train_path = "NEU Surface Defect Database/train"
    val_path = "NEU Surface Defect Database/validation"
    
    print("Загрузка тренировочных данных...")
    X_train, y_train, class_names = load_dataset_from_folder(train_path)
    print(f"Загружено {len(X_train)} изображений")
    
    print("\nЗагрузка валидационных данных...")
    X_val, y_val, _ = load_dataset_from_folder(val_path)
    print(f"Загружено {len(X_val)} изображений")
    
    # Объединяем train и val для обучения
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    print(f"\nВсего изображений: {len(X_combined)}")
    print(f"Форма изображений: {X_combined.shape}")
    
    # Запуск обучения и тестирования
    print("\n" + "="*50)
    print("ЗАПУСК ОБУЧЕНИЯ ЛЕГКОВЕСНОЙ CNN")
    print("="*50)
    
    model, history, accuracy = train_and_test_lightweight_cnn(
        X=X_combined,
        y=y_combined,
        test_size=0.2,
        epochs=13,
        batch_size=32,
        lr=0.001
    )
    
    # После обучения, загружаем тестовые данные для детального анализа
    print("\n" + "="*50)
    print("РАСЧЕТ ВСЕХ МЕТРИК КЛАССИФИКАЦИИ")
    print("="*50)
    
    # Получаем тестовую выборку (20% от объединенных данных)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    # Транспонируем в формат (N, C, H, W) для модели
    X_test_tensor = np.transpose(X_test, (0, 3, 1, 2))
    X_test_tensor = torch.FloatTensor(X_test_tensor)
    
    # Расчет метрик
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = calculate_multiclass_metrics(model, X_test_tensor, y_test, class_names, device)
    
    # Вывод всех метрик
    print_all_metrics(metrics, class_names)
    
    # Визуализация матрицы ошибок
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, 'confusion_matrix.png')
    
    # Визуализация метрик по классам
    per_class_metrics = {
        'precision': metrics['per_class_precision'],
        'recall': metrics['per_class_recall'],
        'f1': metrics['per_class_f1']
    }
    plot_metrics_comparison(per_class_metrics, class_names, 'per_class_metrics.png')
    
    # Сохранение всех метрик в файл
    with open('classification_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION METRICS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n\n")
        f.write("Macro Average:\n")
        f.write(f"  Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"  Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1_macro']:.4f}\n\n")
        f.write("Weighted Average:\n")
        f.write(f"  Precision: {metrics['precision_weighted']:.4f}\n")
        f.write(f"  Recall: {metrics['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1_weighted']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'])
    
    print("\nВсе метрики сохранены в файл 'classification_metrics.txt'")