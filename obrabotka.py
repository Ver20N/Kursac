import cv2
import numpy as np

def preprocess_for_lightweight_cnn(image_path, target_size=(128, 128)):
   
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить: {image_path}")
    
    # 1. Улучшение локального контраста (CLAHE) - ключевой шаг
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    # 2. Unsharp masking для выделения границ дефектов
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
    
    # 3. Быстрое выделение границ (адаптивный Кэнни)
    median = np.median(sharpened)
    edges = cv2.Canny(sharpened, 
                      int(max(0, 0.5 * median)), 
                      int(min(255, 1.2 * median)))
    
    # 4. Комбинированный фич-канал (оригинал + усиленный + границы)
    combined = np.stack([
        cv2.resize(enhanced, target_size),
        cv2.resize(sharpened, target_size),
        cv2.resize(edges, target_size)
    ], axis=-1)
    
    # 5. Нормализация в [0, 1]
    combined = combined.astype(np.float32) / 255.0
    
    return combined