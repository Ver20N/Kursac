import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def train_and_test_lightweight_cnn(X, y, test_size=0.2, epochs=13, batch_size=32, lr=0.001):
    
    # Определение архитектуры модели
    class LightweightCNN(nn.Module):
        def __init__(self, num_classes=6):
            super(LightweightCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
    
    # Подготовка данных
    X = np.transpose(X, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Конвертация в тензоры
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Инициализация модели, оптимизатора и функции потерь
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightCNN(num_classes=len(np.unique(y))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Обучение
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        # Сохранение истории
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Снижение learning rate
        scheduler.step(avg_train_loss)
        
        # Сохранение лучшей модели
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('best_model.pth'))
    print(f'\nЛучшая точность на тесте: {best_test_acc:.2f}%')
    
    return model, history, best_test_acc


def train_and_test_vgg_like_cnn(X, y, test_size=0.2, epochs=13, batch_size=32, lr=0.001):
    """
    Более тяжелая VGG-подобная модель для сравнения скорости и точности.
    """
    class VGGLikeCNN(nn.Module):
        def __init__(self, num_classes=6):
            super(VGGLikeCNN, self).__init__()
            # Блок 1: 3 -> 64
            self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1_1 = nn.BatchNorm2d(64)
            self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn1_2 = nn.BatchNorm2d(64)
            # Блок 2: 64 -> 128
            self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2_1 = nn.BatchNorm2d(128)
            self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
            self.bn2_2 = nn.BatchNorm2d(128)
            # Блок 3: 128 -> 256
            self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3_1 = nn.BatchNorm2d(256)
            self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
            self.bn3_2 = nn.BatchNorm2d(256)
            # Блок 4: 256 -> 512
            self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
            self.bn4_1 = nn.BatchNorm2d(512)
            self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
            self.bn4_2 = nn.BatchNorm2d(512)

            self.pool = nn.MaxPool2d(2)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = F.relu(self.bn1_1(self.conv1_1(x)))
            x = F.relu(self.bn1_2(self.conv1_2(x)))
            x = self.pool(x)
            x = F.relu(self.bn2_1(self.conv2_1(x)))
            x = F.relu(self.bn2_2(self.conv2_2(x)))
            x = self.pool(x)
            x = F.relu(self.bn3_1(self.conv3_1(x)))
            x = F.relu(self.bn3_2(self.conv3_2(x)))
            x = self.pool(x)
            x = F.relu(self.bn4_1(self.conv4_1(x)))
            x = F.relu(self.bn4_2(self.conv4_2(x)))
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

    # Подготовка данных
    X = np.transpose(X, (0, 3, 1, 2))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGGLikeCNN(num_classes=len(np.unique(y))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        test_acc = 100. * test_correct / test_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        scheduler.step(avg_train_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model_vgg.pth')

        print(f'[VGG-like] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

    model.load_state_dict(torch.load('best_model_vgg.pth'))
    print(f'\n[VGG-like] Лучшая точность: {best_test_acc:.2f}%')
    return model, history, best_test_acc