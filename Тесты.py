import matplotlib.pyplot as plt

# Данные
epochs = list(range(1, 14))
train_loss = [1.2114, 0.6560, 0.5061, 0.4513, 0.3791, 0.2907, 0.2807, 
              0.3234, 0.2274, 0.2195, 0.1977, 0.1661, 0.1822]
train_acc = [58, 77, 80, 82, 86, 89, 88, 87, 91, 93, 92, 94, 93]
test_acc = [31, 81, 75, 62, 71, 83, 50, 89, 50, 84, 90, 90, 91]

# Настройка стиля (похоже на стиль из научных статей)
plt.style.use('seaborn-v0_8-darkgrid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График потерь
ax1.plot(epochs, train_loss, marker='o', linestyle='-', linewidth=2, 
         markersize=8, color='#2E86AB', label='Train Loss')
ax1.fill_between(epochs, train_loss, alpha=0.2, color='#2E86AB')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('(a) Loss Function', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xticks(epochs)

# График точности
ax2.plot(epochs, train_acc, marker='s', linestyle='-', linewidth=2, 
         markersize=8, color='#A23B72', label='Train Accuracy')
ax2.plot(epochs, test_acc, marker='^', linestyle='-', linewidth=2, 
         markersize=8, color='#F18F01', label='Test Accuracy')
ax2.fill_between(epochs, train_acc, alpha=0.1, color='#A23B72')
ax2.fill_between(epochs, test_acc, alpha=0.1, color='#F18F01')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Classification Accuracy', fontsize=14, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xticks(epochs)
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()  