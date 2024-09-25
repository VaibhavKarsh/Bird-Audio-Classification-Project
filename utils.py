import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, x_test, y_test, class_names):
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

def plot_class_accuracy(y_true, y_pred, class_names):
    y_pred_classes = np.argmax(y_pred, axis=-1)
    cm = confusion_matrix(y_true, y_pred_classes)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_names, y=per_class_accuracy, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Per-Class Accuracy Comparison', fontsize=16)
    plt.xlabel('Class Labels', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
