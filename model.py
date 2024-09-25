from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=448, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),
        
        Conv1D(filters=512, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.4),
        
        GlobalAveragePooling1D(),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.02)),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    history = model.fit(
        x_train, y_train, 
        validation_data=(x_val, y_val),
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=[early_stopping, reduce_lr]
    )
    
    return history
