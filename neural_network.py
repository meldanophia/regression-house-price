from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

#Muat dataset
data = fetch_california_housing()
X = data.data #fitur
y = data.target #target (harga rumah)

#Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalisasi data (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Bangun model (pilih 1)
# model = models.Sequential([
#     layers.Dense(50, activation='relu', input_shape=(X_train.shape[1],)), #Hidden layer 1
#     layers.Dense(100, activation='relu'), #Hidden layer 2
#     layers.Dense(1) #output layer
# ])

# Bangun model dengan Leaky ReLU
# model = models.Sequential([
#     layers.Dense(64, input_shape=(X_train.shape[1],)),
#     LeakyReLU(alpha=0.1),  # Leaky ReLU dengan alpha = 0.1
#     layers.Dense(32),
#     LeakyReLU(alpha=0.1),
#     layers.Dense(1)
# ])

# Dropout: menonaktifkan sebagian neuron scr acak selama training untuk menghindari overfitting
# model = models.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     layers.Dropout(0.2),  # Dropout 20% neuron
#     layers.Dense(32, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(1)
# ])

# L2 Regularization: menambahkan penalty pada weights yg besar
# model = models.Sequential([
#     layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
#     layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#     layers.Dense(1)
# ])

#Kombinasi antara dropout dengan L2 Regularization
model = models.Sequential([
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(1)
])

#kompilasi model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#Summary model
model.summary()

#Training (pilih 1)
# history = model.fit(X_train, y_train, epochs=70, validation_split=0.2)

early_stopping = EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

#Evaluasi
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")

# Hitung prediksi
y_pred = model.predict(X_test).flatten()  # flatten() untuk mengubah bentuk (n, 1) menjadi (n,)

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Plot MAE
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training vs Validation MAE')
plt.legend()
plt.show()

# Plot distribusi error (prediksi vs aktual)
error = y_test - y_pred
plt.hist(error, bins=50)
plt.xlabel('Error (Aktual - Prediksi)')
plt.ylabel('Frekuensi')
plt.title('Distribusi Error')
plt.show()