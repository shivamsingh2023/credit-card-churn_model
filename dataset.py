import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. LOAD DATA

df = pd.read_csv("dataset.csv")

print("Dataset shape:", df.shape)

# 2. ENCODE CATEGORICAL DATA
df = pd.get_dummies(df, drop_first=True)



# 3. SPLIT FEATURES & TARGET

X = df.drop("Exited", axis=1)
y = df["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# 5. FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 6. CLASS WEIGHTS (IMPORTANT)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: 1.0,
    1: 2.5   # not too high
}

print("Class weights:", class_weights)


model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])



model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



# 9. EARLY STOPPING

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# 10. TRAIN MODEL

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)


# 11. MODEL EVALUATION

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 12. SAVE MODEL

model.save("churn_model.h5")
print("\nModel saved as churn_model.h5")
