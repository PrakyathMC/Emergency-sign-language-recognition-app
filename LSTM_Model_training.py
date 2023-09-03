#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
import glob as gb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
sns.set()
import optuna
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score
import optuna


# In[20]:


EPOCHS = 50
N_TRIALS = 10  # Number of Optuna trials
def prepare_data(dire):
    total_images = 3200
    signLang = ['accident', 'call', 'doctor', 'help', 'hot', 'lose', 'pain', 'thief']
    X = list(np.zeros(shape=(total_images , 128, 128)))
    y = list(np.zeros(shape=(total_images)))
    i=0
    cnt = 0
    y_value = 0
    for sign in signLang : 
        available_images = gb.glob(pathname= dire + f'{sign}/*.png')
        for image in available_images : 
            try : 
                if cnt == 400:
                    continue
                x = plt.imread(image)
                x = cv2.resize(x, (128, 128))

                X[i] = x
                y[i] = y_value
                i+=1
                cnt+=1
                    
            except : 
                pass
            
        y_value+=1
        cnt = 0
    ohe  = OneHotEncoder()
    y = np.array(y)
    y = y.reshape(len(y), 1)
    ohe.fit(y)
    y = ohe.transform(y).toarray()
    X = np.array(X)
    print(f'X shape is {X.shape}')
    print(f'y shape is {y.shape}')
    return X, y


# In[22]:


def create_lstm_model(input_shape, num_classes, trial=None):
    model = Sequential(
    Reshape((128, 384)),  # Reshape to a sequence of 64 time steps with 192 features
)
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(trial.suggest_int("dense_units1", 120, 128), activation="relu"))
    model.add(Dropout(trial.suggest_float("dropout2", 0.1, 0.2)))
    model.add(Dense(trial.suggest_int("dense_units1", 64, 128), activation="relu"))
    model.add(Dropout(trial.suggest_float("dropout3", 0.2, 0.5)))
    model.add(Dense(trial.suggest_int("dense_units1", 32, 64), activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model


# In[23]:


def objective(trial):
    input_dir = "/kaggle/input/emergency-sign-language/Image_Data/"
    log_dir = "Logs"
    X, y = prepare_data(input_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = X_train.shape[1:]
    print(input_shape)
    num_classes = y_train.shape[1]

    model = create_lstm_model(input_shape, num_classes, trial)

    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )

    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=EPOCHS, callbacks=[tb_callback], verbose=1, batch_size = 32)

    res = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(res, axis=1).tolist()

    acc = accuracy_score(ytrue, yhat)
    precision = precision_score(ytrue, yhat, average='weighted')
    recall = recall_score(ytrue, yhat, average='weighted')
    f1 = f1_score(ytrue, yhat, average='weighted')
    conf_matrix = confusion_matrix(ytrue, yhat)

    # Display the metrics
    print("Accuracy", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    return acc


# ### Model Folder

# In[24]:


input_dir = "/kaggle/input/emergency-sign-language/Image_Data/"
model_dir = "Models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# ### Running Optuna Hyperparamer Tunning 

# In[25]:


study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=N_TRIALS)  # Change number of trials here

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# ### Training The Best Model

# In[26]:


X, y = prepare_data(input_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


best_model = create_lstm_model(X_train.shape[1:], y_train.shape[1], trial)
best_model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)
history = best_model.fit(X_train, y_train, epochs=EPOCHS, batch_size= 32, validation_data = [X_test, y_test])


# ### Evaluation

# In[27]:


res = best_model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(res, axis=1).tolist()

acc = accuracy_score(ytrue, yhat)
precision = precision_score(ytrue, yhat, average='weighted')
recall = recall_score(ytrue, yhat, average='weighted')
f1 = f1_score(ytrue, yhat, average='weighted')
conf_matrix = confusion_matrix(ytrue, yhat)

# Display the metrics
print("Accuracy", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)


# In[28]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Test Accuracy : {}".format(accuracy_score(ytrue, yhat)))
print("\nClassification Report : ")
print(classification_report(ytrue, yhat))
print("\nConfusion Matrix : ")
print(confusion_matrix(ytrue, yhat))


# In[29]:


sns.set()
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(acc,color='b', label='Training Accuracy')
ax1.plot(val_acc,color='r', label='Validation Accuracy')
ax1.legend(loc='best', shadow=True)
ax1.set_ylabel('Accuracy')
ax1.set_title('Training and Validation Accuracy')

ax2.plot(loss,color='b', label='Training Loss')
ax2.plot(val_loss,color='r', label='Validation Loss')
ax2.legend(loc='best', shadow=True)
ax2.set_ylabel('Cross Entropy')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('epoch')
plt.tight_layout()

plt.show()
plt.savefig('Accuracy.png')


# In[30]:


from sklearn.metrics import confusion_matrix
import scikitplot as skplt
target_classes = ['accident', 'call', 'doctor', 'help', 'hot', 'lose', 'pain', 'thief']

skplt.metrics.plot_confusion_matrix([target_classes[i] for i in ytrue], [target_classes[i] for i in yhat],
                                    normalize=True,
                                    title="Confusion Matrix",
                                    cmap="Blues",
                                    hide_zeros=True,
                                    figsize=(15,10)
                                    );
plt.xticks(rotation=90);


# In[31]:


best_model.save(os.path.join(model_dir, "best_lstm_model.h5"))


# In[ ]:




