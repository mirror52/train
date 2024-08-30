import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import itertools
import matplotlib
from sklearn.preprocessing import LabelEncoder
matplotlib.rc('font', family='MicroSoft YaHei')
import seaborn as sns
from sklearn.metrics import accuracy_score

def plot_cm(label, preds, name='BP'):
    accuracy = accuracy_score(label, preds)
    # 混淆矩阵
    cm = confusion_matrix(label, preds)
    print(f"Confusion Matrix:\n{cm}")

    cm_df = pd.DataFrame(cm, index=true_labels, columns=true_labels)
    cm_df = cm_df.div(cm_df.sum(axis=1), axis=0) #归一化
    plt.figure(figsize=(8, 6))
    # sns.set()
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt=".1f", vmax=1, vmin=0)
    plt.title(name + " Overall accuracy: " + f"{accuracy:.4f}", fontweight="bold") #Confusion matrix
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ax = plt.gca()
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    plt.savefig(f'./data/Confusion Matrix_{name}', bbox_inches='tight', dpi=600)


data = pd.read_excel("data.xlsx")
true_labels = np.unique(data["Class"])
true_labels = ['An I', 'An II', 'An III', 'OJ', 'PJ']
original_labels = data["Class"]
print(true_labels)
label_encoder = LabelEncoder()
data["Class"] = label_encoder.fit_transform(data['Class'])

data = data.values
labels = np.array(data[:, -1], dtype=int) #(1274,)
data = data[:, :-1] #(1274,) (1274, 8)
print(data.shape)
print(labels.shape)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# 初始化 StandardScaler 对象
scaler = StandardScaler()

# 对训练集和验证集进行标准化
# 对测试集进行标准化
# 注意：测试集应该使用训练集的均值和标准差进行标准化，以避免数据泄露
# scaler.fit(x_train)
# x_train, x_val, x_test = scaler.transform(x_train), \
#                         scaler.transform(x_val), \
#                         scaler.transform(x_test)

# 将数据转换为TensorFlow Dataset
def numpy_to_dataset(X, y, batch_size):
    # 将NumPy数组转换为TensorFlow常量
    X_tensor = tf.constant(X)
    y_tensor = tf.constant(y)
    # 创建数据集
    return tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor)).batch(batch_size)

bs = 512
# 创建训练集、验证集和测试集的Dataset
train_dataset = numpy_to_dataset(x_train, y_train, bs)
val_dataset = numpy_to_dataset(x_val, y_val, bs)
test_dataset = numpy_to_dataset(x_test, y_test, bs)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(true_labels))
])

# 编译模型
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 模型摘要
model.summary()

model_checkpoint = ModelCheckpoint(
    r".\best_model.h5", monitor="val_loss", save_best_only=True
)

# 训练模型
epochs = 1000
history = model.fit(train_dataset, 
                    epochs=epochs, 
                    validation_data=val_dataset, 
                    callbacks=[model_checkpoint])

with open('history.txt', 'wb') as f:
    pickle.dump(history.history, f)

# 评估模型
model.load_weights('./best_model.h5')
labels, preds = [], []
for i in test_dataset:
    x, y = i
    pre = model.predict(x, verbose=0)
    #选择概率值最大的为其类别

    labels.extend(list(np.array(y)))
    pre = np.argmax(pre, axis=1)
    preds.extend(pre)
labels = np.array(labels)
preds = np.array(preds)
acc = (preds == labels).sum() / len(preds)
print(f"Test accuracy: {acc * 100:.2f}%")
print(classification_report(labels, preds))


plot_cm(labels, preds)
# plt.savefig('./混淆矩阵.png')

# for i in test_dataset:
#     test_x1, test_y1 = i
#     test_x1 = test_x1[0]
#     test_y1 = test_y1[0]
#     break

# test_x1 = np.expand_dims(test_x1, 0)
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_x1, verbose=0)
# pre_num = np.argmax(predictions)

# import cv2
# plt.figure()
# fig, axes = plt.subplots(1, 2, figsize=(8,8))
# fig.suptitle(f'预测为{name[pre_num]}的概率为{predictions[0, pre_num]*100:.2f}%', fontsize=12, y=0.8)
# label_path = f'./Final_Figure/{test_y1.numpy()+1}.jpg'
# pre_path = f'./Final_Figure/{pre_num+1}.jpg'

# label = cv2.imread(label_path)
# pre = cv2.imread(pre_path)

# axes[0].set_title('True Object')
# axes[0].imshow(label/255.)

# axes[1].set_title('Predict Object')
# axes[1].imshow(pre/255)
# for i in range(2):
#     axes[i].axis('off')
#     axes[i].tick_params(axis='both',which='both',length=0)

# plt.savefig('demo.png')