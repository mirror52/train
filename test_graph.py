import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.manifold import TSNE
import seaborn as sns
import shap

import matplotlib.pyplot as plt

params = {
    'font.family':'Times New Roman, SimSun',
    # 'font.serif':'Times New Roman',
    'font.weight':'normal', # or 'blod'
    'font.style':'normal',
    'font.size':15, #TODO:修改全局字体大小，但有些内容修改失败需要在代码里手动修改
    'ytick.direction': 'in',
    'xtick.direction': 'in',
    'axes.unicode_minus': False,
    'figure.dpi': 400
}
plt.rcParams.update(params)

#data = pd.read_csv("letterdate.csv")
data = pd.read_excel("data.xlsx")
true_labels = np.unique(data["Class"])
true_labels = ['An I', 'An II', 'An III', 'OB', 'MD']
original_labels = data["Class"]
print(true_labels)
label_encoder = LabelEncoder()
data["Class"] = label_encoder.fit_transform(data['Class'])

data = data.values
labels = np.array(data[:, -1], dtype=int) #(1274,)
data = data[:, :-1] #(1274,) (1274, 8)
print(data.shape)
print(labels.shape)

col = pd.read_excel('对应关系.xlsx', header=None)[0].tolist()
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 准确度评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
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
    #plt.show()

    # ROC曲线
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)#[:, 1]
    else:
        raise AttributeError("The selected model does not have decision_function or predict_proba")

    # 多类别情况下，将标签二值化
    if len(np.unique(labels)) > 2:
        y_test_bin = label_binarize(y_test, classes=np.unique(labels))
    else:
        y_test_bin = y_test

    # 计算每个类别的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(np.unique(labels).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算 macro-average AUC
    macro_auc = np.mean(list(roc_auc.values()))
    print(f"Macro Avg AUC :{macro_auc:.6f}")

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    for i in range(np.unique(labels).shape[0]):
        #TODO:lw 线宽度
        plt.plot(fpr[i], tpr[i], lw=4, label=f'{true_labels[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='white', alpha=0, label=f'Avg AUC = {macro_auc:.4f}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate',  size=18) #TODO:size修改字体大小
    plt.ylabel('True Positive Rate',  size=18)
    #plt.text(0.6, 0.4, f'Macro Avg AUC = {macro_auc:.2f}', bbox=dict(facecolor='white', alpha=0.5))
    plt.title(name + ' ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'./data/ROC Curve_{name}', bbox_inches='tight', dpi=600)
    #plt.show()

    if name == 'XGBoost' or name == 'Random Forest':
        # Create an explainer object    
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X_test)

        # Plot the SHAP value impact graph
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, feature_names=col, class_names=true_labels, show=False)
        plt.title(f'{name}')
        plt.xticks(size=16) #TODO:size修改x轴边框字体大小
        plt.yticks(size=16) #TODO:size修改y轴边框字体大小
        plt.xlabel('mean([SHAP value])', size=18)
        plt.tight_layout()
        plt.savefig(f'./data/shap_summary_plot_{name}', bbox_inches='tight', dpi=600)

        fig = plt.figure(figsize=(10, 6))
        for i in range(len(true_labels)):
            ax = fig.add_subplot(1, len(true_labels), i+1)
            ax.title.set_text(f'{true_labels[i]}')
            plt.yticks(size=16)
            if i == len(true_labels)-1:
                color = True
            else:
                color = False
            shap.summary_plot(shap_values[i], X_test, feature_names=col, show=False, color_bar=color)
            if i != 0:
                ax.set_yticks([])
            ax.set_xlabel(r'SHAP values', fontsize=12)
            mean_shape = np.mean(abs(shap_values[1]), axis=0)
            np.savetxt(f'./data/SHAP_{true_labels[i]}.csv', mean_shape, delimiter=',')
        plt.tight_layout()
        plt.savefig(f'./data/shap_summary_plot_{name}_mutilclass', bbox_inches='tight', dpi=600)


        # Plot the SHAP decisiooooon graph
        plt.figure(figsize=(8, 6))
        if isinstance(explainer.expected_value, np.ndarray):
            shap.multioutput_decision_plot(explainer.expected_value.tolist(), shap_values, feature_order='hclust', feature_names=col,row_index=0, show=False)
        else:
            shap.multioutput_decision_plot(explainer.expected_value, shap_values, feature_order='hclust', feature_names=col, row_index=0, show=False)
        plt.title(f'{name}')
        plt.tight_layout()
        plt.yticks(size=16)
        plt.xlabel('Model output value', size=18)
        plt.savefig(f'./data/shap_decision_plot_{name}', bbox_inches='tight', dpi=600)

# 选择分类器并评估
classifiers = {
    "XGBoost": xgb.XGBClassifier(),
    "SVM": SVC(probability=True),
    "MLP": MLPClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

for name, model in classifiers.items():
    print(f"\n{name}:")
    evaluate_model(name, model, X_train, y_train, X_test, y_test)

for cls in range(5):
    plt.figure(figsize=(8, 6))
    for name, model in classifiers.items():
        y_scores = model.predict_proba(X_test)
        # 计算精确度和召回率
        precision, recall, _ = precision_recall_curve(y_test==cls, y_scores[:,cls])
        
        # 绘制精确度-召回率曲线
        plt.plot(recall, precision, lw=4, label=name)

    # 设置图形的标题和标签
    plt.xlabel('Recall',  size=18)
    plt.ylabel('Precision',  size=18)
    plt.title(f'Precision-Recall Curves(Positive label: {true_labels[cls]})')
    plt.legend(fontsize=16)
    plt.tight_layout()

    # 保存并显示图形
    plt.savefig(f'./data/precision_recall_curves_positive_label_{true_labels[cls]}', bbox_inches='tight', dpi=600)
    #plt.show()

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data)

# 将数据和标签合并成一个 DataFrame
df_tsne = pd.DataFrame(data_tsne, columns=['Dimension 1', 'Dimension 2'])
df_tsne['Label'] = labels  #labels original_labels
label_colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'purple', 'E': 'orange'} #'rainbow'

# 绘制 t-SNE 可视化
plt.figure(figsize=(8, 6))
my_palette = ['#4B4390', '#5E9AB8', '#AF4E86', '#AD6E6B', '#00A0E3']
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Label', data=df_tsne, palette=my_palette, legend='full', alpha=0.7, s=80) #vlag gist_stern
# Adjust legend font size

plt.xlabel('Dimension 1', fontsize=18)
plt.ylabel('Dimension 2', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.title('t-SNE Visualization of Data', fontsize=20)
plt.savefig('./data/t_SNE_Visualization', bbox_inches='tight', dpi=600)
#plt.show()