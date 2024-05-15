import NeuralNetwork
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import utils
from sklearn.preprocessing import StandardScaler
# 标准化数据集
scaler = StandardScaler()
# 从mnist_784中获取数据，version=1表示版本，parser="auto"表示使用自动解析器
mnist = fetch_openml("mnist_784", version=1, parser="auto")
datas, labels = (mnist["data"] / 255.0).to_numpy(), mnist["target"].astype(int).to_numpy()
# datas, labels = scaler.fit_transform(mnist["data"]), mnist["target"].astype(int).to_numpy()
# 将标签转换为one-hot编码
encoder = LabelBinarizer()
labels_one_hot = encoder.fit_transform(labels)
# 从数据中分割出训练集和测试集，test_size=0.1表示测试集所占比例，random_state=42表示随机种子
datas_train, datas_test, labels_train, labels_test = train_test_split(datas, labels_one_hot, test_size=0.1, random_state=42)
# 打印训练集、测试集的形状
print(f"train datas:{datas_train.shape}, train labels:{labels_train.shape}, test datas:{datas_test.shape}, test labels{labels_test.shape}")
# # 定义隐藏层的维度
layer_dimensions = [datas_train.shape[1], 128, 64, labels_train.shape[1]]
# 定义参数
parameters = {
    "learning_rate": 5e-3,
    "epochs": 10,
    "batch_size": 16,
    "activation": "selu",
    "loss": "CE",
}
best_val_accuracy = 0.0  # 用于跟踪最佳验证精度
best_weights = None  # 用于保存最佳模型权重
# 实例化网络
model = NeuralNetwork.MLP(layer_dimensions, parameters)
for epoch in range(parameters["epochs"]):
    # 训练模型
    train_pred=model.train(datas_train, labels_train,epoch)
    train_correct_predictions = np.argmax(train_pred, axis=1) == np.argmax(labels_train, axis=1)
    train_accuracy = np.mean(train_correct_predictions)
    print(f"Train Accuracy: {train_accuracy*100:.3f}%")
    # 验证
    val_pred = model.predict(datas_test)
    val_correct_predictions = np.argmax(val_pred, axis=1) == np.argmax(labels_test, axis=1)
    val_accuracy = np.mean(val_correct_predictions)
    print(f"Test Accuracy: {val_accuracy*100:.3f}%")
    # 如果当前验证精度更高，则更新最佳精度并保存权重
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        utils.save_weights(model.get_weights(), filename="best_model_weights.npy")
        print(f"New best validation accuracy: {val_accuracy*100:.3f}%, saving model weights.")

# 在训练结束后，可以将最佳权重加载回模型中
best_weights = utils.load_weights("best_model_weights.npy")
model.set_weights(best_weights)  # 假设模型有一个set_weights方法可以设置权重
print(f"Best validation accuracy during training: {best_val_accuracy*100:.3f}%")
