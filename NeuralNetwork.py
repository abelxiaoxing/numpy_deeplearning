import numpy as np
from rich.progress import Progress
import utils
import copy 

class MLP:
    def __init__(self, layer_dimensions, parameters):
        # 初始化
        self.weights = {}
        for i in range(len(layer_dimensions) - 1):
            self.weights[i] = np.random.uniform(
                -0.1, 0.1, (layer_dimensions[i], layer_dimensions[i + 1])
            )
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.epochs = parameters["epochs"]
        # 初始化激活函数和损失函数
        activation_name = parameters["activation"]

        if (isinstance(activation_name, str)and activation_name in utils.activation_table):
            self.activation = utils.activation_table[activation_name]
        else:
            self.activation = activation_name
        loss_name = parameters["loss"]
        if isinstance(loss_name, str) and loss_name in utils.loss_table:
            self.loss = utils.loss_table[loss_name]
        else:
            self.loss = loss_name

    def get_weights(self):
        return copy.deepcopy(self.weights)

    def set_weights(self, new_weights):
        self.weights = new_weights

    def feedforward(self, datas):
        # 计算每一层的输入和输出
        self.layer_input = {}
        self.layer_output = {0: datas}
        for i in range(len(self.weights)):
            self.layer_input[i] = np.dot(self.layer_output[i], self.weights[i])
            # 最后一层不加激活函数
            if i == len(self.weights) - 1:
                self.layer_output[i + 1] = self.layer_input[i]
            else:
                self.layer_output[i + 1] = self.activation(self.layer_input[i])[0]
        return self.layer_output[len(self.weights)]

    def backpropagation(self, y_true, y_pred):
        # 计算每一层的梯度
        num_layers = len(self.weights)
        delta = self.loss(y_true, y_pred)[1]
        if num_layers != 1:
            delta *= self.activation(self.layer_input[num_layers - 1])[1]
        gradient_weights = {
            num_layers - 1: np.dot(self.layer_output[num_layers - 1].T, delta)
        }
        for i in reversed(range(num_layers - 1)):
            delta = np.dot(delta, self.weights[i + 1].T)
            if i != 0:
                delta *= self.activation(self.layer_input[i])[1]
            gradient_weights[i] = np.dot(self.layer_output[i].T, delta)
        return gradient_weights


    def train(self, features, labels,epoch):
        with Progress() as progress:
            # 打乱数据集顺序
            indices = np.random.permutation(features.shape[0])
            shuffled_features = features[indices]
            shuffled_labels = labels[indices]
            # 划分 mini-batch
            num_batches = features.shape[0] // self.batch_size
            task = progress.add_task(f"[cyan]Epoch {epoch + 1}/{self.epochs}...", total=num_batches)
            for batch in range(num_batches):
                # 更新进度条
                progress.update(task, advance=1)

                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                features_batch = shuffled_features[start:end]
                labels_batch = shuffled_labels[start:end]
                # 计算前向传播
                outputs = self.feedforward(features_batch)
                # 计算反向传播
                gradient_weights = self.backpropagation(labels_batch, outputs)
                # 更新权重
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.learning_rate * gradient_weights[layer]
        # 计算预测值
        activation_values = features
        for layer in range(len(self.weights)):
            weighted_sum = np.dot(activation_values, self.weights[layer])
            # 最后一层不加激活函数
            if layer == len(self.weights) - 1:
                activation_values = weighted_sum
            else:
                activation_values = self.activation(weighted_sum)[0]
        return activation_values

    def predict(self, features):
        # 计算预测值
        activation_values = features
        for layer in range(len(self.weights)):
            weighted_sum = np.dot(activation_values, self.weights[layer])
            # 最后一层不加激活函数
            if layer == len(self.weights) - 1:
                activation_values = weighted_sum
            else:
                activation_values = self.activation(weighted_sum)[0]
        return activation_values
