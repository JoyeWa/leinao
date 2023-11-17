import random
import matplotlib.pyplot as plt
import numpy as np

# Step 1: 定义数字0-9的矩阵
patterns = [
    # 0
    [-1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1],
    # 1
    [-1, -1, 1, -1, -1,
     -1, 1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1,
     -1, -1, 1, -1, -1],
    # 2
    [-1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     -1, -1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, 1, -1, -1, -1,
     1, 1,  1,  1,  1],
    # 3
    [1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, 1, 1, 1, -1],
    # 4
    [-1, -1, -1, 1, -1,
     -1, -1, 1, 1, -1,
     -1, 1, -1, 1, -1,
     1, 1, 1, 1, 1,
     -1, -1, -1, 1,- 1,
     -1, -1, -1, 1, -1],
    # 5
    [1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, -1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, -1, 1,
     1, 1, 1, 1, -1],
    # 6
    [-1, -1, 1, 1, 1,
     -1, 1, -1, -1, -1,
     1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1],
    # 7
    [1, 1, 1, 1, 1,
     -1, -1, -1, -1, 1,
     -1, -1, -1, 1, -1,
     -1, -1, 1, -1, -1,
     -1, 1, -1, -1, -1,
     1, -1, -1, -1, -1],
    # 8
    [-1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, -1],
    # 9
    [-1, 1, 1, 1, -1,
     1, -1, -1, -1, 1,
     1, -1, -1, -1, 1,
     -1, 1, 1, 1, 1,
     -1, -1, -1, 1, -1,
     1, 1, 1, -1, -1]

]

# Step 2: 创建Hopfield神经网络
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        self.threshold = 0  # 添加动态阈值

    def hebbian_learn(self, pattern):
        pattern = np.array(pattern)
        self.weights += np.outer(pattern, pattern) # 计算外积
        np.fill_diagonal(self.weights, 0)  # 对角线元素置零

    def storkey_learn(self, pattern):
        pattern = np.array(pattern)
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.weights[i, j] += (
                                pattern[i] * pattern[j] - pattern[i] * np.dot(self.weights[i, :], pattern) - np.dot(
                            self.weights[:, j], pattern) * pattern[j])

        np.fill_diagonal(self.weights, 0)  # 对角线元素置零

    def train(self, patterns):
        for pattern in patterns:
            # self.storkey_learn(pattern)
            self.hebbian_learn((pattern))

    def recall(self, noisy_pattern, max_iter=10):
        recalled_pattern = noisy_pattern.copy()
        for _ in range(max_iter):
            for i in range(len(noisy_pattern)):
                activation = np.dot(self.weights[i], recalled_pattern)
                # recalled_pattern[i] = 1 if activation > self.threshold else -1  # 使用动态阈值
                percentage_threshold = 0.7  # x0% 的神经元激活作为阈值
                recalled_pattern[i] = 1 if activation > percentage_threshold * len(self.weights[i]) else -1

        return recalled_pattern


# Step 3: 产生带噪声的数字点阵（随机噪声）
def add_noise(pattern, noise_percentage=0.1):
    noisy_pattern = pattern.copy()
    num_noise = int(len(pattern) * noise_percentage)
    for _ in range(num_noise):
        index = random.randint(0, len(pattern) - 1)
        noisy_pattern[index] *= -1  # 如果是1则变为-1，如果是-1则变为1
    return noisy_pattern

# Step 4: 数字识别测试
if __name__ == "__main__":
    network_size = 30  # 6*5矩阵展开成一维向量
    hopfield_net = HopfieldNetwork(network_size)
    hopfield_net.train(patterns)

    for i, pattern in enumerate(patterns):
        # 测试每个数字的识别效果
        noisy_pattern = add_noise(pattern)
        recalled_pattern = hopfield_net.recall(noisy_pattern)

        # 绘制图形
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(np.array(pattern).reshape((6, 5)), cmap='gray')
        plt.title(f"Original Pattern {i}")

        plt.subplot(1, 3, 2)
        plt.imshow(np.array(noisy_pattern).reshape((6, 5)), cmap='gray')
        plt.title(f"Noisy Pattern {i}")

        plt.subplot(1, 3, 3)
        plt.imshow(np.array(recalled_pattern).reshape((6, 5)), cmap='gray')
        plt.title(f"Recalled Pattern {i}")

        plt.show()
