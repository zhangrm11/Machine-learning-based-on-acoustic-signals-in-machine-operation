import os
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import skew, kurtosis
import random
from collections import defaultdict
from sklearn.metrics import r2_score

# 决策树节点类
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value  

# 决策树实现
class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 终止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            return DecisionTreeNode(value=np.mean(y))
        
        # 随机选择特征子集
        n_features_subset = int(np.sqrt(n_features))
        feature_indices = random.sample(range(n_features), n_features_subset)
        
        # 找到最佳分裂
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)
        
        # 分裂数据集
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        
        # 递归构建子树
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _find_best_split(self, X, y, feature_indices):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                # 计算均方误差(MSE)
                mse = self._calculate_mse(y[left_indices], y[right_indices])
                
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_mse(self, left_y, right_y):
        # 计算左右子集的MSE
        left_mean = np.mean(left_y) if len(left_y) > 0 else 0
        right_mean = np.mean(right_y) if len(right_y) > 0 else 0
        
        left_mse = np.mean((left_y - left_mean) ** 2)
        right_mse = np.mean((right_y - right_mean) ** 2)
        
        return (len(left_y) * left_mse + len(right_y) * right_mse) / (len(left_y) + len(right_y))
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

# 随机森林实现
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators  
        self.max_depth = max_depth        
        self.min_samples_split = min_samples_split  
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        
        # 创建多棵决策树
        for _ in range(self.n_estimators):
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

# 统计特征提取，用于描述信号的分布特性
def extract_features(signal):
    return [
        np.mean(signal),    #均值
        np.std(signal),     #标准差
        np.min(signal),     #最小值
        np.max(signal),     #最大值
        skew(signal),       #偏度
        kurtosis(signal)    #峰度
    ]

# 制造参数表（A6、C5为None）
param_table = {
    'A1': [350, 1200], 'A2': [400, 600], 'A3': [250, 600], 'A4': [250, 1000], 'A5': [400, 1000],
    'A6': [None, None], 'A7': [350, 700], 'A8': [200, 500], 'A9': [350, 500], 'A10': [200, 600],
    'A11': [200, 700], 'A12': [250, 500], 'A13': [400, 1200],
    'C1': [150, 800], 'C2': [200, 1600], 'C3': [200, 2000], 'C4': [250, 400],
    'C5': [None, None], 'C6': [250, 2000], 'C7': [300, 1200], 'C8': [300, 2000], 'C9': [330, 1600],
    'C10': [330, 1050], 'C11': [350, 1600], 'C12': [350, 1600], 'C13': [375, 700]
}

#数据加载与特征处理
# os.path.dirname用于获取当前文件所在的目录；os.path.abspath(__file__)用于获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# os.path.join安全拼接多个路径组件
data_dir = os.path.join(current_dir, 'work_data')
features = []
labels = []
names = []

# 遍历文件，提取特征
# os.listdir返回指定目录下所有文件的名称列表，例如[A1.xlsx, A2.xlsx, ...]
for i in os.listdir(data_dir):
    if i.endswith('.xlsx'):
        name = i.split('.')[0]
        df = pd.read_excel(os.path.join(data_dir, i), header=None)
        signal = df.iloc[:, 0].values
        feat = extract_features(signal)
        features.append(feat)
        labels.append(param_table[name])
        names.append(name)

features = np.array(features)
labels = np.array(labels)

# 划分训练/测试集
train_idx = [i for i, n in enumerate(names) if n not in ['A6', 'C5']]
test_idx = [i for i, n in enumerate(names) if n in ['A6', 'C5']]

X_train, y_train = features[train_idx], labels[train_idx]
X_test = features[test_idx]

# RandomForestRegressor是随机森林算法回归器，42是种子；MultiOutputRegressor是多输出回归包装器，同时预测多个目标变量
reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, 
                                                max_depth=5, random_state=42))
# 开始训练，使用训练数据拟合模型，模型会学习特征与目标参数之间的映射关系
reg.fit(X_train, y_train)

# 在训练集上进行预测并计算准确度
train_pred = reg.predict(X_train)
train_score = r2_score(y_train, train_pred)
print(f"训练集R^2准确度: {train_score:.4f}")

# 使用训练好的模型对测试数据进行预测
y_pred = reg.predict(X_test)

print(f"A6预测: 激光功率={y_pred[0,0]:.1f} W, 扫描速度={y_pred[0,1]:.1f} mm/s")
print(f"C5预测: 激光功率={y_pred[1,0]:.1f} W, 扫描速度={y_pred[1,1]:.1f} mm/s")
