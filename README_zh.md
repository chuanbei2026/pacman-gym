[In English](README.md) | 中文版

# PacMan 深度 Q 网络 (DQN)

一个经典的吃豆人游戏，同时也是一个可上手实践的 Deep Q-Network (DQN) 学习环境。你可以亲自玩游戏、训练 DQN 智能体，或者实时观看智能体的学习过程及网络内部可视化。

<p align="center">
  <img src="assets/demo.gif" alt="吃豆人教学模式：实时 DQN 训练可视化">
</p>

## 为什么做这个项目？

大多数 DQN 教程止步于理论或者使用过于简单的环境。这个项目让你能够**直观看到** DQN 在真实游戏上的学习过程：

- 观察 Q 值随智能体发现策略而变化
- 查看权重热图在训练中的演变
- 观察探索率（epsilon）如何从探索衰减到利用
- 通过调整奖励塑形来理解其效果

## 快速开始

```bash
git clone https://github.com/chuanbei2026/pacman-gym.git
cd pacman-gym

# 安装包
pip install -e .

# 自己玩游戏
python3 -m pacman_gym

# 训练 DQN 智能体（无界面）
python3 -m pacman_gym train

# 观看训练好的智能体玩游戏
python3 -m pacman_gym play

# 实时训练可视化（推荐）
python3 -m pacman_gym teach
```

## 模式

### 手动游玩 (`python3 -m pacman_gym`)

经典吃豆人，方向键控制。吃掉所有豆子即可获胜，躲避幽灵，抓取能量豆获得无敌状态。

### 教学模式 (`python3 -m pacman_gym teach`)

核心功能。打开一个分屏窗口：

- **左侧**：实时运行的游戏
- **右侧**：研究风格的面板，显示：
  - **Q 值**：智能体对每个方向的估计价值（argmax 策略）
  - **权重热图**：各层权重的实时可视化
  - **损失曲线**：训练损失随时间的变化
  - **统计表格**：回合数、分数、epsilon、缓冲区大小等
- **底部**：速度滑块，控制训练速度（1x 到 360x）

训练异步运行——后台线程更新网络，主线程处理数据采集和渲染，两者互不阻塞。

### 训练模式 (`python3 -m pacman_gym train`)

无界面训练，速度最快。自动保存最佳模型检查点。

```bash
python3 -m pacman_gym train --episodes 5000
```

### 智能体回放 (`python3 -m pacman_gym play`)

加载训练好的模型，观看智能体玩游戏。按空格键重新开始，ESC 退出。

## DQN 原理

### 背景

**Deep Q-Network (DQN)** 是 DeepMind 在 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)（Mnih et al., 2013）中提出、并在 [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)（Mnih et al., 2015）中进一步完善的强化学习算法。核心思想：

1. **Q-Learning** 估计在状态 *s* 下执行动作 *a* 的预期未来奖励：Q(s, a)。智能体总是选择 Q 值最高的动作（贪心策略）。

2. **问题所在**：在复杂环境中，无法用表格存储每个状态的 Q 值——状态太多了。DQN 通过使用**神经网络**来近似 Q(s, a) 来解决这个问题。

3. **经验回放**：智能体将过去的转移 (s, a, r, s') 存储在缓冲区中，并在随机小批量上训练。这打破了连续样本之间的相关性，提高了稳定性。

4. **目标网络**：使用一个单独的、缓慢更新的 Q 网络副本来计算 TD 目标。这防止了"移动目标"问题——即网络追逐自己不断变化的预测。

本项目实现了 **Double DQN**（[van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)），通过使用策略网络*选择*动作、目标网络*评估*动作来减少 Q 值的过高估计。

### 观测空间（258 维特征）

智能体看到的不是原始像素，而是紧凑的特征向量：

| 特征 | 维度 | 描述 |
|------|------|------|
| 局部网格 | 225 | 吃豆人周围 15x15 的区域（墙壁、豆子、幽灵按空间编码） |
| 幽灵信息 | 24 | 每个幽灵的相对位置、距离、方向、活跃/眩晕状态 |
| 吃豆人方向 | 1 | 当前移动方向 |
| 无敌状态 | 1 | 能量豆是否激活 |
| 豆子扫描 | 4 | 每个方向上最近豆子的距离（通道扫描） |
| 最近豆子 | 2 | 全局最近豆子的相对位置 |
| 剩余豆子 | 1 | 剩余豆子比例 |

### 网络架构

```
输入 (258) -> Linear(512) -> ReLU -> Linear(256) -> ReLU -> Linear(64) -> ReLU -> Linear(4)
```

输出 4 个 Q 值，对应每个方向（上、右、下、左）。智能体选择 Q 值最高的方向。

### 奖励塑形

| 事件 | 奖励 |
|------|------|
| 吃豆子 | +10 |
| 吃能量豆 | +50 |
| 吃苹果 | +100 |
| 吃幽灵（无敌时） | +200 |
| 获胜（吃完所有豆子） | +500 |
| 被幽灵吃掉 | -1000 |
| 掉头 | -5 |
| 访问新格子 | +2 |
| 每步（时间压力） | -(0.5 + step/200) |

### 训练细节

- **算法**：Double DQN + 经验回放
- **异步训练**：数据采集使用冻结的策略副本；后台线程训练实时网络
- **迷宫随机化**：每回合从 5 种迷宫变体中随机选择，提升泛化能力
- **Epsilon 衰减**：在 80k 步内从 1.0 线性衰减到 0.05
- **回放缓冲区**：200,000 条转移
- **批量大小**：128
- **目标网络同步**：每 500 个训练步

## 项目结构

```
pacman-gym/
  pyproject.toml
  README.md
  LICENSE
  src/
    pacman_gym/
      __main__.py   # 入口和 CLI 路由
      game.py       # 核心游戏引擎（纯逻辑，无渲染）
      main.py       # Pygame 手动游玩界面
      gym_env.py    # Gymnasium 环境封装
      train.py      # DQN 智能体、训练循环、回放缓冲区
      teach.py      # 实时训练可视化
```

## 参考文献

- Mnih et al., [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), 2013
- Mnih et al., [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), 2015
- van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), 2015

## 许可证

MIT
