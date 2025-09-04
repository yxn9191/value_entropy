# Value Entropy - 价值熵系统 / Value Entropy System

[English](#english) | [中文](#chinese)

---

## English

### Overview

Value Entropy is a comprehensive multi-agent simulation system for cloud manufacturing networks that studies the relationship between agent intelligence levels, system entropy, and value creation. The system implements reinforcement learning algorithms to optimize agent behavior and system performance, providing two distinct simulation environments for different research scenarios.

### System Architecture

The project contains two main simulation environments:

#### 1. CloudManufacturing (Geographic-based)
- **Environment**: Geographic space simulation with real-world coordinates
- **Government Regulation**: Tax-based regulation system with redistribution mechanisms
- **Agent Types**: Service agents (enterprises) and order agents
- **Intelligence Levels**: Low, Medium, High (reinforcement learning)
- **Key Features**:
  - Fixed tax rates (15% in Phase1, 25% in Phase2)
  - Tax collection and redistribution every 10 steps
  - Geographic distance-based collaboration
  - Static enterprise population

#### 2. CloudManufacturing_network (Network-based)
- **Environment**: Network topology simulation with various graph structures
- **Government Regulation**: No government intervention (pure market dynamics)
- **Agent Types**: Service agents (enterprises) and order agents
- **Intelligence Levels**: Low, Medium, Imitation, High (reinforcement learning)
- **Key Features**:
  - Multiple network topologies (ER, BA, WS, RG, None)
  - Dynamic enterprise evolution (death and rebirth)
  - Network distance-based collaboration
  - Imitation learning mechanism

### Features

- **Multi-Agent Simulation**: Cloud manufacturing environment with service agents and order agents
- **Dual Environment Support**: Geographic and network-based simulation environments
- **Intelligence Levels**: Multiple agent intelligence types (low, medium, high, imitate)
- **Reinforcement Learning**: RLlib-based training with A2C algorithm
- **Real-time Visualization**: Mesa-based web interface for simulation monitoring
- **Government Regulation**: Tax-based policy simulation (in geographic environment)
- **Network Topology**: Support for various network structures (ER, BA, WS, RG)
- **Metrics Tracking**: Comprehensive metrics including:
  - **ht**: System entropy (H_t)
  - **sys_utility**: System utility
  - **avg_agent_utility**: Average agent utility
  - **gini**: Gini coefficient for inequality measurement
  - **value_entropy**: Value entropy

### Installation

#### Prerequisites

- Python 3.10 (required for compatibility)
- Conda (recommended)
- CUDA 11.8 (optional, for GPU support)

#### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd value_entropy
   ```

2. **Create and activate conda environment with Python 3.10**
   ```bash
   conda create -n value-entropy python=3.10 -y
   conda activate value-entropy
   ```

3. **Install PyTorch with correct version**
   ```bash
   # For CUDA 11.8 support (GPU)
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install remaining dependencies**
   ```bash
   pip install -r value-entropy_requirements.txt
   ```

5. **Set PYTHONPATH**
   ```bash
   conda env config vars set PYTHONPATH=/path/to/value_entropy
   conda deactivate && conda activate value-entropy
   ```

#### Troubleshooting

**Common Issues:**
- **torch version error**: Use the official PyTorch index-url as shown above
- **Python version mismatch**: Ensure Python 3.10 is used
- **mesa/gym compatibility**: Install dependencies after PyTorch to avoid conflicts

### Usage

#### CloudManufacturing (Geographic-based with Government Regulation)

1. **Training Phase 1 (Basic Training)**
   ```bash
   cd example/cloudManufacturing/rl
   python doTrain.py --run-dir phase1
   ```

2. **Training Phase 2 (Advanced Training)**
   ```bash
   cd example/cloudManufacturing/rl
   python doTrain.py --run-dir phase2
   ```

3. **Run Visualization**
   ```bash
   cd example/cloudManufacturing
   python run.py --run-dir phase1  # or phase2
   ```
   Open browser at: http://localhost:8523

#### CloudManufacturing_network (Network-based without Government)

1. **Run Training**
   ```bash
   cd example/cloudManufacturing_network/rl
   python doTrain.py
   ```

2. **Start Visualization Server**
   ```bash
   cd example/cloudManufacturing_network
   python server.py
   ```
   Open browser at: http://localhost:8523

#### Configuration

**CloudManufacturing Configuration:**
Edit `example/cloudManufacturing/rl/phase1/config.yaml` or `phase2/config.yaml`:
- Tax rates (15% vs 25%)
- Number of orders and services
- Training parameters
- Episode length

**CloudManufacturing_network Configuration:**
Edit `example/cloudManufacturing_network/rl/phase/config.yaml`:
- Network parameters (node degree, number of agents)
- Network topology type (ER, BA, WS, RG, None)
- Training parameters (batch size, learning rate)
- Agent intelligence ratios
- Episode length

### Project Structure

```
value_entropy/
├── algorithm/          # RL algorithms and wrappers
├── base/              # Core utilities and reward functions
├── example/           # Simulation examples
│   ├── cloudManufacturing/           # Geographic-based simulation
│   │   ├── rl/                      # Training scripts
│   │   │   ├── phase1/              # Phase 1 training config
│   │   │   └── phase2/              # Phase 2 training config
│   │   ├── run.py                   # Visualization server
│   │   └── env.py                   # Environment definition
│   └── cloudManufacturing_network/  # Network-based simulation
│       ├── rl/                      # Training scripts
│       ├── server.py                # Visualization server
│       └── env.py                   # Environment definition
├── analysis/          # Analysis tools
└── README.md
```

### Key Metrics

- **System Entropy (ht)**: Measures system complexity and diversity
- **System Utility**: Overall system performance
- **Average Agent Utility**: Mean performance across all agents
- **Gini Coefficient**: Inequality measure among agents
- **Value Entropy**: Value creation efficiency

### Research Applications

- **Policy Analysis**: Study the impact of government regulation on cloud manufacturing
- **Network Effects**: Analyze how network topology affects agent collaboration
- **Intelligence Evolution**: Research the relationship between agent intelligence and system performance
- **Value Creation**: Understand the dynamics of value generation in multi-agent systems

---

## Chinese

### 概述

价值熵系统是一个综合性的云制造网络多智能体仿真系统，研究智能体智能水平、系统熵与价值创造之间的关系。系统实现了强化学习算法来优化智能体行为和系统性能，提供两种不同的仿真环境以适应不同的研究场景。

### 系统架构

项目包含两个主要的仿真环境：

#### 1. CloudManufacturing（基于地理空间）
- **环境**: 基于真实地理坐标的空间仿真
- **政府调控**: 基于税收的调控系统，包含再分配机制
- **智能体类型**: 服务智能体（企业）和订单智能体
- **智能水平**: 低、中、高（强化学习）
- **核心特点**:
  - 固定税率（Phase1为15%，Phase2为25%）
  - 每10步征税并再分配
  - 基于地理距离的协作
  - 静态企业数量

#### 2. CloudManufacturing_network（基于网络拓扑）
- **环境**: 基于网络拓扑的仿真，支持多种图结构
- **政府调控**: 无政府干预（纯市场动态）
- **智能体类型**: 服务智能体（企业）和订单智能体
- **智能水平**: 低、中、模仿、高（强化学习）
- **核心特点**:
  - 多种网络拓扑（ER、BA、WS、RG、None）
  - 动态企业进化（死亡和重生）
  - 基于网络距离的协作
  - 模仿学习机制

### 功能特点

- **多智能体仿真**: 包含服务智能体和订单智能体的云制造环境
- **双环境支持**: 地理空间和网络拓扑两种仿真环境
- **智能水平**: 多种智能体智能类型（低、中、高、模仿）
- **强化学习**: 基于RLlib的A2C算法训练
- **实时可视化**: 基于Mesa的Web界面进行仿真监控
- **政府调控**: 基于税收的政策仿真（地理环境）
- **网络拓扑**: 支持多种网络结构（ER、BA、WS、RG）
- **指标跟踪**: 全面的指标监测，包括：
  - **ht**: 系统熵值 (H_t)
  - **sys_utility**: 系统效能
  - **avg_agent_utility**: 平均个体效能
  - **gini**: 基尼系数（不平等性测量）
  - **value_entropy**: 价值熵

### 安装方法

#### 环境要求

- Python 3.10（必需，用于兼容性）
- Conda（推荐）
- CUDA 11.8（可选，用于GPU支持）

#### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd value_entropy
   ```

2. **创建并激活Python 3.10的conda环境**
   ```bash
   conda create -n value-entropy python=3.10 -y
   conda activate value-entropy
   ```

3. **安装正确版本的PyTorch**
   ```bash
   # 支持CUDA 11.8（GPU）
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   
   # 仅CPU版本
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   ```

4. **安装其余依赖包**
   ```bash
   pip install -r value-entropy_requirements.txt
   ```

5. **设置PYTHONPATH**
   ```bash
   conda env config vars set PYTHONPATH=/path/to/value_entropy
   conda deactivate && conda activate value-entropy
   ```

#### 故障排除

**常见问题：**
- **torch版本错误**: 使用上述官方PyTorch index-url
- **Python版本不匹配**: 确保使用Python 3.10
- **mesa/gym兼容性**: 在PyTorch之后安装依赖以避免冲突

### 使用方法

#### CloudManufacturing（基于地理空间，有政府调控）

1. **训练阶段1（基础训练）**
   ```bash
   cd example/cloudManufacturing/rl
   python doTrain.py --run-dir phase1
   ```

2. **训练阶段2（高级训练）**
   ```bash
   cd example/cloudManufacturing/rl
   python doTrain.py --run-dir phase2
   ```

3. **运行可视化**
   ```bash
   cd example/cloudManufacturing
   python run.py --run-dir phase1  # 或 phase2
   ```
   在浏览器中打开: http://localhost:8523

#### CloudManufacturing_network（基于网络拓扑，无政府调控）

1. **运行训练**
   ```bash
   cd example/cloudManufacturing_network/rl
   python doTrain.py
   ```

2. **启动可视化服务器**
   ```bash
   cd example/cloudManufacturing_network
   python server.py
   ```
   在浏览器中打开: http://localhost:8523

#### 配置参数

**CloudManufacturing配置:**
编辑 `example/cloudManufacturing/rl/phase1/config.yaml` 或 `phase2/config.yaml`:
- 税率设置（15% vs 25%）
- 订单和服务数量
- 训练参数
- 回合长度

**CloudManufacturing_network配置:**
编辑 `example/cloudManufacturing_network/rl/phase/config.yaml`:
- 网络参数（节点度、智能体数量）
- 网络拓扑类型（ER、BA、WS、RG、None）
- 训练参数（批次大小、学习率）
- 智能体智能比例
- 回合长度

### 项目结构

```
value_entropy/
├── algorithm/          # 强化学习算法和包装器
├── base/              # 核心工具和奖励函数
├── example/           # 仿真示例
│   ├── cloudManufacturing/           # 基于地理空间的仿真
│   │   ├── rl/                      # 训练脚本
│   │   │   ├── phase1/              # 阶段1训练配置
│   │   │   └── phase2/              # 阶段2训练配置
│   │   ├── run.py                   # 可视化服务器
│   │   └── env.py                   # 环境定义
│   └── cloudManufacturing_network/  # 基于网络拓扑的仿真
│       ├── rl/                      # 训练脚本
│       ├── server.py                # 可视化服务器
│       └── env.py                   # 环境定义
├── analysis/          # 分析工具
└── README.md
```

### 核心指标

- **系统熵值 (ht)**: 测量系统复杂性和多样性
- **系统效能**: 整体系统性能
- **平均个体效能**: 所有智能体的平均性能
- **基尼系数**: 智能体间的不平等性测量
- **价值熵**: 价值创造效率

### 技术特点

- **生态位分析**: 基于智能体智能水平的生态位划分
- **价值熵计算**: 结合系统熵和个体效能的综合指标
- **网络拓扑**: 支持多种网络结构（ER、BA、WS等）
- **实时监控**: 动态显示系统状态和性能指标
- **政府调控**: 税收再分配机制研究
- **动态进化**: 企业死亡重生机制

### 应用场景

- **政策分析**: 研究政府调控对云制造系统的影响
- **网络效应**: 分析网络拓扑对智能体协作的影响
- **智能进化**: 研究智能体智能水平与系统性能的关系
- **价值创造**: 理解多智能体系统中的价值生成动态
- **云制造优化**: 云制造系统的协作优化研究
- **多智能体协作**: 多智能体协作行为研究
