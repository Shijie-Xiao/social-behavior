# SRNN Model Testing and Usage Guide

## Mouse Dataset: Full Pipeline (Unique Paths)

Paths are unique per `exp_tag` (auto: `obs{N}_pred{M}_fs{frame_subsample}`).

### 1. Training (Multi-GPU)

```bash
cd /pscratch/sd/s/sixao74/MS/socialAttention/social-attention/srnn-pytorch/srnn
ml conda && conda activate sa

# Distributed (e.g. 2 GPUs)
torchrun --nproc_per_node=2 train.py --use_mouse_data \
  --seq_length 20 --pred_length 5 --frame_subsample 4 \
  --batch_size 8 --num_epochs 200

# Output: save/mice/obs15_pred5_fs4/save_attention/
```

### 2. Training (Single GPU)

```bash
python train_single.py --use_mouse_data \
  --seq_length 20 --pred_length 5 --frame_subsample 4 \
  --batch_size 8 --num_epochs 200

# Output: save/mice/obs15_pred5_fs4/save_attention_single/
```

### 3. Testing (sample.py)

```bash
# After distributed training (train.py)
python sample.py --use_mouse_data --from_distributed \
  --exp_tag obs15_pred5_fs4 --obs_length 15 --pred_length 5 \
  --epoch 0 --frame_subsample 4

# After single-GPU training (train_single.py)
python sample.py --use_mouse_data \
  --exp_tag obs15_pred5_fs4 --obs_length 15 --pred_length 5 \
  --epoch 0 --frame_subsample 4

# Output: save/mice/obs15_pred5_fs4/save_attention/results.pkl (or save_attention_single)
```

### 4. Visualization

```bash
# Trajectory plots (after distributed)
python visualize.py --attention --use_mouse_data --from_distributed \
  --exp_tag obs15_pred5_fs4

# Trajectory plots (after single-GPU)
python visualize.py --attention --use_mouse_data \
  --exp_tag obs15_pred5_fs4

# Output: plot/mice/obs15_pred5_fs4/plot_attention/sequence*.png
```

### 5. Custom exp_tag (Multiple Runs)

```bash
# Custom tag for unique run
python train.py --use_mouse_data --exp_tag my_run_001 \
  --seq_length 20 --pred_length 5 --frame_subsample 4

python sample.py --use_mouse_data --from_distributed \
  --exp_tag my_run_001 --epoch 0

python visualize.py --attention --use_mouse_data --from_distributed \
  --exp_tag my_run_001
```

---

## Human Dataset: Testing

### 1. 测试脚本说明

#### **sample.py** - 模型测试脚本
- **功能**：在测试数据集上运行训练好的模型，计算预测误差
- **输出**：
  - 控制台输出：平均位移误差（Mean Error）和最终位移误差（Final Error）
  - `results.pkl`：包含所有测试结果的pickle文件

#### **visualize.py** - 可视化脚本
- **功能**：从`results.pkl`读取结果，绘制真实轨迹和预测轨迹的对比图
- **输出**：轨迹对比图像文件（PNG格式）

#### **attn_visualize.py** - 注意力可视化脚本
- **功能**：可视化模型的注意力权重

### 2. 测试运行命令

#### 步骤1：运行测试（sample.py）

```bash
cd /pscratch/sd/s/sixao74/MS/socialAttention/social-attention/srnn-pytorch

# 测试分布式训练的模型
python srnn/sample.py \
    --test_dataset 3 \
    --epoch 175 \
    --obs_length 8 \
    --pred_length 12

# 参数说明：
# --test_dataset: 测试数据集索引（0-4，对应5个数据集）
# --epoch: 要加载的模型epoch（根据val.txt中的最佳epoch选择）
# --obs_length: 观察轨迹长度（默认8）
# --pred_length: 预测轨迹长度（默认12）
```

**重要提示**：
- `sample.py`第49行硬编码了保存路径为`save_attention`
- 如果测试单卡训练的模型，需要修改`sample.py`第49行为`save_attention_single`
- 或者测试不同模型时，需要手动修改保存路径

#### 步骤2：可视化结果（visualize.py）

```bash
# 可视化测试结果
python srnn/visualize.py \
    --attention \
    --test_dataset 3

# 参数说明：
# --attention: 使用attention模型（对应save_attention路径）
# --test_dataset: 测试数据集索引
```

### 3. 结果文件位置

- **测试结果**：`save/{test_dataset}/save_attention/results.pkl`
- **可视化图像**：`plot/plot_attention/sequence{编号}.png`
- **模型检查点**：`save/{test_dataset}/save_attention/srnn_model_{epoch}.tar`

---

## 二、数据预处理部分

### 1. 数据预处理流程

数据预处理在 `utils.py` 的 `DataLoader` 类中完成：

#### **预处理函数**：`frame_preprocess()` (utils.py 第63-157行)

**输入格式**：
- CSV文件：`{数据目录}/pixel_pos_interpolate.csv`
- CSV文件格式（列顺序）：
  ```
  frameID, pedID, y坐标, x坐标
  ```
  - 第0列：frameID（帧ID）
  - 第1列：pedID（行人ID）
  - 第2列：y坐标
  - 第3列：x坐标

**处理流程**：
1. 读取CSV文件
2. 按帧组织数据（每帧包含所有行人的位置）
3. 划分训练集和验证集（前20%作为验证集）
4. 保存为pickle文件：`./data/trajectories.cpkl`

**输出格式**：
- Pickle文件包含4个元素：
  - `all_frame_data`: 训练帧数据列表
  - `frameList_data`: 帧ID列表
  - `numPeds_data`: 每帧的行人数量
  - `valid_frame_data`: 验证帧数据列表


### 3. 数据预处理触发

数据预处理在 `DataLoader` 初始化时自动触发（如果pickle文件不存在或`forcePreProcess=True`）：

```python
dataloader = DataLoader(batch_size, seq_length, datasets, forcePreProcess=True)
```

