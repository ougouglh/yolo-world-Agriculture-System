# 🌾 智慧农业病虫害诊断系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

**基于 YOLO-World + RAG + LLM 的智能植物病害检测与诊断系统**

[功能特性](#-功能特性) • [在线演示](#-在线演示) • [快速开始](#-快速开始) • [技术架构](#-技术架构) • [效果展示](#-效果展示)

</div>

---

## 📋 项目简介

本项目是一个端到端的智慧农业病虫害诊断系统，结合了最新的计算机视觉技术和大语言模型，能够：

- 🔍 **零样本检测**：识别训练时从未见过的新发病害（准确率 71.26%）
- 📊 **智能诊断**：基于专业知识库生成详细的诊断报告
- 💊 **防治建议**：提供针对性的农业和化学防治方案
- 🌐 **友好界面**：简洁直观的 Web 操作界面
- 💰 **低成本**：单次诊断成本不到 0.002 元

---

## ✨ 功能特性

### 核心功能

- ✅ **开放词汇检测**：基于 YOLO-World，突破传统模型的类别限制
- ✅ **知识增强生成**：RAG 技术结合 25 种病害专业知识库
- ✅ **多维度分析**：病原、症状、发病条件、危害程度全面分析
- ✅ **实用建议**：农业措施、化学防治、用药安全等详细指导
- ✅ **快速响应**：5-10 秒完成从检测到报告生成的全流程

### 技术亮点

| 特性 | 说明 | 指标 |
|------|------|------|
| 🎯 **零样本检测** | 可识别未见过的新病害 | 71.26% mAP50 |
| 🚀 **高准确率** | 已知病害检测准确率 | 98%+ |
| 💡 **专业诊断** | 基于知识库的精准分析 | 95 个知识文档 |
| ⚡ **快速推理** | 端到端诊断时间 | 5-10 秒 |
| 💰 **低成本** | 单次诊断成本 | < 0.002 元 |

---

## 🎬 在线演示

> 🔗 **演示地址**：[https://your-demo-link.com](https://your-demo-link.com)
> 
> 📹 **演示视频**：[YouTube](https://youtube.com) / [Bilibili](https://bilibili.com)

### 快速体验

```bash
# 克隆项目
git clone https://github.com/your-username/plant-disease-diagnosis.git
cd plant-disease-diagnosis

# 安装依赖
pip install -r requirements.txt

# 下载模型（见下方说明）

# 启动应用
python app/gradio_app.py
```

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.12+ 
- **CUDA**: 12.4+ (推荐 GPU 运行)
- **内存**: 16GB+
- **磁盘**: 5GB+

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

<details>
<summary>点击查看完整依赖列表</summary>

```
ultralytics>=8.3.246
torch>=2.5.1
torchvision>=0.16.1
chromadb
sentence-transformers
dashscope
gradio>=6.0
langchain-community
pillow
numpy
```

</details>

### 2️⃣ 下载模型权重

**选项 A：从 HuggingFace 下载** (推荐)

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/plant-disease-yoloworld",
    filename="best.pt",
    local_dir="./runs/yoloworld/train/weights/"
)
```

**选项 B：从网盘下载**

- 🔗 百度网盘：[下载链接](https://pan.baidu.com) | 提取码：`xxxx`
- 🔗 Google Drive：[下载链接](https://drive.google.com)
- 📦 解压到：`./runs/yoloworld/train/weights/best.pt`

### 3️⃣ 配置 API Key

获取阿里云百炼 API Key：[https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/)

```bash
# 方法 1：环境变量（推荐）
export DASHSCOPE_API_KEY="your-api-key"

# 方法 2：修改配置文件
# 编辑 src/rag/llm_generator.py，修改 api_key 参数
```

### 4️⃣ 构建向量数据库

```bash
python build_vectorstore.py \
    --json_path knowledge_base/disease_knowledge_base.json \
    --persist_dir ./vectorstore/chroma_db
```

### 5️⃣ 启动应用

```bash
python app/gradio_app.py
```

浏览器访问：`http://localhost:6006`

---

## 🏗️ 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web Interface                      │
│            (Image Upload → Detection → Report)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Diagnosis Pipeline                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  YOLO-World  │→→│  RAG         │→→│  LLM Generator   │  │
│  │  Detection   │  │  Retrieval   │  │  (Qwen-Turbo)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ YOLO Model   │    │  ChromaDB    │    │ Alibaba Cloud│
│ (Fine-tuned) │    │  Vector DB   │    │ Qwen API     │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 核心技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 🔍 **目标检测** | YOLO-World (Ultralytics) | 开放词汇检测，支持零样本 |
| 🗄️ **向量数据库** | ChromaDB | 高效的语义检索 |
| 🧠 **Embedding** | BGE-base-zh-v1.5 | 中文语义向量化 |
| 💬 **大语言模型** | 通义千问 Qwen-Turbo | 智能诊断报告生成 |
| 🌐 **Web 框架** | Gradio 6.0 | 快速构建 ML 应用界面 |
| 🔗 **编排框架** | LangChain | RAG 流程编排 |

---

## 📊 效果展示

### 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **零样本检测 mAP50** | 71.26% | 未见过的病害检测准确率 |
| **已知病害检测** | 98%+ | 训练集病害检测准确率 |
| **检索准确率** | 0.75-0.85 | 知识库相似度得分 |
| **响应时间** | 5-10 秒 | 完整诊断流程耗时 |
| **成本** | 0.001-0.002 元 | 单次诊断 API 成本 |

### 零样本检测对比

| 模型 | 训练集 mAP50 | 零样本 mAP50 | 零样本能力 |
|------|-------------|-------------|-----------|
| YOLOv8 (Baseline) | 99.50% | **0%** | ❌ 无法检测 |
| **YOLO-World (Ours)** | 99.50% | **71.26%** | ✅ 强大 |

### 诊断报告示例

<details>
<summary>点击查看完整诊断报告</summary>

```markdown
## 🔍 检测结果概览
检测到番茄植株上存在**番茄早疫病**，置信度 98.6%。根据叶片症状观察，
病斑呈现褐色同心轮纹状，表明病情已进入中等阶段，需尽快采取防治措施。

## 📋 病害详细分析

### 病害：番茄早疫病
- **病原**：茄链格孢菌 (Alternaria solani)
- **主要症状**：叶片上出现圆形或近圆形的褐色病斑，具有明显的同心轮纹...
- **发病条件**：温暖潮湿环境（20-30°C），相对湿度高于80%...
- **危害程度**：影响光合作用，导致植株生长不良，严重时可造成减产...

## 🛡️ 防治建议

### 1. 农业防治措施
- 选用抗病品种，实行3年以上轮作
- 及时清除病残体，保持田间通风
- 合理施肥，增强植株抗病力

### 2. 化学防治方案
- 75%百菌清可湿性粉剂 600倍液，每7-10天喷施一次
- 70%代森锰锌可湿性粉剂 500倍液，连续2-3次
...
```

</details>

---

## 📁 项目结构

```
plant-disease-diagnosis/
├── src/                          # 源代码
│   ├── rag/                      # RAG 模块
│   │   ├── retriever.py         # 知识检索器
│   │   └── llm_generator.py     # LLM 生成器
│   └── pipeline/                 # 流程整合
│       └── diagnosis_pipeline.py # 完整诊断流程
├── app/                          # Web 应用
│   └── gradio_app.py            # Gradio 界面
├── knowledge_base/               # 知识库
│   └── disease_knowledge_base.json
├── runs/                         # 模型权重（需下载）
│   └── yoloworld/train/weights/best.pt
├── vectorstore/                  # 向量数据库（自动生成）
├── build_vectorstore.py         # 向量化脚本
├── requirements.txt             # 依赖列表
└── README.md                    # 本文件
```

---

## 🔬 数据集

### PlantVillage Dataset

- **来源**：[Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **规模**：31,397 张高清图像
- **类别**：25 种植物病害
- **作物**：番茄、马铃薯、辣椒等

### 数据划分

| 数据集 | 类别数 | 图像数 | 用途 |
|--------|--------|--------|------|
| 训练集 | 20 类 | 18,657 | 模型训练 |
| 验证集 | 20 类 | 4,674 | 模型验证 |
| 零样本测试 | 5 类 | 8,066 | 零样本能力测试 |

**零样本测试类别**（训练时完全未见）：
1. Tomato Early Blight（番茄早疫病）
2. Tomato Late Blight（番茄晚疫病）
3. Tomato Septoria Leaf Spot（番茄斑枯病）
4. Potato Early Blight（马铃薯早疫病）
5. Tomato Yellow Leaf Curl Virus（番茄黄化曲叶病毒病）

---

## 🎓 使用指南

### 基本使用

1. **上传图像**：拖拽或点击上传病害图像
2. **自动检测**：系统自动识别病害类型和位置
3. **查看报告**：阅读详细的诊断报告和防治建议
4. **导出结果**：复制报告文本或保存标注图像

### 高级使用

#### 批量诊断

```python
from src.pipeline.diagnosis_pipeline import PlantDiseaseDiagnosisPipeline

# 初始化流程
pipeline = PlantDiseaseDiagnosisPipeline(
    yolo_model_path="./runs/yoloworld/train/weights/best.pt",
    vectorstore_path="./vectorstore/chroma_db",
    api_key="your-api-key"
)

# 批量处理
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img_path in image_paths:
    result = pipeline.diagnose(img_path)
    print(result['diagnosis_report']['report'])
```

#### 自定义知识库

```python
# 添加新的病害知识
import json

with open('knowledge_base/disease_knowledge_base.json', 'r') as f:
    knowledge = json.load(f)

# 添加新病害
new_disease = {
    "id": 26,
    "name_cn": "新病害名称",
    "name_en": "New Disease Name",
    "crop": "作物名称",
    "pathogen": "病原体",
    "symptoms": "症状描述",
    # ...
}
knowledge.append(new_disease)

# 保存并重新构建向量库
with open('knowledge_base/disease_knowledge_base.json', 'w') as f:
    json.dump(knowledge, f, ensure_ascii=False, indent=2)

# 重新构建向量库
!python build_vectorstore.py
```

---

## 🛠️ 开发指南

### 测试

```bash
# 测试 RAG 检索器
python src/rag/retriever.py

# 测试 LLM 生成器
python src/rag/llm_generator.py

# 测试完整 Pipeline
python src/pipeline/diagnosis_pipeline.py
```

### 环境变量

```bash
# 必需
export DASHSCOPE_API_KEY="your-api-key"

# 可选（加速模型加载）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
```

### 自定义配置

编辑 `src/pipeline/diagnosis_pipeline.py`：

```python
pipeline = PlantDiseaseDiagnosisPipeline(
    yolo_model_path="your/model/path",
    vectorstore_path="your/vectorstore/path",
    api_key="your-api-key",
    llm_model="qwen-turbo",  # 可选：qwen-plus, qwen-max
    confidence_threshold=0.25  # 检测置信度阈值
)
```

---

## 🐛 常见问题

<details>
<summary><b>Q1: 如何解决模型下载慢的问题？</b></summary>

**A:** 使用镜像源加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或直接从网盘下载预训练权重。
</details>

<details>
<summary><b>Q2: 为什么显示 "未检测到病害"？</b></summary>

**A:** 可能原因：
- 图像质量不佳（模糊、光线不足）
- 病害不明显
- 不在系统支持的病害类型范围内

建议：
- 使用清晰的高分辨率图像
- 确保病害特征明显
- 查看支持的病害列表
</details>

<details>
<summary><b>Q3: API 调用失败怎么办？</b></summary>

**A:** 检查步骤：
1. 确认 API Key 是否正确
2. 检查网络连接
3. 查看 API 余额是否充足
4. 查看错误日志获取详细信息
</details>

<details>
<summary><b>Q4: 如何提高检测准确率？</b></summary>

**A:** 优化建议：
- 使用高质量、清晰的图像
- 确保病害区域在图像中心
- 避免复杂背景干扰
- 可以调整 `confidence_threshold` 参数
</details>

---

## 📈 路线图

### v1.0 ✅ (当前版本)
- ✅ YOLO-World 零样本检测
- ✅ RAG 知识检索
- ✅ LLM 诊断报告生成
- ✅ Gradio Web 界面

### v1.1 🚧 (计划中)
- [ ] 支持更多病害类别（50+）
- [ ] 批量诊断功能
- [ ] 历史记录管理
- [ ] PDF 报告导出

### v2.0 💡 (未来)
- [ ] 移动端应用（小程序/App）
- [ ] 视频流实时检测
- [ ] 多语言支持
- [ ] 用户反馈系统

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 编码规范
- 添加适当的注释和文档
- 编写单元测试
- 确保代码通过所有测试

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 📧 联系方式

- **作者**：Your Name
- **邮箱**：your.email@example.com
- **GitHub**：[@your-username](https://github.com/your-username)
- **项目主页**：[https://github.com/your-username/plant-disease-diagnosis](https://github.com/your-username/plant-disease-diagnosis)

---

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO-World 实现
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [Alibaba Cloud](https://www.alibabacloud.com/) - 通义千问 API
- [Gradio](https://gradio.app/) - Web 界面框架
- [PlantVillage](https://plantvillage.psu.edu/) - 数据集来源

---

## ⭐ Star History

如果这个项目对你有帮助，请给个 Star ⭐️！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/plant-disease-diagnosis&type=Date)](https://star-history.com/#your-username/plant-disease-diagnosis&Date)

---

## 📊 项目统计

![GitHub stars](https://img.shields.io/github/stars/your-username/plant-disease-diagnosis?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/plant-disease-diagnosis?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/your-username/plant-disease-diagnosis?style=social)

---

<div align="center">

**[⬆ 回到顶部](#-智慧农业病虫害诊断系统)**

Made with ❤️ by [Your Name](https://github.com/your-username)

</div>
