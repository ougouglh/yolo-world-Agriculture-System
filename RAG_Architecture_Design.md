# 智慧农业病虫害诊断 RAG 系统架构设计

---

## 一、系统总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户界面层                                      │
│                        (Gradio / Streamlit Web UI)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              应用服务层                                      │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐   │
│  │  图像检测模块  │ ──▶│  RAG 诊断模块  │ ──▶│      报告生成模块         │   │
│  │  YOLO-World   │    │   LangChain   │    │  Structured Output       │   │
│  └───────────────┘    └───────────────┘    └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────────────────┐
│     向量数据库       │ │    LLM 服务     │ │          模型仓库               │
│     ChromaDB        │ │  OpenAI / 本地  │ │  YOLO-World 微调模型 + 预训练   │
│  (农业知识向量库)    │ │  (Qwen/GLM)    │ │                                 │
└─────────────────────┘ └─────────────────┘ └─────────────────────────────────┘
```

---

## 二、核心模块设计

### 2.1 模块划分

| 模块 | 职责 | 核心技术 |
|------|------|----------|
| **图像检测模块** | 接收图片，输出病害类别+置信度+位置 | YOLO-World (微调+预训练集成) |
| **知识检索模块** | 根据检测结果检索相关知识 | ChromaDB + Embedding |
| **诊断生成模块** | 结合检测+知识生成诊断报告 | LangChain + LLM |
| **报告输出模块** | 格式化输出结构化报告 | Pydantic / JSON Schema |

---

### 2.2 数据流程图

```
┌──────────┐
│ 用户上传  │
│   图片    │
└────┬─────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│                    Step 1: 图像检测                          │
│  ┌─────────────────┐      ┌─────────────────┐               │
│  │ YOLO-World 微调 │      │ YOLO-World 预训练│               │
│  └────────┬────────┘      └────────┬────────┘               │
│           │                        │                         │
│           └──────────┬─────────────┘                         │
│                      ▼                                       │
│              ┌──────────────┐                                │
│              │  并集融合策略  │                                │
│              │  NMS 去重     │                                │
│              └──────┬───────┘                                │
└─────────────────────┼────────────────────────────────────────┘
                      │
                      ▼
              检测结果 (JSON)
              {
                "disease": "番茄早疫病",
                "confidence": 0.95,
                "bbox": [x1, y1, x2, y2]
              }
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    Step 2: 知识检索 (RAG)                    │
│                                                              │
│  检测结果 ──▶ Query 构造 ──▶ Embedding ──▶ ChromaDB 检索     │
│                                              │                │
│                                              ▼                │
│                                    Top-K 相关知识片段          │
│                                    (病因/症状/防治方法)        │
└──────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    Step 3: LLM 诊断生成                       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    Prompt Template                      │  │
│  │  你是农业病虫害诊断专家...                               │  │
│  │  检测结果: {detection_result}                           │  │
│  │  参考知识: {retrieved_context}                          │  │
│  │  请生成诊断报告...                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│                    LLM (GPT/Qwen/GLM)                        │
└──────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    Step 4: 报告输出                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  诊断报告                                               │  │
│  │  ├── 病害名称: 番茄早疫病                                │  │
│  │  ├── 置信度: 95%                                        │  │
│  │  ├── 病原: 茄链格孢菌 (Alternaria solani)               │  │
│  │  ├── 症状描述: 叶片出现同心轮纹状褐色斑点...              │  │
│  │  ├── 发病条件: 高温高湿环境...                           │  │
│  │  ├── 防治建议:                                          │  │
│  │  │   ├── 农业防治: 清除病叶，轮作换茬                    │  │
│  │  │   ├── 化学防治: 75%百菌清 600倍液喷施                 │  │
│  │  │   └── 生物防治: 枯草芽孢杆菌制剂                      │  │
│  │  └── 预防措施: ...                                      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、知识库设计

### 3.1 知识结构

每个病害类别包含以下字段：

```json
{
  "disease_id": "tomato_early_blight",
  "disease_name_cn": "番茄早疫病",
  "disease_name_en": "Tomato Early Blight",
  "crop": "番茄",
  "pathogen": {
    "name": "茄链格孢菌",
    "scientific_name": "Alternaria solani",
    "type": "真菌"
  },
  "symptoms": {
    "leaf": "叶片出现圆形或不规则形褐色病斑，具有明显的同心轮纹...",
    "stem": "茎部病斑椭圆形，褐色，略凹陷...",
    "fruit": "果实蒂部附近出现褐色凹陷病斑..."
  },
  "conditions": {
    "temperature": "20-25°C",
    "humidity": "高湿 (>80%)",
    "season": "春末夏初"
  },
  "treatment": {
    "agricultural": ["清除病叶病株", "合理密植", "轮作换茬"],
    "chemical": [
      {"name": "75%百菌清可湿性粉剂", "dosage": "600倍液", "interval": "7-10天"},
      {"name": "64%杀毒矾可湿性粉剂", "dosage": "500倍液", "interval": "7天"}
    ],
    "biological": ["枯草芽孢杆菌", "木霉菌制剂"]
  },
  "prevention": ["选用抗病品种", "加强田间管理", "避免过量氮肥"]
}
```

### 3.2 知识切分策略

```
┌─────────────────────────────────────────────────────────────┐
│                    原始知识文档                              │
│  番茄早疫病.txt / 番茄早疫病.md                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ 切分
┌─────────────────────────────────────────────────────────────┐
│  Chunk 1: [症状描述]                                         │
│  "番茄早疫病主要危害叶片，也可侵染茎和果实。叶片染病..."       │
│  metadata: {disease: "番茄早疫病", section: "症状", crop: "番茄"} │
├─────────────────────────────────────────────────────────────┤
│  Chunk 2: [病原信息]                                         │
│  "病原为茄链格孢菌，属于半知菌类..."                          │
│  metadata: {disease: "番茄早疫病", section: "病原", crop: "番茄"} │
├─────────────────────────────────────────────────────────────┤
│  Chunk 3: [防治方法]                                         │
│  "农业防治：及时清除病叶，增施磷钾肥..."                       │
│  metadata: {disease: "番茄早疫病", section: "防治", crop: "番茄"} │
└─────────────────────────────────────────────────────────────┘
```

**切分参数建议**：
- chunk_size: 500-800 字符
- chunk_overlap: 100 字符
- 按章节自然切分（症状/病原/防治）

### 3.3 ChromaDB Collection 设计

```python
# Collection 结构
collection = chroma_client.create_collection(
    name="plant_disease_knowledge",
    metadata={"description": "农业病虫害知识库"},
    embedding_function=embedding_fn  # 使用中文 Embedding 模型
)

# 文档存储格式
collection.add(
    documents=["番茄早疫病主要危害叶片..."],
    metadatas=[{
        "disease_id": "tomato_early_blight",
        "disease_name": "番茄早疫病",
        "crop": "番茄",
        "section": "症状",
        "source": "植物病理学手册"
    }],
    ids=["tomato_early_blight_symptom_001"]
)
```

---

## 四、技术选型

### 4.1 核心组件

| 组件 | 推荐方案 | 备选方案 |
|------|----------|----------|
| **向量数据库** | ChromaDB | FAISS / Milvus |
| **Embedding 模型** | bge-base-zh-v1.5 (BAAI) | text2vec-chinese / m3e |
| **LLM** | OpenAI GPT-4o-mini | Qwen2.5 / ChatGLM4 / DeepSeek |
| **编排框架** | LangChain | LlamaIndex |
| **Web 框架** | Gradio | Streamlit |

### 4.2 Embedding 模型选择

针对中文农业领域，推荐使用：

| 模型 | 维度 | 优点 | 适用场景 |
|------|------|------|----------|
| **bge-base-zh-v1.5** | 768 | 中文效果好，开源免费 | ✅ 推荐 |
| bge-large-zh-v1.5 | 1024 | 效果更好，资源占用多 | 高精度需求 |
| text-embedding-3-small | 1536 | OpenAI 官方，需付费 | 使用 OpenAI 全家桶 |

### 4.3 LLM 选择对比

| 模型 | 部署方式 | 成本 | 中文效果 | 推荐度 |
|------|----------|------|----------|--------|
| GPT-4o-mini | API 调用 | 低 | 好 | ⭐⭐⭐⭐ |
| GPT-4o | API 调用 | 高 | 很好 | ⭐⭐⭐ |
| Qwen2.5-7B-Instruct | 本地部署 | 免费 | 很好 | ⭐⭐⭐⭐⭐ |
| ChatGLM4-9B | 本地部署 | 免费 | 很好 | ⭐⭐⭐⭐ |
| DeepSeek-V2 | API 调用 | 很低 | 很好 | ⭐⭐⭐⭐⭐ |

**推荐**：
- 如果追求简单快速：**OpenAI GPT-4o-mini** 或 **DeepSeek API**
- 如果需要本地部署：**Qwen2.5-7B-Instruct**（您的 4090 完全可以运行）

---

## 五、RAG Pipeline 详细设计

### 5.1 检索增强策略

```
┌─────────────────────────────────────────────────────────────┐
│                     Query 构造策略                          │
└─────────────────────────────────────────────────────────────┘

检测结果: {"disease": "番茄早疫病", "confidence": 0.95}
                          │
                          ▼
            ┌─────────────────────────┐
            │     Query 模板构造       │
            └─────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Query 1:          Query 2:          Query 3:
"番茄早疫病症状"   "番茄早疫病防治"   "番茄早疫病病原"
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                    多路检索合并
                   (Multi-Query)
                          │
                          ▼
                  Top-K 结果 (K=5)
```

### 5.2 检索策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **基础向量检索** | 直接用病害名称检索 | 简单场景 |
| **多查询检索** | 生成多个查询合并结果 | 提高召回率 |
| **元数据过滤** | 先过滤作物类型再检索 | 提高精度 |
| **混合检索** | 向量 + BM25 关键词 | 最佳效果 |

推荐使用 **元数据过滤 + 向量检索**：

```python
# 伪代码示例
results = collection.query(
    query_texts=["番茄早疫病的症状和防治方法"],
    n_results=5,
    where={"crop": "番茄"}  # 元数据过滤
)
```

### 5.3 Prompt 设计

```python
DIAGNOSIS_PROMPT = """
你是一位专业的农业植物病理学专家，请根据以下信息生成病害诊断报告。

## 检测信息
- 检测到的病害: {disease_name}
- 置信度: {confidence}%
- 检测模型: YOLO-World

## 参考知识
{retrieved_context}

## 任务要求
请根据以上信息，生成一份结构化的诊断报告，包含以下内容：

1. **病害确认**: 确认检测结果，简述该病害的基本信息
2. **症状描述**: 该病害的典型症状表现
3. **病原信息**: 病原体类型及其特点
4. **发病条件**: 容易发病的环境条件
5. **防治建议**: 
   - 农业防治措施
   - 化学防治方案（农药名称、浓度、施用方法）
   - 生物防治选项
6. **预防措施**: 日常预防建议

请确保建议专业、实用、安全，符合农业生产实际。
"""
```

### 5.4 输出结构化

使用 Pydantic 定义输出格式：

```python
from pydantic import BaseModel
from typing import List

class Treatment(BaseModel):
    agricultural: List[str]      # 农业防治
    chemical: List[dict]         # 化学防治 [{name, dosage, method}]
    biological: List[str]        # 生物防治

class DiagnosisReport(BaseModel):
    disease_name: str            # 病害名称
    confidence: float            # 置信度
    crop: str                    # 作物类型
    pathogen: str                # 病原信息
    symptoms: str                # 症状描述
    conditions: str              # 发病条件
    treatment: Treatment         # 防治建议
    prevention: List[str]        # 预防措施
    notes: str                   # 补充说明
```

---

## 六、系统文件结构

```
/root/autodl-tmp/project/
├── models/                              # 模型文件
│   ├── yoloworld_finetuned.pt          # YOLO-World 微调模型
│   ├── yoloworld_pretrained.pt         # YOLO-World 预训练模型
│   └── bge-base-zh-v1.5/               # Embedding 模型 (可选本地)
│
├── knowledge_base/                      # 知识库原始文件
│   ├── diseases/                        # 病害知识文档
│   │   ├── tomato_early_blight.md
│   │   ├── tomato_late_blight.md
│   │   ├── apple_black_rot.md
│   │   └── ...
│   └── knowledge_schema.json            # 知识结构定义
│
├── vectorstore/                         # 向量数据库
│   └── chroma_db/                       # ChromaDB 持久化目录
│
├── src/                                 # 源代码
│   ├── __init__.py
│   ├── detection/                       # 检测模块
│   │   ├── __init__.py
│   │   ├── detector.py                  # YOLO-World 推理
│   │   └── ensemble.py                  # 模型集成逻辑
│   │
│   ├── rag/                             # RAG 模块
│   │   ├── __init__.py
│   │   ├── embeddings.py                # Embedding 封装
│   │   ├── vectorstore.py               # ChromaDB 操作
│   │   ├── retriever.py                 # 检索逻辑
│   │   └── chain.py                     # LangChain 链
│   │
│   ├── llm/                             # LLM 模块
│   │   ├── __init__.py
│   │   ├── prompts.py                   # Prompt 模板
│   │   ├── llm_client.py                # LLM 客户端
│   │   └── output_parser.py             # 输出解析
│   │
│   └── pipeline/                        # 完整流水线
│       ├── __init__.py
│       └── diagnosis_pipeline.py        # 端到端诊断
│
├── scripts/                             # 工具脚本
│   ├── build_vectorstore.py             # 构建向量库
│   ├── test_retrieval.py                # 测试检索
│   └── test_pipeline.py                 # 测试完整流程
│
├── app/                                 # Web 应用
│   ├── gradio_app.py                    # Gradio 界面
│   └── assets/                          # 静态资源
│
├── configs/                             # 配置文件
│   ├── config.yaml                      # 主配置
│   └── prompts.yaml                     # Prompt 配置
│
├── requirements.txt                     # 依赖列表
└── README.md                            # 项目说明
```

---

## 七、实现步骤

### Phase 1: 知识库构建 (0.5天)

| 步骤 | 任务 | 产出 |
|------|------|------|
| 1.1 | 整理 25 类病害知识文档 | `knowledge_base/diseases/*.md` |
| 1.2 | 下载 Embedding 模型 | `bge-base-zh-v1.5` |
| 1.3 | 文档切分 + 向量化 | `vectorstore/chroma_db/` |
| 1.4 | 测试检索效果 | 验证 Top-5 相关性 |

### Phase 2: RAG 链构建 (0.5天)

| 步骤 | 任务 | 产出 |
|------|------|------|
| 2.1 | 封装 ChromaDB 检索器 | `src/rag/retriever.py` |
| 2.2 | 设计 Prompt 模板 | `src/llm/prompts.py` |
| 2.3 | 对接 LLM (OpenAI/本地) | `src/llm/llm_client.py` |
| 2.4 | 构建 LangChain 链 | `src/rag/chain.py` |

### Phase 3: Pipeline 整合 (0.5天)

| 步骤 | 任务 | 产出 |
|------|------|------|
| 3.1 | 整合 YOLO-World 检测 | `src/detection/detector.py` |
| 3.2 | 串联完整流程 | `src/pipeline/diagnosis_pipeline.py` |
| 3.3 | 输出结构化报告 | JSON / Markdown 报告 |

### Phase 4: Web 界面 (1天)

| 步骤 | 任务 | 产出 |
|------|------|------|
| 4.1 | Gradio 界面开发 | `app/gradio_app.py` |
| 4.2 | 可视化检测结果 | 标注框 + 标签 |
| 4.3 | 诊断报告展示 | 结构化卡片 |

---

## 八、关键代码框架

### 8.1 检索器

```python
# src/rag/retriever.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class DiseaseRetriever:
    def __init__(self, persist_dir: str):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-zh-v1.5",
            model_kwargs={'device': 'cuda'}
        )
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
    
    def retrieve(self, disease_name: str, k: int = 5) -> list:
        """根据病害名称检索相关知识"""
        query = f"{disease_name}的症状、病原和防治方法"
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
```

### 8.2 诊断链

```python
# src/rag/chain.py
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class DiagnosisChain:
    def __init__(self, retriever, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.retriever = retriever
        self.prompt = PromptTemplate.from_template(DIAGNOSIS_PROMPT)
        
    def diagnose(self, disease_name: str, confidence: float) -> dict:
        """生成诊断报告"""
        # 1. 检索相关知识
        docs = self.retriever.retrieve(disease_name)
        context = "\n".join([doc.page_content for doc in docs])
        
        # 2. 构造 Prompt
        prompt = self.prompt.format(
            disease_name=disease_name,
            confidence=confidence * 100,
            retrieved_context=context
        )
        
        # 3. 调用 LLM
        response = self.llm.invoke(prompt)
        
        return self._parse_response(response.content)
```

### 8.3 完整 Pipeline

```python
# src/pipeline/diagnosis_pipeline.py
class PlantDiseaseDiagnosisPipeline:
    def __init__(self):
        self.detector = YOLOWorldEnsemble(...)
        self.retriever = DiseaseRetriever(...)
        self.diagnosis_chain = DiagnosisChain(...)
    
    def run(self, image_path: str) -> dict:
        """完整诊断流程"""
        # Step 1: 检测
        detection = self.detector.detect(image_path)
        
        # Step 2: 诊断
        report = self.diagnosis_chain.diagnose(
            disease_name=detection['disease'],
            confidence=detection['confidence']
        )
        
        # Step 3: 组装结果
        return {
            "detection": detection,
            "diagnosis": report,
            "annotated_image": detection['annotated_image']
        }
```

---

## 九、配置文件示例

```yaml
# configs/config.yaml
project:
  name: "智慧农业病虫害诊断系统"
  version: "1.0.0"

detection:
  finetuned_model: "models/yoloworld_finetuned.pt"
  pretrained_model: "models/yoloworld_pretrained.pt"
  confidence_threshold: 0.25
  nms_threshold: 0.5

rag:
  vectorstore_path: "vectorstore/chroma_db"
  embedding_model: "BAAI/bge-base-zh-v1.5"
  top_k: 5
  chunk_size: 500
  chunk_overlap: 100

llm:
  provider: "openai"  # or "local"
  model: "gpt-4o-mini"
  temperature: 0
  max_tokens: 2000
  # 本地模型配置 (如使用 Qwen)
  # local_model_path: "models/Qwen2.5-7B-Instruct"

app:
  host: "0.0.0.0"
  port: 7860
  share: true
```

---

## 十、预期效果

### 示例输入输出

**输入**：一张番茄叶片病害图片

**输出**：
```markdown
# 🌱 植物病害诊断报告

## 基本信息
| 项目 | 内容 |
|------|------|
| 检测病害 | 番茄早疫病 |
| 置信度 | 95.2% |
| 作物类型 | 番茄 |

## 病原信息
**茄链格孢菌** (*Alternaria solani*)，属半知菌亚门真菌。

## 症状描述
叶片出现圆形或不规则形褐色病斑，病斑上有明显的同心轮纹，
边缘有黄色晕圈。严重时病斑连片，导致叶片枯死。

## 发病条件
- 温度：20-25°C
- 湿度：>80%
- 易感期：连续阴雨天气后

## 防治建议

### 🌾 农业防治
1. 及时清除病叶、病株，集中烧毁
2. 合理密植，改善通风透光
3. 避免过量施用氮肥

### 💊 化学防治
| 药剂 | 浓度 | 施用方法 |
|------|------|----------|
| 75%百菌清WP | 600倍液 | 叶面喷施，间隔7-10天 |
| 64%杀毒矾WP | 500倍液 | 发病初期喷施 |

### 🦠 生物防治
- 枯草芽孢杆菌制剂
- 木霉菌制剂

## 预防措施
- 选用抗病品种
- 种子消毒处理
- 加强田间排水

---
*诊断时间: 2026-01-09 14:30:00*
*模型: YOLO-World + GPT-4o-mini*
```

---

## 十一、总结

本架构设计覆盖了从**图像检测**到**智能诊断**的完整流程：

| 模块 | 技术选型 | 特点 |
|------|----------|------|
| 检测 | YOLO-World 集成 | 支持零样本，71% 新病害检测率 |
| 知识库 | ChromaDB + BGE | 中文友好，易于维护 |
| 检索 | LangChain RAG | 灵活，可扩展 |
| 生成 | GPT-4o-mini / Qwen | 成本低，效果好 |
| 界面 | Gradio | 快速开发，易部署 |

**预计开发时间**：2-3 天

---

确认此架构后，我可以帮您编写具体实现代码。
