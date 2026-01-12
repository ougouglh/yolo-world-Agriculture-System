"""
知识库向量化脚本
将 disease_knowledge_base.json 切分并存入 ChromaDB

使用方法:
    python build_vectorstore.py --json_path disease_knowledge_base.json --persist_dir ./vectorstore/chroma_db

依赖安装:
    pip install chromadb langchain langchain-community sentence-transformers
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings


def load_knowledge_base(json_path: str) -> Dict:
    """加载知识库JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_chunks(disease: Dict) -> List[Dict]:
    """
    将单个病害信息切分成多个chunks
    
    切分策略:
    - Chunk 1: 基本信息 + 症状
    - Chunk 2: 发病条件 + 预防措施
    - Chunk 3: 农药防治方案
    - Chunk 4: 完整摘要（用于通用检索）
    """
    chunks = []
    
    disease_name_cn = disease.get('name_cn', '')
    disease_name_en = disease.get('name_en', '')
    crop = disease.get('crop', '')
    category = disease.get('category', '')
    
    # 基础 metadata（每个 chunk 都会有）
    base_metadata = {
        "disease_id": disease.get('id', 0),
        "disease_name_cn": disease_name_cn,
        "disease_name_en": disease_name_en,
        "crop": crop,
        "category": category,  # training 或 zeroshot
    }
    
    # ==================== Chunk 1: 症状信息 ====================
    symptoms = disease.get('symptoms', '')
    pathogen = disease.get('pathogen', '')
    visual_features = disease.get('visual_features', '')
    
    if symptoms:
        chunk1_text = f"""病害名称：{disease_name_cn}（{disease_name_en}）
作物：{crop}
病原：{pathogen}
症状描述：{symptoms}
视觉特征：{visual_features}"""
        
        chunks.append({
            "text": chunk1_text,
            "metadata": {
                **base_metadata,
                "chunk_type": "symptoms",
                "section": "症状与病原"
            }
        })
    
    # ==================== Chunk 2: 发病条件与预防 ====================
    conditions = disease.get('conditions', '')
    prevention = disease.get('prevention', [])
    
    if conditions or prevention:
        prevention_text = '\n'.join([f"  - {p}" for p in prevention]) if prevention else '无'
        
        chunk2_text = f"""病害名称：{disease_name_cn}（{disease_name_en}）
作物：{crop}
发病条件：{conditions}
预防措施：
{prevention_text}"""
        
        chunks.append({
            "text": chunk2_text,
            "metadata": {
                **base_metadata,
                "chunk_type": "prevention",
                "section": "发病条件与预防"
            }
        })
    
    # ==================== Chunk 3: 农药防治 ====================
    pesticides = disease.get('pesticides', [])
    
    if pesticides:
        pesticides_text = '\n'.join([f"  - {p}" for p in pesticides])
        
        chunk3_text = f"""病害名称：{disease_name_cn}（{disease_name_en}）
作物：{crop}
推荐农药与用法：
{pesticides_text}"""
        
        chunks.append({
            "text": chunk3_text,
            "metadata": {
                **base_metadata,
                "chunk_type": "treatment",
                "section": "化学防治"
            }
        })
    
    # ==================== Chunk 4: 完整摘要 ====================
    prevention_brief = '、'.join(prevention[:3]) if prevention else '无'
    pesticides_brief = '、'.join([p.split(' ')[0] for p in pesticides[:3]]) if pesticides else '无'
    
    chunk4_text = f"""病害名称：{disease_name_cn}（{disease_name_en}）
作物类型：{crop}
病原：{pathogen}
主要症状：{symptoms}
发病条件：{conditions}
视觉特征：{visual_features}
主要预防措施：{prevention_brief}
推荐药剂：{pesticides_brief}"""
    
    chunks.append({
        "text": chunk4_text,
        "metadata": {
            **base_metadata,
            "chunk_type": "summary",
            "section": "综合摘要"
        }
    })
    
    return chunks


def build_vectorstore(
    json_path: str,
    persist_dir: str,
    collection_name: str = "plant_disease_knowledge",
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
):
    """
    构建向量数据库
    
    Args:
        json_path: 知识库JSON文件路径
        persist_dir: ChromaDB持久化目录
        collection_name: 集合名称
        embedding_model: Embedding模型名称
    """
    print("=" * 60)
    print("知识库向量化脚本")
    print("=" * 60)
    
    # 1. 加载知识库
    print(f"\n[1/5] 加载知识库: {json_path}")
    kb = load_knowledge_base(json_path)
    diseases = kb.get('diseases', [])
    print(f"      共 {len(diseases)} 个病害类别")
    
    # 2. 切分文档
    print(f"\n[2/5] 切分文档...")
    all_chunks = []
    for disease in diseases:
        chunks = create_chunks(disease)
        all_chunks.extend(chunks)
    print(f"      共生成 {len(all_chunks)} 个文档块")
    
    # 3. 加载 Embedding 模型
    print(f"\n[3/5] 加载 Embedding 模型: {embedding_model}")
    print("      (首次运行会自动下载模型，请耐心等待...)")
    
    from chromadb.utils import embedding_functions
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        device="cuda"  # 如果没有GPU，改为 "cpu"
    )
    print("      模型加载完成!")
    
    # 4. 创建/连接 ChromaDB
    print(f"\n[4/5] 初始化 ChromaDB: {persist_dir}")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 删除已存在的同名集合（如果有）
    try:
        client.delete_collection(collection_name)
        print(f"      删除已存在的集合: {collection_name}")
    except:
        pass
    
    # 创建新集合
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "智慧农业病虫害知识库"}
    )
    print(f"      创建集合: {collection_name}")
    
    # 5. 写入向量数据库
    print(f"\n[5/5] 写入向量数据库...")
    
    # 准备数据
    documents = [chunk["text"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    ids = [f"{chunk['metadata']['disease_name_en'].replace(' ', '_')}_{chunk['metadata']['chunk_type']}_{i}" 
           for i, chunk in enumerate(all_chunks)]
    
    # 批量写入
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"      已写入 {end_idx}/{len(documents)} 条记录")
    
    # 完成
    print("\n" + "=" * 60)
    print("向量化完成!")
    print("=" * 60)
    print(f"  - 知识库路径: {json_path}")
    print(f"  - 向量库路径: {persist_dir}")
    print(f"  - 集合名称: {collection_name}")
    print(f"  - 文档总数: {collection.count()}")
    print(f"  - Embedding模型: {embedding_model}")
    
    # 测试检索
    print("\n" + "-" * 60)
    print("测试检索...")
    test_query = "番茄叶片出现褐色斑点怎么办"
    results = collection.query(
        query_texts=[test_query],
        n_results=3
    )
    
    print(f"  查询: {test_query}")
    print(f"  Top-3 结果:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n  [{i+1}] {meta['disease_name_cn']} - {meta['section']}")
        print(f"      {doc[:100]}...")
    
    return collection


def main():
    parser = argparse.ArgumentParser(description='知识库向量化脚本')
    parser.add_argument('--json_path', type=str, default='disease_knowledge_base.json',
                        help='知识库JSON文件路径')
    parser.add_argument('--persist_dir', type=str, default='./vectorstore/chroma_db',
                        help='ChromaDB持久化目录')
    parser.add_argument('--collection_name', type=str, default='plant_disease_knowledge',
                        help='集合名称')
    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-base-zh-v1.5',
                        help='Embedding模型')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    build_vectorstore(
        json_path=args.json_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model
    )


if __name__ == "__main__":
    main()
