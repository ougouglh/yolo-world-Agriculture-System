"""
RAG 检索模块
从 ChromaDB 向量数据库中检索相关病害知识
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseKnowledgeRetriever:
    """病害知识检索器"""

    def __init__(
            self,
            persist_dir: str = "./vectorstore/chroma_db",
            collection_name: str = "plant_disease_knowledge",
            model_name: str = "BAAI/bge-base-zh-v1.5",
            top_k: int = 5
    ):
        """
        初始化检索器

        Args:
            persist_dir: ChromaDB 持久化目录
            collection_name: 集合名称
            model_name: Embedding 模型名称
            top_k: 返回的文档数量
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.top_k = top_k

        logger.info(f"初始化 ChromaDB 客户端: {persist_dir}")

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 加载 Embedding 函数
        logger.info(f"加载 Embedding 模型: {model_name}")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # 获取集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            logger.info(f"成功加载集合: {collection_name} (文档数: {self.collection.count()})")
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            raise

    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数量（默认使用初始化时的值）
            filter_dict: 过滤条件（例如 {"crop": "番茄"}）

        Returns:
            检索结果列表，每个元素包含 document, metadata, distance
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"检索查询: {query[:50]}... (top_k={top_k})")

        try:
            # 执行检索
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_dict
            )

            # 格式化结果
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })

            logger.info(f"检索到 {len(formatted_results)} 条结果")
            return formatted_results

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def retrieve_by_disease_name(self, disease_name: str, top_k: int = 3) -> List[Dict]:
        """
        根据病害名称检索

        Args:
            disease_name: 病害名称（中文或英文）
            top_k: 返回文档数量

        Returns:
            检索结果列表
        """
        return self.retrieve(
            query=f"{disease_name}的症状、病原、防治方法",
            top_k=top_k
        )

    def format_context(self, results: List[Dict], max_length: int = 2000) -> str:
        """
        将检索结果格式化为上下文字符串

        Args:
            results: 检索结果列表
            max_length: 最大长度（防止 token 过多）

        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            document = result['document']

            # 构建单条记录
            part = f"\n【参考资料 {i}】\n"
            part += f"病害名称：{metadata.get('disease_name', '未知')}\n"
            part += f"作物：{metadata.get('crop', '未知')}\n"
            part += f"内容：{document}\n"
            part += f"相似度分数：{1 - result['distance']:.3f}\n"

            # 检查长度限制
            if current_length + len(part) > max_length:
                logger.warning(f"上下文超过最大长度 {max_length}，截断剩余内容")
                break

            context_parts.append(part)
            current_length += len(part)

        return "".join(context_parts)

    def get_statistics(self) -> Dict:
        """获取向量库统计信息"""
        try:
            count = self.collection.count()
            # 获取所有元数据以统计病害类别
            all_data = self.collection.get()
            diseases = set()
            crops = set()

            for metadata in all_data['metadatas']:
                if 'disease_name' in metadata:
                    diseases.add(metadata['disease_name'])
                if 'crop' in metadata:
                    crops.add(metadata['crop'])

            return {
                'total_documents': count,
                'unique_diseases': len(diseases),
                'unique_crops': len(crops),
                'diseases': sorted(list(diseases)),
                'crops': sorted(list(crops))
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


def test_retriever():
    """测试检索器功能"""
    print("=" * 60)
    print("测试 RAG 检索器")
    print("=" * 60)

    # 初始化检索器
    retriever = DiseaseKnowledgeRetriever()

    # 获取统计信息
    stats = retriever.get_statistics()
    print(f"\n向量库统计：")
    print(f"  总文档数: {stats.get('total_documents', 0)}")
    print(f"  病害类别数: {stats.get('unique_diseases', 0)}")
    print(f"  作物种类数: {stats.get('unique_crops', 0)}")

    # 测试检索
    test_queries = [
        "番茄叶片出现褐色斑点怎么办",
        "辣椒叶子发黄卷曲",
        "马铃薯晚疫病的防治方法"
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"查询: {query}")
        print(f"{'=' * 60}")

        results = retriever.retrieve(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n[结果 {i}] {result['metadata'].get('disease_name', '未知')}")
            print(f"作物: {result['metadata'].get('crop', '未知')}")
            print(f"相似度: {1 - result['distance']:.3f}")
            print(f"内容: {result['document'][:150]}...")

        # 测试格式化上下文
        context = retriever.format_context(results)
        print(f"\n格式化上下文长度: {len(context)} 字符")


if __name__ == "__main__":
    test_retriever()