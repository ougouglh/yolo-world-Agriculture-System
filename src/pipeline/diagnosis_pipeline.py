"""
完整诊断流程
整合 YOLO-World 检测 + RAG 检索 + LLM 生成
路径: src/pipeline/diagnosis_pipeline.py

优化内容：
- 新增进度回调支持
- 新增 ConversationalDiagnosisPipeline 支持多轮对话追问
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import logging
from http import HTTPStatus
import dashscope
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retriever import DiseaseKnowledgeRetriever
from src.rag.llm_generator import DiagnosisReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantDiseaseDiagnosisPipeline:
    """植物病害诊断完整流程"""

    def __init__(
            self,
            yolo_model_path: str,
            vectorstore_path: str = "./vectorstore/chroma_db",
            api_key: Optional[str] = None,
            llm_model: str = "qwen-turbo",
            confidence_threshold: float = 0.25
    ):
        """
        初始化诊断流程

        Args:
            yolo_model_path: YOLO-World 模型路径
            vectorstore_path: 向量数据库路径
            api_key: 阿里云百炼 API Key
            llm_model: LLM 模型名称
            confidence_threshold: 检测置信度阈值
        """
        logger.info("=" * 60)
        logger.info("初始化植物病害诊断系统")
        logger.info("=" * 60)

        self.confidence_threshold = confidence_threshold

        # 1. 加载 YOLO-World 模型
        logger.info(f"[1/3] 加载 YOLO-World 模型: {yolo_model_path}")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            logger.info("✅ YOLO-World 模型加载成功")
        except Exception as e:
            logger.error(f"❌ YOLO-World 模型加载失败: {e}")
            raise

        # 2. 初始化 RAG 检索器
        logger.info(f"[2/3] 初始化 RAG 检索器: {vectorstore_path}")
        try:
            self.retriever = DiseaseKnowledgeRetriever(persist_dir=vectorstore_path)
            logger.info("✅ RAG 检索器初始化成功")
        except Exception as e:
            logger.error(f"❌ RAG 检索器初始化失败: {e}")
            raise

        # 3. 初始化 LLM 生成器
        logger.info(f"[3/3] 初始化 LLM 生成器: {llm_model}")
        try:
            self.generator = DiagnosisReportGenerator(api_key=api_key, model=llm_model)
            logger.info("✅ LLM 生成器初始化成功")
        except Exception as e:
            logger.error(f"❌ LLM 生成器初始化失败: {e}")
            raise

        logger.info("=" * 60)
        logger.info("🎉 诊断系统初始化完成！")
        logger.info("=" * 60)

    def detect_diseases(self, image_path: str) -> Tuple[List[Dict], np.ndarray]:
        """
        使用 YOLO-World 检测病害

        Args:
            image_path: 图像路径

        Returns:
            (检测结果列表, 标注后的图像数组)
        """
        logger.info(f"检测图像: {image_path}")

        # 执行检测
        results = self.yolo_model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            save=False,
            verbose=False
        )

        # 解析检测结果
        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                bbox = box.xyxy[0].cpu().numpy().tolist()

                detections.append({
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': bbox,
                    'class_id': cls_id
                })

                logger.info(f"  检测到: {cls_name} (置信度: {conf:.3f})")
        else:
            logger.warning("  未检测到病害")

        # 获取标注后的图像
        annotated_image = result.plot()

        return detections, annotated_image

    def retrieve_knowledge(self, disease_names: List[str], top_k: int = 3) -> Tuple[List[Dict], str]:
        """
        从知识库检索相关信息

        Args:
            disease_names: 病害名称列表
            top_k: 每种病害检索的文档数量

        Returns:
            (所有检索结果, 格式化的上下文)
        """
        logger.info(f"检索知识库，病害: {disease_names}")

        all_results = []

        for disease_name in disease_names:
            results = self.retriever.retrieve_by_disease_name(
                disease_name=disease_name,
                top_k=top_k
            )
            all_results.extend(results)
            logger.info(f"  {disease_name}: 检索到 {len(results)} 条记录")

        # 格式化上下文
        context = self.retriever.format_context(all_results, max_length=3000)

        return all_results, context

    def generate_report(
            self,
            disease_names: List[str],
            context: str,
            image_description: Optional[str] = None
    ) -> Dict:
        """
        生成诊断报告

        Args:
            disease_names: 病害名称列表
            context: 知识库上下文
            image_description: 图像描述

        Returns:
            生成结果字典
        """
        logger.info("生成诊断报告...")

        result = self.generator.generate_diagnosis_report(
            detected_diseases=disease_names,
            context=context,
            image_description=image_description
        )

        if result['success']:
            logger.info("✅ 诊断报告生成成功")
        else:
            logger.error(f"❌ 诊断报告生成失败: {result.get('error', '未知错误')}")

        return result

    def diagnose(
            self,
            image_path: str,
            retrieve_top_k: int = 3,
            return_annotated_image: bool = True,
            progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """
        完整诊断流程

        Args:
            image_path: 图像路径
            retrieve_top_k: 检索文档数量
            return_annotated_image: 是否返回标注图像
            progress_callback: 进度回调函数 (新增)

        Returns:
            完整诊断结果字典
        """
        logger.info("\n" + "=" * 60)
        logger.info("开始完整诊断流程")
        logger.info("=" * 60)

        result = {
            'success': False,
            'image_path': image_path,
            'detections': [],
            'disease_names': [],
            'knowledge_retrieval': [],
            'diagnosis_report': None,
            'annotated_image': None,
            'error': None
        }

        try:
            # 步骤 1: 病害检测
            if progress_callback:
                progress_callback("正在检测病害...")

            logger.info("\n[步骤 1/3] 执行病害检测...")
            detections, annotated_image = self.detect_diseases(image_path)
            result['detections'] = detections

            if return_annotated_image:
                result['annotated_image'] = annotated_image

            # 如果未检测到病害
            if not detections:
                result['error'] = "未检测到病害"
                logger.warning("诊断终止：未检测到病害")
                return result

            # 提取病害名称（去重）
            disease_names = list(set([d['class_name'] for d in detections]))
            result['disease_names'] = disease_names
            logger.info(f"检测到 {len(disease_names)} 种病害: {', '.join(disease_names)}")

            # 步骤 2: 知识检索
            if progress_callback:
                progress_callback("正在检索知识库...")

            logger.info("\n[步骤 2/3] 检索知识库...")
            retrieval_results, context = self.retrieve_knowledge(
                disease_names=disease_names,
                top_k=retrieve_top_k
            )
            result['knowledge_retrieval'] = retrieval_results

            # 步骤 3: 生成报告
            if progress_callback:
                progress_callback("正在生成诊断报告...")

            logger.info("\n[步骤 3/3] 生成诊断报告...")
            report_result = self.generate_report(
                disease_names=disease_names,
                context=context,
                image_description=f"检测到 {len(detections)} 个病害区域"
            )

            result['diagnosis_report'] = report_result
            result['success'] = report_result.get('success', False)

            if result['success']:
                if progress_callback:
                    progress_callback("诊断完成！")
                logger.info("\n" + "=" * 60)
                logger.info("🎉 诊断完成！")
                logger.info("=" * 60)
            else:
                result['error'] = report_result.get('error', '报告生成失败')

            return result

        except Exception as e:
            logger.error(f"诊断过程出错: {e}")
            result['error'] = str(e)
            return result

    def get_statistics(self) -> Dict:
        """获取向量库统计信息"""
        return self.retriever.get_statistics()


class ConversationalDiagnosisPipeline(PlantDiseaseDiagnosisPipeline):
    """
    支持多轮对话的诊断流程（新增功能）

    继承自 PlantDiseaseDiagnosisPipeline，增加：
    - 对话历史管理
    - 追问功能
    - 上下文保持
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
        self.last_context = ""
        self.last_disease_names = []

    def diagnose(self, *args, **kwargs) -> Dict:
        """重写诊断方法，保存上下文供追问使用"""
        result = super().diagnose(*args, **kwargs)

        # 保存上下文
        if result.get('knowledge_retrieval'):
            self.last_context = self.retriever.format_context(
                result['knowledge_retrieval'],
                max_length=2000
            )

        if result.get('disease_names'):
            self.last_disease_names = result['disease_names']

        # 清空之前的对话历史（新的诊断开始新对话）
        self.conversation_history = []

        return result

    def ask_followup(self, question: str, additional_context: str = "") -> str:
        """
        追问功能 - 基于之前的诊断结果回答问题

        Args:
            question: 用户追问问题
            additional_context: 额外上下文（可选）

        Returns:
            回答文本
        """
        if not question.strip():
            return "请输入您的问题"

        # 构建历史对话
        history_str = ""
        if self.conversation_history:
            history_str = "\n".join([
                f"用户: {h['user']}\n助手: {h['assistant']}"
                for h in self.conversation_history[-3:]  # 最近3轮
            ])

        # 合并上下文
        context = additional_context or self.last_context

        # 病害信息
        disease_info = ""
        if self.last_disease_names:
            disease_info = f"当前诊断的病害: {', '.join(self.last_disease_names)}"

        # 构建追问 prompt
        prompt = f"""你是农业病害诊断专家。用户之前对作物进行了病害诊断，现在有追问。

{disease_info}

之前的对话：
{history_str}

相关知识库信息：
{context}

用户追问：{question}

请专业、清晰、简洁地回答。如果问题与之前的诊断相关，请基于诊断结果回答；如果是新问题，请给出专业建议。"""

        try:
            response = dashscope.Generation.call(
                model=self.generator.model,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7,
                result_format='message'
            )

            if response.status_code == HTTPStatus.OK:
                answer = response.output.choices[0].message.content

                # 保存到对话历史
                self.conversation_history.append({
                    "user": question,
                    "assistant": answer
                })

                return answer
            else:
                return f"回答失败: {response.message}"

        except Exception as e:
            logger.error(f"追问回答失败: {e}")
            return f"回答失败: {str(e)}"

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        self.last_context = ""
        self.last_disease_names = []


def test_pipeline():
    """测试完整诊断流程"""
    print("=" * 60)
    print("测试植物病害诊断流程")
    print("=" * 60)

    # 配置参数
    yolo_model_path = "/root/autodl-tmp/project/runs/yoloworld/train/weights/best.pt"
    vectorstore_path = "./vectorstore/chroma_db"
    api_key = "sk-26d1261b1bd44fae92985f4cdee517e5"

    # 测试图像路径
    test_image = "/root/autodl-tmp/project/dataset/images/test_zeroshot/Tomato_Tomato_Yellow_Leaf_Curl_Virus_5350.jpg"

    # 使用支持对话的 Pipeline
    pipeline = ConversationalDiagnosisPipeline(
        yolo_model_path=yolo_model_path,
        vectorstore_path=vectorstore_path,
        api_key=api_key,
        confidence_threshold=0.25
    )

    # 执行诊断
    result = pipeline.diagnose(
        image_path=test_image,
        retrieve_top_k=3
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)

    if result['success']:
        print(f"\n✅ 诊断成功")
        print(f"\n检测到的病害: {', '.join(result['disease_names'])}")
        print(f"\n检测详情:")
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class_name']} (置信度: {det['confidence']:.3f})")

        print(f"\n知识库检索: 检索到 {len(result['knowledge_retrieval'])} 条记录")

        print("\n" + "-" * 60)
        print("诊断报告:")
        print("-" * 60)
        print(result['diagnosis_report']['report'])

        if 'token_usage' in result['diagnosis_report']:
            usage = result['diagnosis_report']['token_usage']
            print(f"\nToken 使用: {usage['total_tokens']} tokens")

        # 测试追问功能
        print("\n" + "=" * 60)
        print("测试追问功能")
        print("=" * 60)

        answer = pipeline.ask_followup("这种病害在什么季节最容易发生？")
        print(f"\n追问: 这种病害在什么季节最容易发生？")
        print(f"回答: {answer}")

        answer2 = pipeline.ask_followup("有没有生物防治的方法？")
        print(f"\n追问: 有没有生物防治的方法？")
        print(f"回答: {answer2}")

    else:
        print(f"\n❌ 诊断失败: {result.get('error', '未知错误')}")


if __name__ == "__main__":
    test_pipeline()