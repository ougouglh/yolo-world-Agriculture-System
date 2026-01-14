"""
LLM 诊断报告生成模块
使用阿里云百炼（通义千问）生成病害诊断报告
"""

import os
from http import HTTPStatus
import dashscope
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnosisReportGenerator:
    """诊断报告生成器"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "qwen-turbo"
    ):
        """
        初始化生成器

        Args:
            api_key: 阿里云百炼 API Key（如果不提供，从环境变量读取）
            model: 使用的模型名称
        """
        # 设置 API Key
        if api_key:
            dashscope.api_key = api_key
        else:
            dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

        if not dashscope.api_key:
            raise ValueError("请提供 API Key 或设置环境变量 DASHSCOPE_API_KEY")

        self.model = model
        logger.info(f"初始化 LLM 生成器，模型: {model}")

    def generate_diagnosis_report(
            self,
            detected_diseases: List[str],
            context: str,
            image_description: Optional[str] = None
    ) -> Dict:
        """
        生成诊断报告

        Args:
            detected_diseases: 检测到的病害名称列表
            context: 从知识库检索到的上下文信息
            image_description: 图像描述（可选）

        Returns:
            生成的诊断报告字典
        """
        # 构建提示词
        prompt = self._build_prompt(detected_diseases, context, image_description)

        logger.info(f"生成诊断报告，检测到病害: {detected_diseases}")
        logger.debug(f"提示词长度: {len(prompt)} 字符")

        try:
            # 调用通义千问 API
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9,
                result_format='message'
            )

            # 检查响应状态
            if response.status_code == HTTPStatus.OK:
                report_text = response.output.choices[0].message.content
                logger.info("诊断报告生成成功")

                return {
                    'success': True,
                    'report': report_text,
                    'detected_diseases': detected_diseases,
                    'model': self.model,
                    'token_usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                }
            else:
                logger.error(f"API 调用失败: {response.code} - {response.message}")
                return {
                    'success': False,
                    'error': f"{response.code}: {response.message}",
                    'detected_diseases': detected_diseases
                }

        except Exception as e:
            logger.error(f"生成报告时出错: {e}")
            return {
                'success': False,
                'error': str(e),
                'detected_diseases': detected_diseases
            }

    def _build_prompt(
            self,
            detected_diseases: List[str],
            context: str,
            image_description: Optional[str] = None
    ) -> str:
        """构建提示词"""

        # 基础系统提示
        system_prompt = """你是一位专业的农业病害诊断专家，擅长识别和分析各种作物病害。
你的任务是根据检测结果和专业知识库，为农民提供准确、实用的病害诊断报告。

请按照以下格式输出诊断报告：

## 🔍 检测结果概览
[简要说明检测到的病害及严重程度]

## 📋 病害详细分析
[对每种检测到的病害进行详细分析]

### 病害1：[病害名称]
- **病原**：[病原体名称]
- **主要症状**：[详细症状描述]
- **发病条件**：[发病环境条件]
- **危害程度**：[对作物的影响]

## 🛡️ 防治建议

### 1. 农业防治措施
[列出具体的农业管理建议]

### 2. 化学防治方案
[推荐的农药及使用方法]

### 3. 注意事项
[使用农药的注意事项和安全提示]

## 📌 总结与建议
[给出综合评估和行动建议]

请确保：
1. 信息准确、专业
2. 语言通俗易懂，适合农民阅读
3. 防治建议具体可操作
4. 包含安全用药提示
"""

        # 构建用户查询
        user_query = f"""
根据以下信息，生成一份详细的病害诊断报告：

**检测到的病害：**
{', '.join(detected_diseases)}
"""

        if image_description:
            user_query += f"\n**图像观察：**\n{image_description}\n"

        user_query += f"""
**专业知识库参考资料：**
{context}

请基于以上信息生成完整的诊断报告。
"""

        # 组合完整提示词
        full_prompt = f"{system_prompt}\n\n{user_query}"

        return full_prompt

    def generate_simple_summary(self, detected_diseases: List[str]) -> str:
        """
        生成简单摘要（不依赖知识库，快速响应）

        Args:
            detected_diseases: 检测到的病害名称列表

        Returns:
            简单摘要文本
        """
        if not detected_diseases:
            return "未检测到病害"

        prompt = f"""请用1-2句话简要说明以下病害的主要特征：
{', '.join(detected_diseases)}

要求：语言简洁，突出重点。"""

        try:
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                max_tokens=200,
                temperature=0.5
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.text
            else:
                return "无法生成摘要"
        except:
            return "无法生成摘要"


def test_generator():
    """测试生成器功能"""
    print("=" * 60)
    print("测试 LLM 诊断报告生成器")
    print("=" * 60)

    # 设置 API Key（从环境变量或直接指定）
    api_key = "sk-26d1261b1bd44fae92985f4cdee517e5"

    # 初始化生成器
    generator = DiagnosisReportGenerator(api_key=api_key)

    # 模拟检测结果
    detected_diseases = ["番茄早疫病", "番茄晚疫病"]

    # 模拟知识库上下文
    context = """
【参考资料 1】
病害名称：番茄早疫病（Tomato Early Blight）
作物：番茄
病原：茄链格孢菌 (Alternaria solani)
症状描述：叶片上出现圆形或近圆形褐色病斑，具有明显的同心轮纹，呈靶心样。病斑周围常有黄色晕圈。
发病条件：温暖潮湿环境（20-30°C），相对湿度高于80%时发病严重。
防治方法：选用抗病品种，实行3年以上轮作，及时清除病残体。
推荐农药：75%百菌清可湿性粉剂 600倍液，64%杀毒矾可湿性粉剂 500倍液，70%代森锰锌可湿性粉剂 500倍液

【参考资料 2】
病害名称：番茄晚疫病（Tomato Late Blight）
作物：番茄
病原：致病疫霉 (Phytophthora infestans)
症状描述：叶片出现暗绿色水渍状不规则病斑，迅速扩大变褐色，潮湿时叶背产生白色霉层。
发病条件：低温高湿（18-22°C，湿度>90%），阴雨连绵时易爆发。
防治方法：选用抗病品种，避免密植，加强通风。
推荐农药：68.75%银法利悬浮剂 1000倍液，72%霜脲锰锌可湿性粉剂 600倍液
"""

    # 生成报告
    print("\n生成诊断报告中...\n")
    result = generator.generate_diagnosis_report(
        detected_diseases=detected_diseases,
        context=context,
        image_description="番茄叶片上观察到褐色同心轮纹状病斑和水渍状不规则斑点"
    )

    if result['success']:
        print("✅ 报告生成成功！")
        print("\n" + "=" * 60)
        print(result['report'])
        print("=" * 60)
        print(f"\nToken 使用统计:")
        print(f"  输入: {result['token_usage']['input_tokens']} tokens")
        print(f"  输出: {result['token_usage']['output_tokens']} tokens")
        print(f"  总计: {result['token_usage']['total_tokens']} tokens")
    else:
        print(f"❌ 报告生成失败: {result['error']}")

    # 测试简单摘要
    print("\n" + "=" * 60)
    print("测试快速摘要功能")
    print("=" * 60)
    summary = generator.generate_simple_summary(detected_diseases)
    print(f"\n快速摘要：{summary}")


if __name__ == "__main__":
    test_generator()