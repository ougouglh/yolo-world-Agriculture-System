"""
Gradio Web 界面
提供用户友好的病害诊断交互界面
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sys
from pathlib import Path
import gradio as gr
from PIL import Image
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.diagnosis_pipeline import PlantDiseaseDiagnosisPipeline

# 全局变量：诊断流程实例
pipeline = None


def initialize_pipeline():
    """初始化诊断流程（全局单例）"""
    global pipeline

    if pipeline is None:
        print("初始化诊断系统...")

        # 配置参数
        yolo_model_path = "/root/autodl-tmp/project/runs/yoloworld/train/weights/best.pt"
        vectorstore_path = "/root/autodl-tmp/project/vectorstore/chroma_db"
        api_key = "sk-26d1261b1bd44fae92985f4cdee517e5"

        pipeline = PlantDiseaseDiagnosisPipeline(
            yolo_model_path=yolo_model_path,
            vectorstore_path=vectorstore_path,
            api_key=api_key,
            llm_model="qwen-turbo",
            confidence_threshold=0.25
        )

        print("✅ 诊断系统初始化完成！")

    return pipeline


def diagnose_image(image):
    """
    诊断上传的图像

    Args:
        image: PIL Image 或 numpy array

    Returns:
        (标注图像, 检测结果文本, 诊断报告文本)
    """
    if image is None:
        return None, "请上传图像", ""

    try:
        # 初始化流程
        pipeline = initialize_pipeline()

        # 保存临时图像
        temp_image_path = "/tmp/temp_diagnosis_image.jpg"
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(temp_image_path)
        else:
            image.save(temp_image_path)

        # 执行诊断
        result = pipeline.diagnose(
            image_path=temp_image_path,
            retrieve_top_k=3,
            return_annotated_image=True
        )

        # 处理结果
        if result['success']:
            # 标注图像
            annotated_image = Image.fromarray(result['annotated_image'])

            # 检测结果摘要
            detection_summary = f"### 🔍 检测结果\n\n"
            detection_summary += f"**检测到 {len(result['detections'])} 个病害区域**\n\n"

            for i, det in enumerate(result['detections'], 1):
                detection_summary += f"{i}. **{det['class_name']}** "
                detection_summary += f"(置信度: {det['confidence']:.1%})\n"

            # 诊断报告
            diagnosis_report = result['diagnosis_report']['report']

            # Token 使用信息
            if 'token_usage' in result['diagnosis_report']:
                usage = result['diagnosis_report']['token_usage']
                diagnosis_report += f"\n\n---\n*Token 使用: {usage['total_tokens']} tokens*"

            return annotated_image, detection_summary, diagnosis_report

        else:
            error_msg = result.get('error', '未知错误')

            if error_msg == "未检测到病害":
                return image, "### ❌ 未检测到病害\n\n图像中未发现明显的病害特征。", ""
            else:
                return image, f"### ❌ 诊断失败\n\n{error_msg}", ""

    except Exception as e:
        return None, f"### ❌ 系统错误\n\n{str(e)}", ""


def create_app():
    """创建 Gradio 应用"""

    # 自定义 CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=custom_css, title="智慧农业病虫害诊断系统") as app:
        # 标题和说明
        gr.Markdown(
            """
            # 🌾 智慧农业病虫害诊断系统

            基于 **YOLO-World + RAG + 通义千问** 的智能病害检测与诊断系统

            ### 使用说明
            1. 📤 上传作物病害图像
            2. 🔍 系统自动检测病害
            3. 📋 查看详细诊断报告
            4. 🛡️ 获取防治建议

            ---
            """
        )

        with gr.Row():
            # 左侧：图像输入和输出
            with gr.Column(scale=1):
                gr.Markdown("### 📸 图像上传与检测结果")

                image_input = gr.Image(
                    label="上传病害图像",
                    type="pil",
                    height=400
                )

                diagnose_btn = gr.Button(
                    "🔍 开始诊断",
                    variant="primary",
                    size="lg"
                )

                image_output = gr.Image(
                    label="检测结果（标注后）",
                    type="pil",
                    height=400
                )

            # 右侧：检测信息和诊断报告
            with gr.Column(scale=1):
                gr.Markdown("### 📊 诊断结果")

                detection_output = gr.Markdown(
                    label="检测摘要",
                    value="等待上传图像..."
                )

                report_output = gr.Markdown(
                    label="详细诊断报告",
                    value=""
                )

        # 示例图像
        gr.Markdown("### 💡 示例图像")
        gr.Examples(
            examples=[
                "/root/autodl-tmp/project/dataset/images/test_zeroshot/Tomato___Early_blight/00a27ff6-3fa4-492c-8c03-a2903bf1a75f___RS_Erly.B 7393.JPG",
                "/root/autodl-tmp/project/dataset/images/test_zeroshot/Tomato___Late_blight/008a2a70-ca6d-4e1e-bfc1-fadce7c9e6d7___RS_Late.B 4946.JPG",
            ],
            inputs=image_input,
            label="点击加载示例"
        )

        # 页脚
        gr.Markdown(
            """
            ---
            ### ⚙️ 技术栈
            - **检测模型**: YOLO-World (零样本检测)
            - **知识库**: ChromaDB + BGE-base-zh-v1.5
            - **大语言模型**: 通义千问 Qwen-Turbo
            - **开发框架**: Ultralytics, LangChain, Gradio

            ### 📌 系统特点
            - ✅ 零样本检测：可识别训练时未见过的新病害
            - ✅ 专业诊断：基于知识库的精准分析
            - ✅ 实用建议：提供具体的防治方案
            - ✅ 低成本部署：API 调用成本极低

            ---
            *Powered by YOLO-World + RAG + 通义千问 | 2026*
            """
        )

        # 绑定事件
        diagnose_btn.click(
            fn=diagnose_image,
            inputs=image_input,
            outputs=[image_output, detection_output, report_output]
        )

        # 也可以在上传图像后自动诊断
        image_input.change(
            fn=diagnose_image,
            inputs=image_input,
            outputs=[image_output, detection_output, report_output]
        )

    return app


def main():
    """启动应用"""
    print("=" * 60)
    print("启动智慧农业病虫害诊断系统")
    print("=" * 60)

    # 创建应用
    app = create_app()

    # 启动服务器
    app.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=6006,
        share=False,  # 生成公开链接
        show_error=True
    )


if __name__ == "__main__":
    main()