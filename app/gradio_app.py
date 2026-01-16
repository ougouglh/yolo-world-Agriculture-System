"""
Gradio Web 界面 - 专业美化版
路径: app/gradio_app.py

UI 优化:
- 现代化配色方案
- 卡片式布局
- 动画效果
- 更好的信息层级
- 响应式设计
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
from pathlib import Path
import gradio as gr
from PIL import Image
import numpy as np
from typing import Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.diagnosis_pipeline import ConversationalDiagnosisPipeline

# 全局变量
pipeline: Optional[ConversationalDiagnosisPipeline] = None
current_diagnosis_result = None
diagnosis_history = []  # 诊断历史


# ========== 自定义 CSS 样式 ==========
CUSTOM_CSS = """
/* 全局样式 */
.gradio-container {
    font-family: 'Microsoft YaHei', 'PingFang SC', 'Helvetica Neue', Arial, sans-serif !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%) !important;
    min-height: 100vh;
}

/* 标题区域 */
.header-title {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.header-title h1 {
    color: white !important;
    font-size: 2.2em !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.header-title p {
    color: rgba(255,255,255,0.9) !important;
    margin: 10px 0 0 0 !important;
}

/* 卡片样式 */
.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 1px solid rgba(0,0,0,0.05);
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}

/* 状态卡片 */
.status-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    margin: 10px 0;
}

.status-card.warning {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.status-card.info {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

/* 按钮样式 */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

.secondary-btn {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%) !important;
    border: 2px solid #667eea !important;
    color: #667eea !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover {
    background: #667eea !important;
    color: white !important;
}

/* 图片上传区域 */
.image-upload {
    border: 3px dashed #667eea !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%) !important;
    transition: all 0.3s ease !important;
}

.image-upload:hover {
    border-color: #764ba2 !important;
    background: linear-gradient(135deg, #f0f4ff 0%, #e8edff 100%) !important;
}

/* 结果展示区域 */
.result-box {
    background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
    border-radius: 16px;
    padding: 20px;
    border-left: 4px solid #667eea;
}

/* 诊断报告样式 */
.report-content {
    background: white;
    border-radius: 12px;
    padding: 20px;
    line-height: 1.8;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
}

.report-content h2 {
    color: #667eea;
    border-bottom: 2px solid #f0f0f0;
    padding-bottom: 10px;
    margin-top: 20px;
}

.report-content h3 {
    color: #764ba2;
}

/* Tab 样式 */
.tabs {
    margin-top: 20px;
}

.tab-nav {
    background: white;
    border-radius: 12px 12px 0 0;
    padding: 8px;
}

.tab-nav button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* 统计数字 */
.stat-number {
    font-size: 2.5em;
    font-weight: 700;
    color: #667eea;
    line-height: 1;
}

.stat-label {
    color: #888;
    font-size: 0.9em;
    margin-top: 5px;
}

/* 动画 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeIn 0.5s ease-out;
}

/* 追问对话区域 */
.chat-container {
    background: #f8f9ff;
    border-radius: 16px;
    padding: 20px;
    max-height: 400px;
    overflow-y: auto;
}

.chat-bubble {
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 80%;
}

.chat-bubble.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: auto;
}

.chat-bubble.assistant {
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}

/* 特性标签 */
.feature-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 4px;
}

.feature-tag.green {
    background: #e6f7f1;
    color: #11998e;
}

.feature-tag.purple {
    background: #f3e8ff;
    color: #764ba2;
}

.feature-tag.blue {
    background: #e8f4ff;
    color: #4facfe;
}

/* 页脚 */
.footer {
    text-align: center;
    padding: 20px;
    color: #888;
    font-size: 0.9em;
    margin-top: 30px;
}

/* 响应式 */
@media (max-width: 768px) {
    .header-title h1 {
        font-size: 1.6em !important;
    }
}
"""


def initialize_pipeline():
    """初始化诊断流程"""
    global pipeline

    if pipeline is None:
        print("🚀 初始化诊断系统...")

        CONFIG = {
            "yolo_model_path": "/root/autodl-tmp/project/runs/yoloworld/train/weights/best.pt",
            "vectorstore_path": "/root/autodl-tmp/project/vectorstore/chroma_db",
            "api_key": "sk-26d1261b1bd44fae92985f4cdee517e5",
            "llm_model": "qwen-turbo",
            "confidence_threshold": 0.25
        }

        pipeline = ConversationalDiagnosisPipeline(
            yolo_model_path=CONFIG["yolo_model_path"],
            vectorstore_path=CONFIG["vectorstore_path"],
            api_key=CONFIG["api_key"],
            llm_model=CONFIG["llm_model"],
            confidence_threshold=CONFIG["confidence_threshold"]
        )

        print("✅ 诊断系统初始化完成！")

    return pipeline


def diagnose_image(image, progress=gr.Progress()):
    """诊断上传的图像"""
    global current_diagnosis_result, diagnosis_history

    if image is None:
        return (
            None,
            create_status_html("warning", "⚠️ 请上传图像", "请选择一张作物病害图像进行诊断"),
            ""
        )

    try:
        progress(0.1, desc="🚀 初始化系统...")
        pipe = initialize_pipeline()

        # 保存临时图像
        temp_image_path = "/tmp/temp_diagnosis_image.jpg"
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(temp_image_path)
        else:
            image.save(temp_image_path)

        def progress_callback(msg):
            if "检测" in msg:
                progress(0.3, desc="🔍 " + msg)
            elif "检索" in msg:
                progress(0.5, desc="📚 " + msg)
            elif "生成" in msg:
                progress(0.7, desc="📝 " + msg)
            elif "完成" in msg:
                progress(0.9, desc="✅ " + msg)

        progress(0.2, desc="🔍 正在检测病害...")
        result = pipe.diagnose(
            image_path=temp_image_path,
            retrieve_top_k=3,
            return_annotated_image=True,
            progress_callback=progress_callback
        )

        current_diagnosis_result = result
        progress(1.0, desc="✅ 诊断完成！")

        if result['success']:
            # 保存到历史记录
            diagnosis_history.append({
                'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'diseases': result['disease_names'],
                'count': len(result['detections'])
            })

            annotated_image = Image.fromarray(result['annotated_image'])

            # 构建检测结果 HTML
            detection_html = create_detection_html(result)

            # 诊断报告
            report = result['diagnosis_report']['report']

            return annotated_image, detection_html, report

        else:
            error_msg = result.get('error', '未知错误')
            if error_msg == "未检测到病害":
                return (
                    image,
                    create_status_html("info", "🔍 未检测到病害", "图像中未发现明显的病害特征，您的作物看起来很健康！"),
                    ""
                )
            else:
                return (
                    image,
                    create_status_html("warning", "❌ 诊断失败", error_msg),
                    ""
                )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            None,
            create_status_html("warning", "❌ 系统错误", str(e)),
            ""
        )


def create_status_html(status_type, title, message):
    """创建状态卡片 HTML"""
    colors = {
        "success": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "warning": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "info": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
    }
    return f"""
    <div style="background: {colors.get(status_type, colors['info'])}; 
                color: white; 
                padding: 20px 24px; 
                border-radius: 12px; 
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 8px 0; font-size: 1.2em;">{title}</h3>
        <p style="margin: 0; opacity: 0.95;">{message}</p>
    </div>
    """


def create_detection_html(result):
    """创建检测结果 HTML"""
    detections = result['detections']
    diseases = result['disease_names']

    # 严重程度判断
    severity = "轻度" if len(detections) <= 2 else ("中度" if len(detections) <= 5 else "重度")
    severity_color = "#11998e" if severity == "轻度" else ("#f5a623" if severity == "中度" else "#f5576c")

    html = f"""
    <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        
        <!-- 统计概览 -->
        <div style="display: flex; justify-content: space-around; margin-bottom: 20px; text-align: center;">
            <div>
                <div style="font-size: 2.5em; font-weight: 700; color: #667eea;">{len(detections)}</div>
                <div style="color: #888; font-size: 0.9em;">检测区域</div>
            </div>
            <div>
                <div style="font-size: 2.5em; font-weight: 700; color: #764ba2;">{len(diseases)}</div>
                <div style="color: #888; font-size: 0.9em;">病害类型</div>
            </div>
            <div>
                <div style="font-size: 2.5em; font-weight: 700; color: {severity_color};">{severity}</div>
                <div style="color: #888; font-size: 0.9em;">严重程度</div>
            </div>
        </div>
        
        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
        
        <!-- 病害详情 -->
        <h4 style="color: #333; margin: 0 0 15px 0;">🦠 检测到的病害</h4>
    """

    for i, det in enumerate(detections, 1):
        confidence = det['confidence']
        conf_color = "#11998e" if confidence > 0.8 else ("#f5a623" if confidence > 0.5 else "#f5576c")

        html += f"""
        <div style="background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); 
                    border-radius: 10px; 
                    padding: 12px 16px; 
                    margin: 8px 0;
                    border-left: 4px solid {conf_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #333;">
                    {i}. {det['class_name']}
                </span>
                <span style="background: {conf_color}; 
                             color: white; 
                             padding: 4px 12px; 
                             border-radius: 20px; 
                             font-size: 0.85em;
                             font-weight: 600;">
                    {confidence:.1%}
                </span>
            </div>
        </div>
        """

    # 知识库检索信息
    html += f"""
        <div style="margin-top: 20px; padding: 12px; background: #f0f4ff; border-radius: 8px;">
            <span style="color: #667eea;">📚 知识库匹配: </span>
            <span style="color: #666;">{len(result['knowledge_retrieval'])} 条相关记录</span>
        </div>
    </div>
    """

    return html


def ask_followup_question(question: str) -> str:
    """追问功能"""
    global pipeline

    if pipeline is None:
        return "⚠️ 请先进行一次诊断"

    if not question.strip():
        return "⚠️ 请输入您的问题"

    try:
        answer = pipeline.ask_followup(question)
        return answer
    except Exception as e:
        return f"❌ 回答失败: {str(e)}"


def clear_conversation():
    """清空对话"""
    global current_diagnosis_result, pipeline

    current_diagnosis_result = None
    if pipeline:
        pipeline.clear_history()

    return "✅ 对话已清空，请重新上传图像进行诊断", ""


def get_system_stats():
    """获取系统统计信息"""
    global diagnosis_history

    if pipeline is None:
        return create_stats_html({}, diagnosis_history)

    try:
        stats = pipeline.get_statistics()
        return create_stats_html(stats, diagnosis_history)
    except Exception as e:
        return f"❌ 获取统计失败: {e}"


def create_stats_html(stats, history):
    """创建统计信息 HTML"""
    total_docs = stats.get('total_documents', 0)
    diseases = stats.get('diseases', [])
    crops = stats.get('crops', [])

    html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;">
        
        <!-- 知识库统计 -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 24px; 
                    border-radius: 16px;
                    text-align: center;">
            <div style="font-size: 2.5em; font-weight: 700;">{total_docs}</div>
            <div style="opacity: 0.9;">知识库文档</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    color: white; 
                    padding: 24px; 
                    border-radius: 16px;
                    text-align: center;">
            <div style="font-size: 2.5em; font-weight: 700;">{len(diseases)}</div>
            <div style="opacity: 0.9;">支持病害</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; 
                    padding: 24px; 
                    border-radius: 16px;
                    text-align: center;">
            <div style="font-size: 2.5em; font-weight: 700;">{len(crops)}</div>
            <div style="opacity: 0.9;">作物种类</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    color: white; 
                    padding: 24px; 
                    border-radius: 16px;
                    text-align: center;">
            <div style="font-size: 2.5em; font-weight: 700;">{len(history)}</div>
            <div style="opacity: 0.9;">今日诊断</div>
        </div>
        
    </div>
    
    <!-- 支持的病害列表 -->
    <div style="background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h4 style="color: #333; margin: 0 0 15px 0;">🦠 支持的病害类型</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
    """

    for disease in diseases[:15]:
        html += f"""
            <span style="background: #f3e8ff; color: #764ba2; padding: 6px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 500;">
                {disease}
            </span>
        """

    if len(diseases) > 15:
        html += f"""
            <span style="background: #eee; color: #666; padding: 6px 14px; border-radius: 20px; font-size: 0.85em;">
                +{len(diseases) - 15} 更多
            </span>
        """

    html += """
        </div>
    </div>
    
    <!-- 支持的作物 -->
    <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h4 style="color: #333; margin: 0 0 15px 0;">🌱 支持的作物</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
    """

    for crop in crops:
        html += f"""
            <span style="background: #e6f7f1; color: #11998e; padding: 6px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 500;">
                {crop}
            </span>
        """

    html += """
        </div>
    </div>
    """

    return html


def create_app():
    """创建 Gradio 应用"""

    with gr.Blocks(css=CUSTOM_CSS, title="智慧农业病虫害诊断系统", theme=gr.themes.Soft()) as app:

        # ========== 头部 ==========
        gr.HTML("""
        <div class="header-title">
            <h1>🌾 智慧农业病虫害诊断系统</h1>
            <p>基于 YOLO-World + RAG + 通义千问 的智能诊断平台</p>
            <div style="margin-top: 15px;">
                <span class="feature-tag green">✓ 零样本检测</span>
                <span class="feature-tag purple">✓ 专业诊断</span>
                <span class="feature-tag blue">✓ 智能问答</span>
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ========== Tab 1: 病害诊断 ==========
            with gr.TabItem("🔍 病害诊断", id=1):
                with gr.Row(equal_height=True):

                    # 左侧：图像上传
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="margin-bottom: 15px;">
                            <h3 style="color: #333; margin: 0;">📸 上传图像</h3>
                            <p style="color: #888; font-size: 0.9em; margin: 5px 0 0 0;">支持 JPG、PNG 格式的作物病害图像</p>
                        </div>
                        """)

                        image_input = gr.Image(
                            label="",
                            type="pil",
                            height=380,
                            elem_classes=["image-upload"]
                        )

                        diagnose_btn = gr.Button(
                            "🚀 开始智能诊断",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"]
                        )

                        gr.HTML("""
                        <div style="margin-top: 15px;">
                            <h3 style="color: #333; margin: 0 0 10px 0;">🎯 检测结果</h3>
                        </div>
                        """)

                        image_output = gr.Image(
                            label="",
                            type="pil",
                            height=380
                        )

                    # 右侧：诊断结果
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="margin-bottom: 15px;">
                            <h3 style="color: #333; margin: 0;">📊 诊断分析</h3>
                            <p style="color: #888; font-size: 0.9em; margin: 5px 0 0 0;">AI 智能识别病害并给出专业建议</p>
                        </div>
                        """)

                        detection_output = gr.HTML(
                            value=create_status_html("info", "👆 请上传图像", "选择一张作物病害图像开始诊断")
                        )

                        gr.HTML("""
                        <div style="margin: 20px 0 10px 0;">
                            <h3 style="color: #333; margin: 0;">📋 详细诊断报告</h3>
                        </div>
                        """)

                        report_output = gr.Markdown(
                            value="",
                            elem_classes=["report-content"]
                        )

            # ========== Tab 2: 智能问答 ==========
            with gr.TabItem("💬 智能问答", id=2):
                gr.HTML("""
                <div style="background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); 
                            border-radius: 16px; 
                            padding: 24px; 
                            margin-bottom: 20px;">
                    <h3 style="color: #333; margin: 0 0 10px 0;">🤖 AI 农业专家助手</h3>
                    <p style="color: #666; margin: 0;">完成诊断后，您可以针对诊断结果进行追问，获取更详细的防治建议。</p>
                    <div style="margin-top: 15px; display: flex; flex-wrap: wrap; gap: 10px;">
                        <span style="background: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9em; color: #667eea; cursor: pointer;">
                            💡 这种病害什么季节容易发生？
                        </span>
                        <span style="background: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9em; color: #667eea; cursor: pointer;">
                            💡 有没有生物防治方法？
                        </span>
                        <span style="background: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9em; color: #667eea; cursor: pointer;">
                            💡 喷药后多久可以采摘？
                        </span>
                    </div>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=5):
                        question_input = gr.Textbox(
                            label="",
                            placeholder="请输入您的问题，例如：这种病害如何预防？",
                            lines=2,
                            max_lines=4
                        )
                    with gr.Column(scale=1, min_width=120):
                        ask_btn = gr.Button("📤 发送", variant="primary", elem_classes=["primary-btn"])
                        clear_btn = gr.Button("🗑️ 清空", elem_classes=["secondary-btn"])

                gr.HTML("""<div style="margin: 15px 0 10px 0;"><h4 style="color: #333; margin: 0;">💬 AI 回答</h4></div>""")

                answer_output = gr.Markdown(
                    value="*等待您的提问...*",
                    elem_classes=["report-content"]
                )

                clear_status = gr.Markdown(value="")

            # ========== Tab 3: 系统统计 ==========
            with gr.TabItem("📊 系统统计", id=3):
                gr.HTML("""
                <div style="margin-bottom: 20px;">
                    <h3 style="color: #333; margin: 0 0 5px 0;">📊 知识库与系统统计</h3>
                    <p style="color: #888; font-size: 0.9em; margin: 0;">查看系统支持的病害类型和诊断统计</p>
                </div>
                """)

                stats_btn = gr.Button("🔄 刷新统计数据", elem_classes=["secondary-btn"])

                stats_output = gr.HTML(value=create_stats_html({}, []))

            # ========== Tab 4: 使用帮助 ==========
            with gr.TabItem("❓ 使用帮助", id=4):
                gr.HTML("""
                <div style="background: white; border-radius: 16px; padding: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                    
                    <h3 style="color: #667eea; margin-top: 0;">📖 使用指南</h3>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 20px 0;">
                        
                        <div style="background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #667eea;">
                            <h4 style="color: #667eea; margin: 0 0 10px 0;">1️⃣ 上传图像</h4>
                            <p style="color: #666; margin: 0; font-size: 0.95em;">拍摄或选择一张作物病害图像，支持叶片、茎秆、果实等部位的病害照片。</p>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #764ba2;">
                            <h4 style="color: #764ba2; margin: 0 0 10px 0;">2️⃣ 智能诊断</h4>
                            <p style="color: #666; margin: 0; font-size: 0.95em;">点击"开始诊断"，AI 将自动识别病害类型、严重程度，并生成专业报告。</p>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #11998e;">
                            <h4 style="color: #11998e; margin: 0 0 10px 0;">3️⃣ 追问咨询</h4>
                            <p style="color: #666; margin: 0; font-size: 0.95em;">如有疑问，可在"智能问答"中继续提问，AI 专家将为您详细解答。</p>
                        </div>
                        
                    </div>
                    
                    <hr style="border: none; border-top: 1px solid #eee; margin: 25px 0;">
                    
                    <h3 style="color: #667eea;">🛠️ 技术架构</h3>
                    
                    <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                        <tr style="background: #f8f9ff;">
                            <td style="padding: 12px; border: 1px solid #eee; font-weight: 600; color: #667eea;">目标检测</td>
                            <td style="padding: 12px; border: 1px solid #eee;">YOLO-World (零样本检测)</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #eee; font-weight: 600; color: #667eea;">知识库</td>
                            <td style="padding: 12px; border: 1px solid #eee;">ChromaDB 向量数据库</td>
                        </tr>
                        <tr style="background: #f8f9ff;">
                            <td style="padding: 12px; border: 1px solid #eee; font-weight: 600; color: #667eea;">语义检索</td>
                            <td style="padding: 12px; border: 1px solid #eee;">BGE-base-zh Embedding</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #eee; font-weight: 600; color: #667eea;">报告生成</td>
                            <td style="padding: 12px; border: 1px solid #eee;">通义千问 Qwen-Turbo</td>
                        </tr>
                    </table>
                    
                    <div style="margin-top: 25px; padding: 15px; background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%); border-radius: 10px;">
                        <p style="margin: 0; color: #666; font-size: 0.95em;">
                            <strong>💡 提示：</strong>为获得最佳诊断效果，请确保图像清晰、光线充足，病害部位在画面中清晰可见。
                        </p>
                    </div>
                    
                </div>
                """)

        # ========== 示例图像 ==========
        gr.HTML("""
        <div style="margin-top: 25px;">
            <h4 style="color: #333; margin: 0 0 15px 0;">💡 示例图像（点击加载）</h4>
        </div>
        """)

        gr.Examples(
            examples=[
                "/root/autodl-tmp/project/dataset/images/test_zeroshot/Tomato___Early_blight/00a27ff6-3fa4-492c-8c03-a2903bf1a75f___RS_Erly.B 7393.JPG",
                "/root/autodl-tmp/project/dataset/images/test_zeroshot/Tomato___Late_blight/008a2a70-ca6d-4e1e-bfc1-fadce7c9e6d7___RS_Late.B 4946.JPG",
            ],
            inputs=image_input,
            label=""
        )

        # ========== 页脚 ==========
        gr.HTML("""
        <div class="footer">
            <p>Powered by <strong>YOLO-World</strong> + <strong>RAG</strong> + <strong>通义千问</strong></p>
            <p style="font-size: 0.85em; color: #aaa;">© 2026 智慧农业病虫害诊断系统</p>
        </div>
        """)

        # ========== 事件绑定 ==========
        diagnose_btn.click(
            fn=diagnose_image,
            inputs=image_input,
            outputs=[image_output, detection_output, report_output]
        )

        image_input.change(
            fn=diagnose_image,
            inputs=image_input,
            outputs=[image_output, detection_output, report_output]
        )

        ask_btn.click(
            fn=ask_followup_question,
            inputs=question_input,
            outputs=answer_output
        )

        clear_btn.click(
            fn=clear_conversation,
            outputs=[clear_status, answer_output]
        )

        stats_btn.click(
            fn=get_system_stats,
            outputs=stats_output
        )

    return app


def main():
    print("=" * 60)
    print("🌾 启动智慧农业病虫害诊断系统 (美化版)")
    print("=" * 60)

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()