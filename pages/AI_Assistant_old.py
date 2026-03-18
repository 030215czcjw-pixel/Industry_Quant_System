import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from openai import OpenAI

st.set_page_config(
    page_title="AI 策略助手",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 AI 策略分析助手")
st.caption("支持 Claude、DeepSeek、智谱等多种大模型")

# 检测是否在 Streamlit Cloud 运行
def is_streamlit_cloud():
    """检测是否在 Streamlit Cloud 环境运行"""
    return os.getenv('STREAMLIT_SHARING_MODE') is not None or \
           os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'

# 生成对话历史的 JSON 数据
def generate_chat_json():
    """生成可下载的对话历史 JSON 字符串"""
    if st.session_state.ai_chat_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            'timestamp': timestamp,
            'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'provider': st.session_state.get('current_provider', 'Unknown'),
            'model': st.session_state.get('current_model', 'Unknown'),
            'messages': st.session_state.ai_chat_history,
            'message_count': len(st.session_state.ai_chat_history)
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    return None

# 从上传的 JSON 加载对话历史
def load_chat_from_json(json_str):
    """从 JSON 字符串加载对话历史"""
    try:
        data = json.loads(json_str)
        return data.get('messages', [])
    except Exception as e:
        st.error(f"加载对话历史失败: {e}")
        return []

# 将DataFrame转换为结构化的文本描述
def dataframe_to_text(df, max_rows=20):
    """将DataFrame转换为AI可读的文本格式"""
    if df is None or df.empty:
        return "数据为空"

    text_parts = []

    # 1. 基本信息
    text_parts.append(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    text_parts.append(f"时间范围: {df.index.min()} 至 {df.index.max()}")
    text_parts.append(f"\n列名: {', '.join(df.columns.tolist())}")

    # 2. 统计摘要
    text_parts.append("\n\n=== 数据统计摘要 ===")
    desc = df.describe()
    text_parts.append(desc.to_string())

    # 3. 最近的数据样本
    text_parts.append(f"\n\n=== 最近 {min(max_rows, len(df))} 条数据 ===")
    recent_data = df.tail(max_rows)
    text_parts.append(recent_data.to_string())

    # 4. 关键指标（如果存在）
    key_metrics = {}
    if '仓位净值' in df.columns:
        key_metrics['最终策略净值'] = df['仓位净值'].iloc[-1]
        key_metrics['策略最大值'] = df['仓位净值'].max()
        key_metrics['策略最小值'] = df['仓位净值'].min()

    if '先验仓位净值' in df.columns:
        key_metrics['最终先验净值'] = df['先验仓位净值'].iloc[-1]

    if '买入信号' in df.columns:
        key_metrics['总买入信号次数'] = int(df['买入信号'].sum())
        key_metrics['信号触发率'] = f"{(df['买入信号'].sum() / len(df)):.2%}"

    if '持有期超额收益率' in df.columns:
        wins = df[df['买入信号'] == 1]['持有期超额收益率'] > 0
        if wins.sum() > 0:
            key_metrics['胜率'] = f"{(wins.sum() / df['买入信号'].sum()):.2%}"

    if key_metrics:
        text_parts.append("\n\n=== 关键指标 ===")
        for key, value in key_metrics.items():
            text_parts.append(f"{key}: {value}")

    return '\n'.join(text_parts)

# 初始化 session state
if 'ai_chat_history' not in st.session_state:
    st.session_state.ai_chat_history = []

# 布局
col_main, col_sidebar = st.columns([3, 1])

with col_sidebar:
    st.subheader("⚙️ 设置", divider="gray")

    
    # # AI 提供商选择
    # ai_provider = st.selectbox(
    #     "AI 提供商",
    #     [
    #         "DeepSeek",
    #         "智谱 AI (GLM)",
    #         "通义千问 (Qwen)",
    #         "Claude (Anthropic)",
    #         "OpenAI"
    #     ],
    #     help="选择要使用的 AI 服务提供商"
    # )

    # # 根据提供商显示不同的配置
    # if ai_provider == "DeepSeek":
    #     api_key = st.text_input(
    #         "DeepSeek API Key",
    #         type="password",
    #         value=os.environ.get("DEEPSEEK_API_KEY", ""),
    #         help="输入你的 DeepSeek API Key"
    #     )
    #     model_choice = st.selectbox(
    #         "选择模型",
    #         ["deepseek-chat", "deepseek-reasoner"],
    #         help="deepseek-chat: 通用对话模型\ndeepseek-reasoner: 推理增强模型"
    #     )
    #     base_url = "https://api.deepseek.com"

    # elif ai_provider == "智谱 AI (GLM)":
    #     api_key = st.text_input(
    #         "智谱 API Key",
    #         type="password",
    #         value=os.environ.get("ZHIPU_API_KEY", ""),
    #         help="输入你的智谱 API Key"
    #     )
    #     model_choice = st.selectbox(
    #         "选择模型",
    #         ["glm-4-plus", "glm-4-flash", "glm-4"],
    #         help="glm-4-plus: 最强性能\nglm-4-flash: 快速响应\nglm-4: 平衡版本"
    #     )
    #     base_url = "https://open.bigmodel.cn/api/paas/v4"

    # elif ai_provider == "通义千问 (Qwen)":
    #     api_key = st.text_input(
    #         "通义千问 API Key",
    #         type="password",
    #         value=os.environ.get("QWEN_API_KEY", ""),
    #         help="输入你的通义千问 API Key"
    #     )
    #     model_choice = st.selectbox(
    #         "选择模型",
    #         ["qwen-plus", "qwen-turbo", "qwen-max"],
    #         help="qwen-plus: 性价比高\nqwen-turbo: 快速响应\nqwen-max: 最强性能"
    #     )
    #     base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # elif ai_provider == "Claude (Anthropic)":
    #     api_key = st.text_input(
    #         "Anthropic API Key",
    #         type="password",
    #         value=os.environ.get("ANTHROPIC_API_KEY", ""),
    #         help="输入你的 Anthropic API Key"
    #     )
    #     model_choice = st.selectbox(
    #         "选择模型",
    #         ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
    #         help="Sonnet: 平衡性能\nOpus: 最强性能\nHaiku: 最快速度"
    #     )
    #     base_url = None  # Anthropic 使用官方 SDK

    # else:  # OpenAI
    #     api_key = st.text_input(
    #         "OpenAI API Key",
    #         type="password",
    #         value=os.environ.get("OPENAI_API_KEY", ""),
    #         help="输入你的 OpenAI API Key"
    #     )
    #     model_choice = st.selectbox(
    #         "选择模型",
    #         ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    #         help="gpt-4o: 最新模型\ngpt-4-turbo: 快速版GPT-4\ngpt-3.5-turbo: 经济实惠"
    #     )
    #     base_url = "https://api.openai.com/v1"
    

    # 温度参数
    temperature = st.slider(
        "创造性",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="数值越高，回答越有创造性"
    )

    ai_provider = "DeepSeek"
    api_key = ''
    model_choice = st.selectbox(
        "选择模型",
        ["deepseek-chat", "deepseek-reasoner"],
        help="deepseek-chat: 通用对话模型\ndeepseek-reasoner: 推理增强模型"
    )
    base_url = "https://api.deepseek.com"

    st.divider()

    # 保存当前提供商和模型
    st.session_state.current_provider = ai_provider
    st.session_state.current_model = model_choice

    # 操作按钮
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        # 下载对话按钮
        chat_json = generate_chat_json()
        if chat_json:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="💾 下载对话",
                data=chat_json,
                file_name=f"chat_{timestamp}.json",
                mime="application/json",
                use_container_width=True,
                help="将当前对话下载到本地"
            )
        else:
            st.button("💾 下载对话", disabled=True, use_container_width=True, help="暂无对话可下载")

    with col_btn2:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.ai_chat_history = []
            st.rerun()

    # 上传并加载历史对话
    with st.expander("📂 上传历史对话", expanded=False):
        uploaded_file = st.file_uploader(
            "选择对话文件（JSON格式）",
            type=['json'],
            help="上传之前下载的对话记录",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                # 读取上传的文件
                json_str = uploaded_file.read().decode('utf-8')

                # 显示预览信息
                preview_data = json.loads(json_str)
                st.info(f"""
                **📋 对话预览**
                - 导出时间: {preview_data.get('export_time', '未知')}
                - AI提供商: {preview_data.get('provider', '未知')}
                - 模型: {preview_data.get('model', '未知')}
                - 消息数量: {preview_data.get('message_count', 0)} 条
                """)

                if st.button("📥 加载此对话", use_container_width=True, type="primary"):
                    loaded_history = load_chat_from_json(json_str)
                    if loaded_history:
                        st.session_state.ai_chat_history = loaded_history
                        st.success(f"✅ 已加载 {len(loaded_history)} 条对话")
                        st.rerun()
                    else:
                        st.error("❌ 对话数据为空或格式错误")
            except Exception as e:
                st.error(f"❌ 文件解析失败: {e}")

    if st.button("📋 查看系统信息", use_container_width=True):
        st.session_state.show_system_info = not st.session_state.get('show_system_info', False)

    st.divider()

    # 回测数据状态
    st.subheader("特征池：")
    use_feature_pool = ('feature_pool' in st.session_state) and (not st.session_state.feature_pool.empty)

    if not use_feature_pool:
        if not ('features' in st.session_state):
            st.warning("⚠️ 请先在 特征 页面生成特征或添加特征到特征池。")
            st.stop()
    if use_feature_pool:
        feature_columns = st.session_state.feature_pool.columns.tolist()
        feature_source = "特征池"
    else:
        feature_columns = st.session_state.features.columns.tolist() if st.session_state.features is not None else []
        feature_source = "单个特征"

with col_main:
    if not api_key:
        st.warning(f"⚠️ 请在右侧输入 {ai_provider} API Key 以使用 AI 助手")

        # 显示获取方法
        if ai_provider == "DeepSeek":
            st.info("""
            ### 如何获取 DeepSeek API Key？
            1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
            2. 注册/登录账号
            3. 在 API Keys 页面创建新的 API Key
            4. 将 Key 粘贴到右侧输入框

            **优势**: 价格极低，推理能力强，适合中文场景
            """)
        elif ai_provider == "智谱 AI (GLM)":
            st.info("""
            ### 如何获取智谱 API Key？
            1. 访问 [智谱 AI 开放平台](https://open.bigmodel.cn/)
            2. 注册/登录账号
            3. 在 API Keys 页面创建新的 API Key
            4. 将 Key 粘贴到右侧输入框

            **优势**: 国产模型，中文理解好，性价比高
            """)
        elif ai_provider == "通义千问 (Qwen)":
            st.info("""
            ### 如何获取通义千问 API Key？
            1. 访问 [阿里云百炼平台](https://dashscope.aliyun.com/)
            2. 注册/登录账号
            3. 在 API Key 管理页面创建新的 Key
            4. 将 Key 粘贴到右侧输入框

            **优势**: 阿里巴巴出品，稳定可靠，多语言支持
            """)
        elif ai_provider == "Claude (Anthropic)":
            st.info("""
            ### 如何获取 Anthropic API Key？
            1. 访问 [Anthropic Console](https://console.anthropic.com/)
            2. 注册/登录账号
            3. 创建 API Key
            4. 将 Key 粘贴到右侧输入框

            **优势**: 推理能力强，安全性高，输出质量好
            """)
        else:  # OpenAI
            st.info("""
            ### 如何获取 OpenAI API Key？
            1. 访问 [OpenAI Platform](https://platform.openai.com/)
            2. 注册/登录账号
            3. 在 API Keys 页面创建新的 Key
            4. 将 Key 粘贴到右侧输入框

            **优势**: 最强大的通用模型，多模态能力
            """)
    else:
        # 准备系统提示词
        if feature_columns:

            system_prompt = f"""
            # Role
你是一名精通 Python 和金融工程的量化开发专家。你的任务是将用户选择的因子池进行筛选并转化为符合特定回测框架的 pandas 代码。

# Context & Constraints
1. **数据环境**：你操作的对象是一个名为 `df` 的 DataFrame。其索引（Index）为日期，列（Columns）包含用户选中的因子池。
2. **输出限制**：
   - **只输出 Python 代码**，严禁包含任何 Markdown 格式（如 ```python）、解释文字或注释。
   - 生成做多逻辑与做空逻辑，如果没有逻辑则生成0
   - 最后生成一个信号列
3. **技术栈**：仅允许使用 `pandas` 和 `numpy` (已预导入为 `pd` 和 `np`)的操作。

# 以下是因子池名称：
{feature_columns}
# 回测标的为：
{st.session_state.stock_chosen}

# Logic Template
必须严格遵循以下逻辑步骤：
1. 分析哪些因子是可以组合成交易逻辑的，哪些因子是不能组合成交易逻辑的。
2. 生成数个交易逻辑，每个交易逻辑包含做多逻辑与做空逻辑。
3. 给出每个交易逻辑的逻辑解释
4. 数量不限，越多越好，可以从简单到复杂。
5. 每句话必须在新的一行
6. 输出必须是一个合法的 JSON 数组，严禁包含任何 Markdown 代码块标签（如 ```json）或解释性文字。格式如下：
[
  {{
    "strategy_name": "策略名称",
    "long_condition": "df['因子A'] > 0",
    "short_condition": "df['因子A'] < 0",
    "description": "策略逻辑解释"
  }}
]

# Example Input/Output
- 用户选择：['库存_原始数据', '基差_原始数据']
- 你的输出:
[
  {{
    "strategy_name": "期货结构策略",
    "long_condition": "df['期货结构（近月-远月）_原始数据'] > 0",
    "short_condition": "df['期货结构（近月-远月）_原始数据'] < 0",
    "description": "Back结构做多，Contango结构做空"
  }},
  {{
    "strategy_name": "库存趋势策略",
    "long_condition": "df['煤矿+港口+口岸库存_原始数据'].diff() < 0",
    "short_condition": "df['煤矿+港口+口岸库存_原始数据'].diff() > 0",
    "description": "库存去化做多，库存累积做空"
  }},
  {{
    "strategy_name": "库存与基差共振策略",
    "long_conditions": "(df['库存_原始数据'].diff() < 0) & (df['基差_原始数据'] > 0)",
    "short_conditions": "0",
    "description": "当库存去化且基差为正（现货升水）时做多；不设做空逻辑。"
  }}
]
"""

        # 显示系统信息（可选）
        if st.session_state.get('show_system_info', False):
            with st.expander("📋 系统提示词", expanded=True):
                st.code(system_prompt, language="text")


        if st.button("生成"):
            # 添加用户消息到历史
            # 调用 AI API
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    # 使用 OpenAI 兼容的 API（适用于 DeepSeek、智谱、通义等）
                    if ai_provider != "Claude (Anthropic)":
                        client = OpenAI(
                            api_key=api_key,
                            base_url=base_url
                        )

                        # 构建消息历史
                        messages = [{"role": "system", "content": system_prompt}]
                        for msg in st.session_state.ai_chat_history:
                            messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })

                        # 流式调用
                        stream = client.chat.completions.create(
                            model=model_choice,
                            messages=messages,
                            temperature=temperature,
                            stream=True
                        )

                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")

                    else:  # Claude 使用官方 SDK
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)

                        # 构建消息历史
                        messages = []
                        for msg in st.session_state.ai_chat_history:
                            messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })

                        # 流式调用
                        with client.messages.stream(
                            model=model_choice,
                            max_tokens=2048,
                            temperature=temperature,
                            system=system_prompt,
                            messages=messages,
                        ) as stream:
                            for text in stream.text_stream:
                                full_response += text
                                message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    # 添加助手回复到历史
                    st.session_state.ai_chat_history.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    # 重新运行脚本，确保下载按钮和其他组件更新
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ API 调用失败: {str(e)}")
                    st.info(f"""
                    请检查：
                    1. {ai_provider} API Key 是否正确
                    2. 网络连接是否正常
                    3. API 配额是否充足
                    4. 模型名称是否正确

                    错误详情: {type(e).__name__}
                    """)

# 页面底部说明
st.divider()
