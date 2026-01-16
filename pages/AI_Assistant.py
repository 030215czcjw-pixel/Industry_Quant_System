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

# 检查是否有回测结果
has_backtest_result = 'df_backtest_result' in st.session_state and st.session_state.df_backtest_result is not None

# 布局
col_main, col_sidebar = st.columns([3, 1])

with col_sidebar:
    st.subheader("⚙️ 设置", divider="gray")

    # AI 提供商选择
    ai_provider = st.selectbox(
        "AI 提供商",
        [
            "DeepSeek",
            "智谱 AI (GLM)",
            "通义千问 (Qwen)",
            "Claude (Anthropic)",
            "OpenAI"
        ],
        help="选择要使用的 AI 服务提供商"
    )

    # 根据提供商显示不同的配置
    if ai_provider == "DeepSeek":
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.environ.get("DEEPSEEK_API_KEY", ""),
            help="输入你的 DeepSeek API Key"
        )
        model_choice = st.selectbox(
            "选择模型",
            ["deepseek-chat", "deepseek-reasoner"],
            help="deepseek-chat: 通用对话模型\ndeepseek-reasoner: 推理增强模型"
        )
        base_url = "https://api.deepseek.com"

    elif ai_provider == "智谱 AI (GLM)":
        api_key = st.text_input(
            "智谱 API Key",
            type="password",
            value=os.environ.get("ZHIPU_API_KEY", ""),
            help="输入你的智谱 API Key"
        )
        model_choice = st.selectbox(
            "选择模型",
            ["glm-4-plus", "glm-4-flash", "glm-4"],
            help="glm-4-plus: 最强性能\nglm-4-flash: 快速响应\nglm-4: 平衡版本"
        )
        base_url = "https://open.bigmodel.cn/api/paas/v4"

    elif ai_provider == "通义千问 (Qwen)":
        api_key = st.text_input(
            "通义千问 API Key",
            type="password",
            value=os.environ.get("QWEN_API_KEY", ""),
            help="输入你的通义千问 API Key"
        )
        model_choice = st.selectbox(
            "选择模型",
            ["qwen-plus", "qwen-turbo", "qwen-max"],
            help="qwen-plus: 性价比高\nqwen-turbo: 快速响应\nqwen-max: 最强性能"
        )
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    elif ai_provider == "Claude (Anthropic)":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="输入你的 Anthropic API Key"
        )
        model_choice = st.selectbox(
            "选择模型",
            ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
            help="Sonnet: 平衡性能\nOpus: 最强性能\nHaiku: 最快速度"
        )
        base_url = None  # Anthropic 使用官方 SDK

    else:  # OpenAI
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="输入你的 OpenAI API Key"
        )
        model_choice = st.selectbox(
            "选择模型",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            help="gpt-4o: 最新模型\ngpt-4-turbo: 快速版GPT-4\ngpt-3.5-turbo: 经济实惠"
        )
        base_url = "https://api.openai.com/v1"

    # 温度参数
    temperature = st.slider(
        "创造性",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="数值越高，回答越有创造性"
    )

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
    st.subheader("📊 数据状态")
    if has_backtest_result:
        df_res = st.session_state.df_backtest_result
        st.success("✅ 回测数据已加载")
        st.metric("数据行数", len(df_res))
        st.metric("起始日期", df_res.index.min().strftime('%Y-%m-%d'))
        st.metric("结束日期", df_res.index.max().strftime('%Y-%m-%d'))
    else:
        st.warning("⚠️ 暂无回测数据")
        st.info("请先在 Backtest 页面运行回测")

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
        if has_backtest_result:
            df_res = st.session_state.df_backtest_result

            # 计算关键指标
            final_nav = df_res['仓位净值'].iloc[-1]
            prior_nav = df_res['先验仓位净值'].iloc[-1]
            excess_gain = final_nav - prior_nav

            context_data = {
                "策略净值": final_nav,
                "先验净值": prior_nav,
                "超额增益": excess_gain,
                "数据行数": len(df_res),
                "起始日期": df_res.index.min().strftime('%Y-%m-%d'),
                "结束日期": df_res.index.max().strftime('%Y-%m-%d'),
                "买入信号次数": int(df_res['买入信号'].sum()),
                "信号触发次数": int(df_res['信号触发'].sum()),
                "标的": st.session_state.get('stock_chosen', '未知'),
                "基准": st.session_state.get('base_chosen', '未知'),
                "持有期": st.session_state.get('holding_period', '未知'),
                "观察期": st.session_state.get('observation_period', '未知'),
                "目标超额收益": st.session_state.get('profit_target', '未知'),
            }

            # 计算更多统计指标
            stats_data = {
                "最终策略净值": f"{final_nav:.4f}",
                "最终先验净值": f"{prior_nav:.4f}",
                "总收益率": f"{(final_nav - 1):.2%}",
                "年化收益率": f"{((final_nav ** (252 / len(df_res))) - 1):.2%}" if len(df_res) > 0 else "N/A",
                "最大回撤": f"{(df_res['仓位净值'] / df_res['仓位净值'].cummax() - 1).min():.2%}",
                "胜率": f"{(df_res[df_res['买入信号'] == 1]['持有期超额收益率'] > 0).sum() / max(df_res['买入信号'].sum(), 1):.2%}",
            }

            # 将DataFrame转换为文本
            df_text = dataframe_to_text(df_res, max_rows=30)

            system_prompt = f"""你是一个专业的量化策略分析助手。你正在帮助用户分析一个基于贝叶斯更新的择时策略回测结果。

回测配置：
- 标的：{context_data['标的']}
- 基准：{context_data['基准']}
- 回测周期：{context_data['起始日期']} 至 {context_data['结束日期']}
- 持有期：{context_data['持有期']} 天
- 观察期：{context_data['观察期']} 天
- 目标超额收益：{context_data['目标超额收益']}

关键指标：
- 策略净值：{stats_data['最终策略净值']} ({stats_data['总收益率']})
- 先验净值：{stats_data['最终先验净值']}
- 超额增益：{context_data['超额增益']:.4f}
- 年化收益率：{stats_data['年化收益率']}
- 最大回撤：{stats_data['最大回撤']}
- 胜率：{stats_data['胜率']}
- 买入信号次数：{context_data['买入信号次数']}
- 信号触发次数：{context_data['信号触发次数']}

=== 完整回测数据 ===
以下是详细的回测数据，你可以基于这些数据进行深入分析：

{df_text}

你可以根据以上完整的数据：
1. 解读回测指标的含义和具体数值
2. 分析策略在不同时间段的表现
3. 识别策略的优缺点和风险点
4. 基于数据提供优化建议
5. 回答关于贝叶斯更新机制的问题
6. 解释先验概率和后验概率的作用
7. 分析信号触发的时机和质量
8. 评估策略的稳定性和可靠性

请用简洁、专业的语言回答，必须使用具体数据支撑你的观点。"""
        else:
            system_prompt = """你是一个专业的量化策略分析助手。虽然当前没有加载回测数据，但你可以：
1. 回答关于量化策略的一般问题
2. 解释贝叶斯更新、择时策略等概念
3. 提供策略设计和优化的建议
4. 讨论风险管理和资金管理
5. 解答关于回测方法论的问题

请用专业、易懂的语言回答用户的问题。"""

        # 显示系统信息（可选）
        if st.session_state.get('show_system_info', False):
            with st.expander("📋 系统提示词", expanded=True):
                st.code(system_prompt, language="text")

        # 显示对话历史
        for message in st.session_state.ai_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 用户输入
        if has_backtest_result:
            placeholder_text = "问我任何关于这个回测的问题..."
        else:
            placeholder_text = "问我关于量化策略的问题..."

        user_question = st.chat_input(placeholder_text)

        if user_question:
            # 添加用户消息到历史
            st.session_state.ai_chat_history.append({
                "role": "user",
                "content": user_question
            })

            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(user_question)

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
with st.expander("💡 使用提示", expanded=False):
    st.markdown("""
    ### 示例问题

    **关于回测结果：**
    - "帮我分析一下这个策略的表现如何？"
    - "为什么策略净值比先验净值高/低？"
    - "这个策略的最大回撤说明了什么？"
    - "胜率这个指标如何解读？"

    **关于策略优化：**
    - "如何优化这个策略？"
    - "应该调整哪些参数？"
    - "持有期和观察期如何设置比较合理？"

    **关于贝叶斯机制：**
    - "贝叶斯更新在这个策略中是如何工作的？"
    - "先验概率和后验概率有什么区别？"
    - "为什么要使用贝叶斯方法？"

    **关于风险管理：**
    - "这个策略的风险点在哪里？"
    - "如何控制回撤？"
    - "仓位管理有什么建议？"

    **关于数据分析：**
    - "帮我分析最近30天的交易信号质量"
    - "哪些时间段策略表现最好/最差？"
    - "信号触发率是否合理？"
    - "后验概率相比先验概率提升了多少？"

    ### 模型推荐

    - **DeepSeek**: 价格最低，推理能力强，适合高频使用
    - **智谱 GLM**: 中文理解好，响应快，性价比高
    - **通义千问**: 阿里巴巴出品，稳定可靠
    - **Claude**: 推理能力最强，输出质量最高
    - **OpenAI GPT-4**: 最全面的能力，多模态支持

    ### 💾 历史对话管理

    **下载对话:**
    1. 与AI对话后，点击侧边栏"💾 下载对话"按钮
    2. 浏览器会自动下载 `chat_YYYYMMDD_HHMMSS.json` 文件
    3. 文件包含：导出时间、AI提供商、模型、完整对话记录

    **上传对话:**
    1. 展开"📂 上传历史对话"面板
    2. 点击"Browse files"选择之前下载的 JSON 文件
    3. 查看对话预览信息（时间、提供商、消息数量）
    4. 点击"📥 加载此对话"按钮恢复对话
    5. 可继续与AI交流，无缝衔接

    **适用场景:**
    - ✅ 本地运行 Streamlit
    - ✅ Streamlit Cloud 部署
    - ✅ 跨设备使用（下载后传输到其他设备）
    - ✅ 备份重要对话

    ### 📊 数据访问能力

    AI助手现在可以访问完整的回测DataFrame，包括：
    - 所有列的统计信息（均值、标准差、分位数等）
    - 最近30条详细数据记录
    - 关键指标的自动计算
    - 时间序列数据的完整信息

    这意味着AI可以：
    - 深入分析具体的交易信号
    - 识别特定时间段的表现
    - 提供基于实际数据的优化建议
    - 回答关于数据细节的问题
    """)
