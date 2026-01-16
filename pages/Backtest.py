import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from code_editor import code_editor
import ast

def execute_and_return(code, context):
    if not code.strip():
        return None
    try:
        # 使用 ast 解析整个代码块
        tree = ast.parse(code)
        if not tree.body:
            return None

        # 分离出最后一行
        last_node = tree.body[-1]

        # 编译并执行除最后一行之外的所有代码
        if len(tree.body) > 1:
            pre_code = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(pre_code, filename="<ast>", mode="exec"), context, context)

        # 处理最后一行：如果是表达式则返回结果，否则执行
        if isinstance(last_node, ast.Expr):
            last_expr = ast.Expression(body=last_node.value)
            return eval(compile(last_expr, filename="<ast>", mode="eval"), context, context)
        else:
            # 如果最后一行是赋值语句，尝试返回赋值的变量
            if isinstance(last_node, ast.Assign) and len(last_node.targets) == 1:
                # 执行赋值语句
                last_stmt = ast.Module(body=[last_node], type_ignores=[])
                exec(compile(last_stmt, filename="<ast>", mode="exec"), context, context)
                # 获取赋值的变量名并返回其值
                target = last_node.targets[0]
                if isinstance(target, ast.Name):
                    return context.get(target.id, None)
            # 其他语句直接执行并返回 None
            last_stmt = ast.Module(body=[last_node], type_ignores=[])
            exec(compile(last_stmt, filename="<ast>", mode="exec"), context, context)
            return None
    except Exception as e:
        return f"Error: {e}"

# 贝叶斯策略回测器
class BayesianStrategyBacktester:           
    def __init__(self, stock_data, baseline_data, feature_data, profit_setted, observation_periods, holding_period, position_strategy):
        """
        初始化回测器，执行数据对齐和基础收益率计算。
        """
        self.profit_setted = profit_setted
        self.observation_periods = observation_periods
        self.holding_period = holding_period
        self.position_strategy = position_strategy
        
        # 1. 数据对齐 (Intersection)
        common_dates = stock_data.index.intersection(baseline_data.index).intersection(feature_data.index).sort_values()
        
        # 保存原始数据副本，以便后续使用
        self.feature_data_aligned = feature_data.loc[common_dates].copy()
        
        # 2. 构建基础价格DataFrame
        self.df = pd.DataFrame({
            '股价': stock_data.loc[common_dates, '收盘'],
            '基准': baseline_data.loc[common_dates, 'close'], 
        }, index=common_dates)
        
        # 3. 计算收益率指标 (预处理)
        self.df['股价收益率'] = self.df['股价'].pct_change()
        self.df['基准收益率'] = self.df['基准'].pct_change()
        self.df['超额收益率'] = self.df['股价收益率'] - self.df['基准收益率']
        
        # 计算超额净值曲线
        self.df['超额净值'] = (1 + self.df['超额收益率'].fillna(0)).cumprod()
        
        # 计算未来持有期收益率 (Label)
        # 注意：这里shift是负数，表示读取未来的数据作为当前的标签
        self.df['持有期超额收益率'] = self.df['超额净值'].shift(-holding_period) / self.df['超额净值'] - 1

    def run_strategy(self, feature_cols, strategy_expression, position_strategy):
        """
        执行贝叶斯分析和信号生成
        :param feature_cols: list, 参与计算的特征列名
        :param strategy_expression: str, 策略触发条件的字符串表达式 (例如: "df['RSI'] > 70")
        :return: DataFrame, 包含完整分析结果
        """
        # 使用副本以免污染原始数据
        df = self.df.copy()
        
        # 合并指定的特征列
        for col in feature_cols:
            if col in self.feature_data_aligned.columns:
                df[col] = self.feature_data_aligned[col]
            else:
                print(f"警告: 特征 {col} 不存在于特征数据中")

        # 1. 定义胜率 (Prior Label)
        df['胜率触发'] = (df['持有期超额收益率'] > self.profit_setted).astype(int)
        df['胜率不触发'] = 1 - df['胜率触发']

        # 2. 计算先验概率 P(W) - 使用滚动窗口
        # shift(holding_period) 是为了防止未来函数，确保只用过去的数据计算当前的先验
        df['P(W)'] = df['胜率触发'].rolling(window=self.observation_periods).mean().shift(self.holding_period + 1)
    
        # 3. 执行策略表达式，计算信号 C
        try:
            # 准备执行环境，确保变量作用域正确
            local_context = {
                'df': df,
                'pd': pd,
                'np': np
            }

            # 执行代码并获取返回值
            result = execute_and_return(strategy_expression, local_context)

            # 检查执行结果
            if isinstance(result, str) and result.startswith("Error"):
                st.error(f"❌ 策略执行错误: {result}")
                st.stop()
            elif result is not None:
                # 确保结果是可转换为布尔值的数组或系列
                try:
                    boolean_result = np.asarray(result).astype(bool)
                    df['信号触发'] = np.where(boolean_result, 1, 0).astype(int)
                except Exception as e:
                    st.error(f"❌ 无法将策略返回值转换为信号条件: {e}")
                    st.stop()
            else:
                st.error("❌ 策略表达式最后一行必须是表达式，不能是赋值语句或其他语句")
                st.stop()
        except Exception as e:
            st.error(f"❌ 策略表达式执行错误: {e}")
            st.stop()

        # 4. 计算条件概率 P(C|W) 和 P(C|not W)
        df['W_and_C'] = ((df['胜率触发'] == 1) & (df['信号触发'] == 1)).astype(int)
        df['notW_and_C'] = ((df['胜率触发'] == 0) & (df['信号触发'] == 1)).astype(int)
        
        # 贝叶斯似然率计算
        rolling_w_c = df['W_and_C'].rolling(self.observation_periods).sum().shift(self.holding_period + 1)
        rolling_w = df['胜率触发'].rolling(self.observation_periods).sum().shift(self.holding_period + 1)
        
        rolling_notw_c = df['notW_and_C'].rolling(self.observation_periods).sum().shift(self.holding_period + 1)
        rolling_notw = df['胜率不触发'].rolling(self.observation_periods).sum().shift(self.holding_period + 1)
        # 避免除以零
        p_c_w = rolling_w_c / rolling_w.replace(0, np.nan)
        p_c_notw = rolling_notw_c / rolling_notw.replace(0, np.nan)
        
        # 5. 计算后验概率 P(W|C)
        # 公式: P(W|C) = P(C|W) * P(W) / [P(C|W)*P(W) + P(C|not W)*P(not W)]
        evidence = p_c_w * df['P(W)'] + p_c_notw * (1 - df['P(W)'])
        df['P(W|C)'] = (p_c_w * df['P(W)']) / evidence.replace(0, np.nan)

        # 6. 生成买入信号
        # 逻辑：后验概率 > 先验概率 且 信号触发 且 (绝对概率>0.5 或 概率动量上升)
        prob_condition = (df['P(W|C)'] > 0.5) | (df['P(W|C)'] > df['P(W|C)'].shift(1) * 0.9)
        improve_condition = df['P(W|C)'] > df['P(W)']
        
        df['买入信号'] = np.where(
            improve_condition & (df['信号触发'] == 1) & prob_condition, 
            1, 0
        )

        # 7. 计算策略净值

        # 根据不同的仓位策略计算仓位
        if position_strategy == "原始策略逐步加仓":
            # 原始策略逐步加仓：根据概率变化和历史表现动态调整
            df['仓位'] = np.where(
                df['买入信号'] == 1, 
                df['信号触发'].shift(1).rolling(self.holding_period).sum() / self.holding_period, 
                0
            )
        # 确保仓位在0-1之间
        #df['仓位'] = df['仓位'].clip(0, 1)     
        
        df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
        df['先验仓位净值'] = (1 + (df['P(W)'].shift(1) * df['超额收益率'].fillna(0))).cumprod()

        st.success("回测完成！")
        return df

st.set_page_config(                         #设置网页的标题和图标
            page_title="策略回测", 
            layout="wide",                
        )

# 检查必要的session_state变量
# 优先使用特征池，如果特征池为空则使用单个特征
use_feature_pool = ('feature_pool' in st.session_state) and (not st.session_state.feature_pool.empty)

if not use_feature_pool:
    if not ('features' in st.session_state):
        st.warning("⚠️ 请先在 特征 页面生成特征或添加特征到特征池。")
        st.stop()

if not ('stock_chosen' in st.session_state) or not ('base_chosen' in st.session_state):
    st.warning("⚠️ 请先在 数据 页面选择标的和基准。")
    st.stop()

cols = st.columns([4, 1])                               #布局：两列，左侧宽度为4，右侧宽度为1
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="top"
)
top_right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="top"
)

with top_left_cell:
    # 确定使用哪个特征源
    if use_feature_pool:
        columns = st.session_state.feature_pool.columns.tolist()
        feature_source = "特征池"
    else:
        columns = st.session_state.features.columns.tolist() if st.session_state.features is not None else []
        feature_source = "单个特征"

    st.subheader("📊 策略配置", divider="rainbow")

    # 当前配置信息
    col1, col2 = st.columns(2)
    with col1:
        st.write("**行业:**", st.session_state.get('Industry_selected', '未设置'))
        st.write("**标的:**", st.session_state.get('stock_chosen', '未设置'))
    with col2:
        st.write("**基准:**", st.session_state.get('base_chosen', '未设置'))
        st.write(f"**特征来源:** {feature_source} ({len(columns)} 个)")

    st.markdown("### 策略表达式")

    if st.session_state.get('strategy_expression') is not None:
        s_input_default = st.session_state.strategy_expression
    else:
        s_input_default = "df[''] < 0"

    # 准备自动补全选项，包括df的属性和可用列
    autocomplete_options = []

    # 添加可用特征列到自动补全选项
    for col in columns:
        # 在 caption 中显示完整信息，列名不截断
        autocomplete_options.append({
            "caption": f"df['{col}']",   # 显示的文本（包含完整列名）
            "value": f"df['{col}']",     # 插入的文本
            "meta": "特征",               # 类型标签
            "score": 1000,               # 排序优先级
        })
    
    # 使用CodeEditor组件，配置行号显示和自动补全
    editor_result = code_editor(
        s_input_default,
        lang="python",
        completions=autocomplete_options,
        options={
            "minLines": 10,
            "maxLines": 30,
            "showLineNumbers": True,  # 显示行号
            "highlightActiveLine": True,  # 高亮当前行
            "enableBasicAutocompletion": True,
            "enableLiveAutocompletion": True,
            "enableSnippets": True,
            "fontSize": 14,  # 字体大小
            "fontFamily": "Monaco, Menlo, 'Ubuntu Mono', Consolas, monospace",
            "tooltipFollowsMouse": True,  # 工具提示跟随鼠标
            "showPrintMargin": False,  # 隐藏打印边距线
        },
        component_props={
            "css": """
                /* 自动补全弹出框样式 */
                .ace_autocomplete {
                    width: 1000px !important;
                    max-height: 500px !important;
                    font-size: 14px !important;
                    line-height: 1.6 !important;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
                    border: 1px solid #d0d0d0 !important;
                    border-radius: 4px !important;
                }

                /* 自动补全项样式 */
                .ace_autocomplete .ace_line {
                    padding: 4px 8px !important;
                    white-space: nowrap !important;
                    overflow: visible !important;
                    text-overflow: clip !important;
                }

                /* 高亮匹配文本 */
                .ace_autocomplete .ace_completion-highlight {
                    color: #0066cc !important;
                    font-weight: bold !important;
                }

                /* 选中项的背景色 */
                .ace_autocomplete .ace_line-hover,
                .ace_autocomplete .ace_line.ace_selected {
                    background-color: #e8f4ff !important;
                }

                /* meta 标签样式 */
                .ace_autocomplete .ace_rightAlignedText {
                    color: #999 !important;
                    font-style: italic !important;
                    margin-left: 20px !important;
                }
            """
        },
        theme="vs-light",  # 使用浅色主题
        response_mode=["blur", "submit"],  # 失去焦点或提交时更新
        key="strategy_code_editor"  # 添加唯一key
    )
    
    # 获取编辑后的代码
    if editor_result is not None and "text" in editor_result and editor_result["text"].strip():
        # 如果编辑器返回非空文本，使用它
        s_input = editor_result["text"]
        st.session_state.strategy_expression = s_input
    else:
        # 否则尝试从 session_state 获取，如果没有则使用默认值
        s_input = st.session_state.get('strategy_expression', s_input_default)
        # 只有在 session_state 中没有时才设置默认值
        if 'strategy_expression' not in st.session_state:
            st.session_state.strategy_expression = s_input_default

    # 使用说明
    with st.expander("💡 策略编写指南", expanded=False):
        st.markdown("""
        **基本语法：**
        - 最后一行必须是返回布尔值的表达式
        - 可以使用多行代码，前面的行可以是赋值语句

        **示例：**
        ```python
        # 简单条件
        df['移动平均5'] < 50

        # 组合条件
        (df['移动平均5'] < 50) & (df['环比'] > 0)

        # 多行代码
        ma5 = df['移动平均5']
        ma10 = df['移动平均10']
        ma5 > ma10
        ```

        **可用变量：**
        - `df`: 包含所有特征和价格数据的DataFrame
        - `pd`: pandas 库
        - `np`: numpy 库
        """)

    st.divider()
   
with top_right_cell:    
    st.subheader("回测参数", divider="gray")

    # 从session_state获取保存的值，如果没有则使用默认值
    hp = st.slider(
        "持有期",
        min_value=1,
        max_value=365,
        value=st.session_state.get('holding_period', 5),
        help="持有期越长，交易频率越低"
    )
    st.session_state.holding_period = hp

    op = st.slider(
        "观察期",
        min_value=1,
        max_value=365,
        value=st.session_state.get('observation_period', 60),
        help="计算先验概率的历史窗口长度"
    )
    st.session_state.observation_period = op

    profit_target = st.number_input(
        "目标超额收益",
        value=st.session_state.get('profit_target', 0.0),
        step=0.01,
        format="%.2f",
        help="定义「胜」的标准，超过此收益率视为成功"
    )
    st.session_state.profit_target = profit_target
    
    # 仓位策略选择
    position_strategy = st.selectbox(
        "仓位策略",
        ["原始策略逐步加仓", "待定（别选）"],
        index=st.session_state.get('position_strategy_index', 1),
        help="选择不同的仓位计算策略"
    )
    st.session_state.position_strategy = position_strategy

    with st.expander("可用特征列", expanded=False):
        if columns:
            cols_display = st.columns(1)
            for i, col in enumerate(columns):
                cols_display[i % 1].write(f"• `{col}`")
        else:
            st.info("暂无特征列")

    st.divider()

    # 确定使用哪个特征数据
    if use_feature_pool:
        feature_data = st.session_state.feature_pool
        feature_cols = st.session_state.feature_pool.columns.tolist()
    else:
        feature_data = st.session_state.features
        feature_cols = st.session_state.features.columns.tolist() if st.session_state.features is not None else []

    if feature_data is None or (isinstance(feature_data, pd.DataFrame) and feature_data.empty):
        st.error("⚠️ 请先在 FEATURES 页面生成特征或添加特征到特征池。")
    else:
        if st.button("🚀 开始回测", type="primary", use_container_width=True):
            with st.spinner("⏳ 正在运行回测..."):
                tester = BayesianStrategyBacktester(
                        stock_data=st.session_state.stock_data,
                        baseline_data=st.session_state.base_data,
                        feature_data=feature_data,
                        profit_setted=st.session_state.profit_target,
                        observation_periods=st.session_state.observation_period,
                        holding_period=st.session_state.holding_period,
                        position_strategy=st.session_state.position_strategy
                    )

                df_res = tester.run_strategy(
                        feature_cols=feature_cols,
                        strategy_expression=st.session_state.strategy_expression,
                        position_strategy=tester.position_strategy
                    )

                # 保存回测结果到 session_state 供 AI 助手使用
                st.session_state.df_backtest_result = df_res


if 'df_res' in locals():
    st.divider()
    st.header("📈 回测结果", divider="rainbow")

    # --- 绩效指标 ---
    final_nav = df_res['仓位净值'].iloc[-1]
    prior_nav = df_res['先验仓位净值'].iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "策略净值",
        f"{final_nav:.3f}",
        f"{(final_nav-1):.2%}",
        delta_color="normal"
    )
    c2.metric(
        "先验净值",
        f"{prior_nav:.3f}",
        f"{(prior_nav-1):.2%}",
        delta_color="off"
    )
    excess_gain = final_nav - prior_nav
    c3.metric(
        "超额增益",
        f"{excess_gain:.2%}"
    )

    # Plotly 图表
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("胜率修正（贝叶斯更新）", "净值表现对比", "信号触发分析", "实时仓位变化"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 图1: 胜率修正
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['P(W)'],
        name='先验概率',
        line=dict(color='#FFA726', width=2),
        hovertemplate='日期: %{x}<br>先验概率: %{y:.2%}<extra></extra>'
    ), 1, 1)
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['P(W|C)'],
        name='后验概率',
        line=dict(color='#BDBDBD', width=2),
        hovertemplate='日期: %{x}<br>后验概率: %{y:.2%}<extra></extra>'
    ), 1, 1)

    # 图2: 净值表现
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['仓位净值'],
        name='策略净值',
        line=dict(color='#EF5350', width=2.5),
        hovertemplate='日期: %{x}<br>策略净值: %{y:.4f}<extra></extra>'
    ), 1, 2)
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['先验仓位净值'],
        name='先验净值',
        line=dict(color='blue', width=2),
        hovertemplate='日期: %{x}<br>先验净值: %{y:.4f}<extra></extra>'
    ), 1, 2)

    # 图3: 信号触发分析
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['超额净值'],
        name='超额净值',
        line=dict(color='#66BB6A', width=2.5),
        hovertemplate='日期: %{x}<br>超额净值: %{y:.4f}<extra></extra>'
    ), 2, 1)

    # 信号触发背景
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['信号触发'],
        name='信号触发',
        fill='tozeroy',
        line=dict(width=0),
        fillcolor='rgba(255, 165, 0, 0.15)',
        hovertemplate='日期: %{x}<br>信号: %{y}<extra></extra>'
    ), 2, 1)

    # 图4: 实时仓位变化
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['超额净值'],
        name='超额净值',
        line=dict(color='#7E57C2', width=2),
        hovertemplate='日期: %{x}<br>超额净值: %{y:.4f}<extra></extra>'
    ), row=2, col=2, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['仓位'],
        name='策略仓位',
        fill='tozeroy',
        line_shape='hv',
        line=dict(color='rgba(255, 112, 67, 0.9)', width=1.5),
        fillcolor='rgba(255, 112, 67, 0.2)',
        hovertemplate='日期: %{x}<br>仓位: %{y:.0%}<extra></extra>'
    ), row=2, col=2, secondary_y=True)

    # 更新Y轴标签
    fig.update_yaxes(title_text="概率", row=1, col=1)
    fig.update_yaxes(title_text="净值", row=1, col=2)
    fig.update_yaxes(title_text="净值 / 信号", row=2, col=1)
    fig.update_yaxes(title_text="净值", secondary_y=False, row=2, col=2)
    fig.update_yaxes(title_text="仓位", range=[0, 1.1], secondary_y=True, row=2, col=2)

    # 更新布局
    fig.update_layout(
        height=750,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 提示用户前往 AI 助手页面
    st.divider()
    st.info("💡 想要AI分析这个策略？请前往 **AI Assistant** 页面与智能助手对话！")