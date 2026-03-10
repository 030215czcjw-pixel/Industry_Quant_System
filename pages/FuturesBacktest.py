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

class FuturesStrategyBacktester:
    def __init__(self, stock_data, feature_data):
        """
        初始化回测器，执行数据对齐和基础收益率计算。
        """
        # 1. 数据对齐 (Intersection)
        common_dates = stock_data.index.intersection(feature_data.index).sort_values()
        
        # 保存原始数据副本，以便后续使用
        self.feature_data_aligned = feature_data.loc[common_dates].copy()
        
        # 2. 构建基础价格DataFrame
        self.df = pd.DataFrame({
            '价格': stock_data.loc[common_dates, '收盘']
        }, index=common_dates)
        
        # 3. 计算收益率指标 (预处理)
        self.df['收益率'] = self.df['价格'].pct_change()
        
        # 4. 计算净值曲线
        self.df['净值'] = (1 + self.df['收益率'].fillna(0)).cumprod()
        
        self.df['仓位'] = 0
        
    def run_backtest(self, feature_cols, strategy_expression, empty_position_expression):
        """
        运行回测，计算未来持有期收益率。
        """
        # 使用副本以免污染原始数据
        df = self.df.copy()
        
        # 合并指定的特征列
        for col in feature_cols:
            if col in self.feature_data_aligned.columns:
                df[col] = self.feature_data_aligned[col]
            else:
                print(f"警告: 特征 {col} 不存在于特征数据中")

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
                st.error(f"❌ 多单开仓错误: {result}")
                st.stop()
            elif result is not None:
                # 确保结果是可转换为布尔值的数组或系列
                try:
                    boolean_result = np.asarray(result).astype(bool)
                    df['多单开仓信号触发'] = np.where(boolean_result, 1, 0).astype(int)
                except Exception as e:
                    st.error(f"❌ 无法将多单开仓信号返回值转换为信号条件: {e}")
                    st.stop()
            else:
                st.error("❌ 策略表达式最后一行必须是表达式，不能是赋值语句或其他语句")
                st.stop()
        except Exception as e:
            st.error(f"❌ 多单开仓表达式执行错误: {e}")
            st.stop()

        try:
            # 准备执行环境
            local_context = {
                'df': df,
                'pd': pd,
                'np': np
            }

            # 执行空仓信号表达式并获取返回值
            empty_result = execute_and_return(empty_position_expression, local_context)

            # 检查执行结果
            if isinstance(empty_result, str) and empty_result.startswith("Error"):
                st.error(f"❌ 空单开仓信号执行错误: {empty_result}")
                st.stop()
            elif empty_result is not None:
                # 确保结果是可转换为布尔值的数组或系列
                try:
                    boolean_result = np.asarray(empty_result).astype(bool)
                    df['空单开仓信号'] = np.where(boolean_result, 1, 0).astype(int)
                except Exception as e:
                    st.error(f"❌ 无法将空单开仓信号返回值转换为信号条件: {e}")
                    st.stop()
            else:
                st.error("❌ 策略表达式最后一行必须是表达式，不能是赋值语句或其他语句")
                st.stop()
        except Exception as e:
            st.error(f"❌ 空单开仓信号表达式执行错误: {e}")
            st.stop()

        # 第一步：标记多空信号同时触发的行
        df['多空信号冲突'] = np.where((df['多单开仓信号触发'] == 1) & (df['空单开仓信号'] == 1), 1, 0)

        # 第二步：重新计算仓位（冲突时赋值0，其余逻辑不变）
        df['仓位'] = np.where(
            df['多空信号冲突'] == 1,  # 优先判断冲突
            0,  # 冲突时仓位为0（无效信号）
            np.where(df['多单开仓信号触发'] == 1, 1,  # 无冲突则按原逻辑
                    np.where(df['空单开仓信号'] == 1, -1, 0))
        )
        df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['收益率'].fillna(0))).cumprod()
        print(df['仓位'])
        print(df['仓位净值'])
        
        st.success("回测完成！")
        return df
            
st.set_page_config(                         #设置网页的标题和图标
            page_title="策略回测", 
            layout="wide",                
        )

cols = st.columns([4, 1])                               #布局：两列，左侧宽度为4，右侧宽度为1
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="top"
)
top_right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="top"
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
            "minLines": 8,
            "maxLines": 25,
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

    st.divider()

    # 空仓信号表达式输入区域
    st.markdown("### 空仓信号表达式")

    if st.session_state.get('empty_position_expression') is not None:
        empty_input_default = st.session_state.empty_position_expression
    else:
        empty_input_default = "0"

    # 使用CodeEditor组件，配置行号显示和自动补全
    empty_editor_result = code_editor(
        empty_input_default,
        lang="python",
        completions=autocomplete_options,
        options={
            "minLines": 5,
            "maxLines": 15,
            "showLineNumbers": True,
            "highlightActiveLine": True,
            "enableBasicAutocompletion": True,
            "enableLiveAutocompletion": True,
            "enableSnippets": True,
            "fontSize": 14,
            "fontFamily": "Monaco, Menlo, 'Ubuntu Mono', Consolas, monospace",
            "tooltipFollowsMouse": True,
            "showPrintMargin": False,
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
        theme="vs-light",
        response_mode=["blur", "submit"],
        key="empty_position_code_editor"  # 添加唯一key
    )

    # 获取编辑后的代码
    if empty_editor_result is not None and "text" in empty_editor_result and empty_editor_result["text"].strip():
        empty_input = empty_editor_result["text"]
        st.session_state.empty_position_expression = empty_input
    else:
        empty_input = st.session_state.get('empty_position_expression', empty_input_default)
        if 'empty_position_expression' not in st.session_state:
            st.session_state.empty_position_expression = empty_input_default
            
    st.divider()
        
with top_right_cell:    
    st.subheader("回测参数", divider="gray")

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
                tester = FuturesStrategyBacktester(
                        stock_data=st.session_state.stock_data,
                        feature_data=feature_data
                    )

                df_res = tester.run_backtest(
                        feature_cols=feature_cols,
                        strategy_expression=st.session_state.strategy_expression,
                        empty_position_expression=st.session_state.get('empty_position_expression', '0')
                    )

                # 保存回测结果到 session_state 供 AI 助手使用
                st.session_state.df_backtest_result = df_res
                
if 'df_res' in locals():
    st.divider()
    st.header("📈 回测结果", divider="rainbow")

    # --- 绩效指标 ---
    final_nav = df_res['仓位净值'].iloc[-1]
    prior_nav = df_res['净值'].iloc[-1]
    
    # 计算最大回撤
    def calculate_max_drawdown(nav_series):
        cumulative_max = nav_series.cummax()
        drawdown = (nav_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        return max_drawdown
    
    max_drawdown = calculate_max_drawdown(df_res['仓位净值'])
    
    # 计算夏普比率 (假设无风险利率为0)
    def calculate_sharpe_ratio(nav_series, risk_free_rate=0):
        # 计算日收益率
        daily_returns = nav_series.pct_change().dropna()
        # 计算年化收益率
        annualized_return = daily_returns.mean() * 252
        # 计算年化波动率
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        # 计算夏普比率
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        return sharpe_ratio
    
    sharpe_ratio = calculate_sharpe_ratio(df_res['仓位净值'])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "策略净值",
        f"{final_nav:.3f}",
        f"{(final_nav-1):.2%}",
        delta_color="normal"
    )
    c2.metric(
        "原始净值",
        f"{prior_nav:.3f}",
        f"{(prior_nav-1):.2%}",
        delta_color="off"
    )
    excess_gain = final_nav - prior_nav
    c3.metric(
        "超额收益",
        f"{excess_gain:.2%}"
    )
    c4.metric(
        "最大回撤",
        f"{max_drawdown:.2%}"
    )
    c5.metric(
        "夏普比率",
        f"{sharpe_ratio:.2f}"
    )

    # Plotly 图表
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("净值表现对比", "实时仓位变化"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 净值表现
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['仓位净值'],
        name='策略净值',
        line=dict(color='#EF5350', width=2.5),
        hovertemplate='日期: %{x}<br>策略净值: %{y:.4f}<extra></extra>'
    ), 1, 1)
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['净值'],
        name='原始净值',
        line=dict(color='blue', width=2),
        hovertemplate='日期: %{x}<br>原始净值: %{y:.4f}<extra></extra>'
    ), 1, 1)

    # 
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['价格'],
        name='价格',
        line=dict(color='blue', width=2.5),
        hovertemplate='日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
    ), 2, 1)

    # 分离多头和空头仓位
    df_res['long_position'] = df_res['仓位'].where(df_res['仓位'] == 1, 0)
    df_res['short_position'] = df_res['仓位'].where(df_res['仓位'] == -1, 0)
    
    # 多头仓位 (红色)
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['long_position'],
        name='多头仓位',
        fill='tozeroy',
        line_shape='hv',
        line=dict(color='rgba(255, 0, 0, 0.9)', width=1.5),
        fillcolor='rgba(255, 0, 0, 0.2)',
        hovertemplate='日期: %{x}<br>仓位: %{y:.0%}<extra></extra>'
    ), row=2, col=1, secondary_y=True)
    
    # 空头仓位 (绿色)
    fig.add_trace(go.Scatter(
        x=df_res.index,
        y=df_res['short_position'],
        name='空头仓位',
        fill='tozeroy',
        line_shape='hv',
        line=dict(color='rgba(0, 255, 0, 0.9)', width=1.5),
        fillcolor='rgba(0, 255, 0, 0.2)',
        hovertemplate='日期: %{x}<br>仓位: %{y:.0%}<extra></extra>'
    ), row=2, col=1, secondary_y=True)

    # 更新Y轴标签
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="价格", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="仓位", range=[-1.1, 1.1], secondary_y=True, row=2, col=1)

    # 更新布局
    fig.update_layout(
        height=900,
        width=None,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=100, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 提示用户前往 AI 助手页面
    st.divider()
    st.info("💡 想要AI分析这个策略？请前往 **AI Assistant** 页面与智能助手对话！")
