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

    def run_strategy(self, feature_cols, strategy_expression, position_strategy, empty_position_expression="0", empty_position_mode="硬清仓"):
        """
        执行贝叶斯分析和信号生成
        :param feature_cols: list, 参与计算的特征列名
        :param strategy_expression: str, 策略触发条件的字符串表达式 (例如: "df['RSI'] > 70")
        :param position_strategy: str, 仓位策略
        :param empty_position_expression: str, 空仓信号表达式，默认为"0"（不触发空仓）
        :param empty_position_mode: str, 空仓模式，可选："硬清仓"、"半仓止损"、"三分之一仓"、"渐进式减仓"
        :return: DataFrame, 包含完整分析结果
        """
        # 使用副本以免污染原始数据
        df = self.df.copy()
        
        # 合并指定的特征列
        for col in feature_cols:
            if col in self.feature_data_aligned.columns:
                df[col] = self.feature_data_aligned[col].shift(1)
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

        # 3.5 执行空仓信号表达式
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
                st.error(f"❌ 空仓信号执行错误: {empty_result}")
                st.stop()
            elif empty_result is not None:
                # 确保结果是可转换为布尔值的数组或系列
                try:
                    boolean_result = np.asarray(empty_result).astype(bool)
                    df['空仓信号'] = np.where(boolean_result, 1, 0).astype(int)
                except Exception as e:
                    st.error(f"❌ 无法将空仓信号返回值转换为信号条件: {e}")
                    st.stop()
            else:
                # 如果返回None，默认为0（不触发空仓）
                df['空仓信号'] = 0
        except Exception as e:
            st.error(f"❌ 空仓信号表达式执行错误: {e}")
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
        elif position_strategy == "先快后慢加仓":
            # 先快后慢加仓：改进版，更及时响应信号
            # 创建一个仓位累积计数器
            df['仓位'] = np.where(
                df['买入信号'] == 1,
                0.3 + 0.7 * np.sqrt(df['信号触发'].shift(1).rolling(self.holding_period).sum() / self.holding_period),
                0
            )
        elif position_strategy == "正金字塔建仓":
            # 正金字塔建仓：底部仓位最重，越涨买得越少
            # 核心思想：在低位时重仓，随着价格上涨逐步减仓，降低风险

            # 计算持有期内的超额净值涨幅（相对于持有期前的最低点）
            df['持有期内最低净值'] = df['超额净值'].shift(1).rolling(self.holding_period).min()
            df['相对底部涨幅'] = (df['超额净值'].shift(1) - df['持有期内最低净值']) / df['持有期内最低净值'].replace(0, np.nan)

            # 初始化仓位为0
            df['仓位'] = 0.0

            # 只在买入信号触发时计算仓位
            buy_signal_mask = df['买入信号'] == 1

            # 正金字塔逻辑：涨幅越大，仓位越小
            # 使用向量化操作提高性能
            relative_rise = df.loc[buy_signal_mask, '相对底部涨幅'].fillna(0)

            # 分段仓位分配：
            # 涨幅 0-5%：80%仓位（底部重仓）
            # 涨幅 5-10%：60%仓位
            # 涨幅 10-15%：40%仓位
            # 涨幅 >15%：20%仓位
            df.loc[buy_signal_mask, '仓位'] = np.select(
                [
                    relative_rise < 0.05,
                    relative_rise < 0.10,
                    relative_rise < 0.15,
                    relative_rise >= 0.15
                ],
                [0.8, 0.6, 0.4, 0.2],
                default=0.8  # 默认使用最大仓位
            )

        elif position_strategy == "时间加权加仓":
            # 时间加权加仓：越近的日期产生的信号权重越大
            # 核心思想：最近的信号更重要，使用指数加权来计算仓位

            # 初始化仓位为0
            df['仓位'] = 0.0

            # 只在买入信号触发时计算仓位
            buy_signal_mask = df['买入信号'] == 1

            # 使用指数加权移动平均(EWM)计算信号的加权和
            # span参数控制衰减速度，span越小，越重视近期信号
            span = max(self.holding_period // 2, 3)  # 至少为3，最大为持有期的一半

            # 计算信号的指数加权移动平均
            df['信号加权'] = df['信号触发'].shift(1).ewm(span=span, adjust=False).mean()

            # 在买入信号触发时，根据加权信号计算仓位
            # 加权信号范围是0-1，可以直接用作仓位比例
            df.loc[buy_signal_mask, '仓位'] = df.loc[buy_signal_mask, '信号加权']

            # 设置最小仓位阈值，避免仓位过小
            df.loc[buy_signal_mask & (df['仓位'] < 0.2), '仓位'] = 0.2

        # 确保仓位在0-1之间
        df['仓位'] = df['仓位'].clip(0, 1)

        # 应用空仓信号：根据不同的空仓模式处理仓位
        if empty_position_mode == "硬清仓":
            # 模式1：硬清仓 - 触发即归零
            df['仓位'] = np.where(df['空仓信号'] == 1, 0, df['仓位'])

        elif empty_position_mode == "半仓止损":
            # 模式2：半仓止损 - 触发时减至原仓位的50%
            df['仓位'] = np.where(df['空仓信号'] == 1, df['仓位'] * 0.5, df['仓位'])

        elif empty_position_mode == "三分之一仓":
            # 模式3：三分之一仓 - 触发时减至原仓位的33%
            df['仓位'] = np.where(df['空仓信号'] == 1, df['仓位'] * 0.33, df['仓位'])

        elif empty_position_mode == "渐进式减仓":
            # 模式4：渐进式减仓 - 连续触发时逐步减仓
            # 创建一个累计触发计数器
            df['空仓累计'] = (df['空仓信号'] == 1).astype(int)

            # 使用shift和cumsum创建连续触发计数
            # 当空仓信号为0时重置计数
            df['空仓连续触发'] = 0
            current_count = 0
            for idx in df.index:
                if df.loc[idx, '空仓信号'] == 1:
                    current_count += 1
                else:
                    current_count = 0
                df.loc[idx, '空仓连续触发'] = current_count

            # 根据连续触发次数递减仓位
            # 第1次：减至80%，第2次：减至60%，第3次：减至40%，第4次：减至20%，第5次及以上：清仓
            df['减仓系数'] = np.select(
                [
                    df['空仓连续触发'] == 0,
                    df['空仓连续触发'] == 1,
                    df['空仓连续触发'] == 2,
                    df['空仓连续触发'] == 3
                ],
                [1.0, 0.6, 0.3, 0.0],
                default=1.0
            )
            df['仓位'] = df['仓位'] * df['减仓系数']     
        
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

    # 空仓信号使用说明
    with st.expander("💡 空仓信号编写指南", expanded=False):
        st.markdown("""
        **功能说明：**
        - 空仓信号用于触发减仓或清仓操作
        - 默认值为 `0`，表示不触发空仓
        - 可以在下方选择不同的空仓模式

        **基本语法：**
        - 最后一行必须是返回布尔值的表达式
        - 返回True时触发空仓，False时不触发

        **示例：**
        ```python
        # 不触发空仓（默认）
        0

        # 当超额净值下跌超过20%时清仓
        df['超额净值'] < df['超额净值'].rolling(20).max() * 0.8

        # 当先验概率低于30%时清仓
        df['P(W)'] < 0.3

        # 组合条件
        (df['超额净值'] < 0.9) & (df['P(W)'] < 0.4)
        ```

        **可用变量：**
        - `df`: 包含所有特征和价格数据的DataFrame
        - `pd`: pandas 库
        - `np`: numpy 库
        """)

    # 空仓模式选择
    st.markdown("##### 空仓模式")
    empty_position_mode = st.selectbox(
        "选择空仓触发后的处理方式",
        ["硬清仓", "半仓止损", "三分之一仓", "渐进式减仓"],
        index=st.session_state.get('empty_position_mode_index', 0),
        help="硬清仓：触发即归零；半仓止损：减至50%；三分之一仓：减至33%；渐进式减仓：连续触发时逐步减仓（80%→60%→40%→20%→0%）"
    )
    st.session_state.empty_position_mode = empty_position_mode

    # 根据模式显示说明
    mode_descriptions = {
        "硬清仓": "⚠️ **风控最强**：触发时立即清仓，仓位归零",
        "半仓止损": "🛡️ **平衡模式**：触发时保留50%仓位，既控制风险又保留机会",
        "三分之一仓": "📊 **保守减仓**：触发时保留33%仓位，留有一定底仓",
        "渐进式减仓": "📉 **渐进退出**：连续触发时逐步减仓（第1次80%→第2次60%→第3次40%→第4次20%→第5次0%）"
    }
    st.info(mode_descriptions[empty_position_mode])

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
        ["原始策略逐步加仓", "先快后慢加仓", "正金字塔建仓", "时间加权加仓"],
        index=st.session_state.get('position_strategy_index', 0),
        help="选择不同的仓位计算策略：\n• 原始策略：根据信号触发次数逐步加仓\n• 先快后慢：使用平方根函数先快速加仓后逐渐放缓\n• 正金字塔：底部仓位最重，越涨买得越少\n• 时间加权：越近的信号权重越大，使用指数加权"
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
                        position_strategy=tester.position_strategy,
                        empty_position_expression=st.session_state.get('empty_position_expression', '0'),
                        empty_position_mode=st.session_state.get('empty_position_mode', '硬清仓')
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

    # 空仓信号背景（如果有触发）
    if df_res['空仓信号'].sum() > 0:
        fig.add_trace(go.Scatter(
            x=df_res.index,
            y=df_res['空仓信号'],
            name='空仓信号',
            fill='tozeroy',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            hovertemplate='日期: %{x}<br>空仓: %{y}<extra></extra>'
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
