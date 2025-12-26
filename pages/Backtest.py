import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BayesianStrategyBacktester:           #贝叶斯策略回测器
    def __init__(self, stock_data, baseline_data, feature_data, profit_setted, observation_periods, holding_period):
        """
        初始化回测器，执行数据对齐和基础收益率计算。
        """
        self.profit_setted = profit_setted
        self.observation_periods = observation_periods
        self.holding_period = holding_period
        
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

    def run_strategy(self, feature_cols, strategy_expression):
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
            # 在 eval 的上下文中，df 变量必须可用
            df['信号触发'] = eval(strategy_expression).astype(int)
        except Exception as e:
            st.error(f"策略表达式错误: {e}")
            st.stop()
            df['信号触发'] = 0

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
        # 仓位逻辑：如果买入，持有 holding_period 天 (这里简化为均摊)
        df['仓位'] = np.where(
            df['买入信号'] == 1, 
            df['信号触发'].rolling(self.holding_period).sum() / self.holding_period, 
            0
        )
        
        df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
        df['先验仓位净值'] = (1 + (df['P(W)'].shift(1) * df['超额收益率'].fillna(0))).cumprod()

        st.success("回测完成！")
        return df

st.set_page_config(                         #设置网页的标题和图标
            page_title="策略回测", 
            layout="wide",                
        )

if not ('features' in st.session_state):                #检查必要的session_state变量
    st.warning("请先在 特征 页面生成特征。")
    st.stop()
if not ('stock_chosen' in st.session_state) or not ('base_chosen' in st.session_state):
    st.warning("请先在 数据 页面选择标的和基准。")
    st.stop()

cols = st.columns([4, 1])                               #布局：两列，左侧宽度为4，右侧宽度为1
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)
top_right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    columns = st.session_state.features.columns.tolist() if st.session_state.features is not None else []
    st.caption("策略设置")
    st.caption(f"行业: {st.session_state.Industry_selected}" if 'Industry_selected' in st.session_state else "请先在 FEATURES 页面选择行业。")
    st.caption(f"标的: {st.session_state.stock_chosen}" if 'stock_chosen' in st.session_state else "请先在 DATA 页面选择标的。")
    st.caption(f"基准: {st.session_state.base_chosen}" if 'base_chosen' in st.session_state else "请先在 DATA 页面选择基准。")
    st.caption(f"因子: {st.session_state.feature_selected}" if 'feature_selected' in st.session_state else "请先在 FEATURES 页面生成特征。")
    st.caption(f"可用特征列: {', '.join(columns)}")
    if st.session_state.get('strategy_expression') is not None:
        s_input_default = st.session_state.strategy_expression
    else:
        s_input_default = "df[''] < 0"
    s_input = st.text_area("策略逻辑 (Python格式)", value=s_input_default)
    st.session_state.strategy_expression = s_input
    st.caption(f"示例: ")
    st.caption(f"df['移动平均5'] < 50 表示 50天移动平均 小于 50 时触发信号。")
    st.caption(f"(df['移动平均5'] < 50) & (df['环比'] > 50) 表示同时满足两个条件时触发信号。")
    st.caption(f"或任意表达式都可以")
   
with top_right_cell:
    st.caption("回测参数")
    hp = st.slider("持有期（以数据频率为单位）", 1, 365, 5)
    st.session_state.holding_period = hp
    op = st.slider("观察期（以数据频率为单位）", 1, 365, 60)
    st.session_state.observation_period = op
    profit_target = st.number_input("目标超额收益", value=0.0, step=0.01)
    st.session_state.profit_target = profit_target     
    if st.session_state.features is None:
        st.error("请先在 FEATURES 页面生成特征。")
    else:
        if st.button("开始回测", width='stretch'):
            tester = BayesianStrategyBacktester(
                    stock_data=st.session_state.stock_data,
                    baseline_data=st.session_state.base_data,
                    feature_data=st.session_state.features,
                    profit_setted=st.session_state.profit_target,   
                    observation_periods=st.session_state.observation_period, 
                    holding_period=st.session_state.holding_period     
                )
                
            df_res = tester.run_strategy(
                    feature_cols=st.session_state.features.columns.tolist(),
                    strategy_expression=s_input
                )


if 'df_res' in locals():                                
    # --- 结果展示 ---
    final_nav = df_res['仓位净值'].iloc[-1]
    prior_nav = df_res['先验仓位净值'].iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("策略净值", f"{final_nav:.3f}", f"{(final_nav-1):.2%}")
    c2.metric("先验净值", f"{prior_nav:.3f}", f"{(prior_nav-1):.2%}", delta_color="off")
    c3.metric("超额增益", f"{(final_nav-prior_nav):.2%}")

    # Plotly 图表
    fig = make_subplots(rows=2, cols=2, subplot_titles=("胜率修正", "净值表现", "信号触发", "实时仓位"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                            [{"secondary_y": False}, {"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W)'], name='先验', line=dict(color='orange')), 1, 1)
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W|C)'], name='后验', line=dict(color='grey')), 1, 1)
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['仓位净值'], name='策略仓位净值', line=dict(color='red')), 1, 2)
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['先验仓位净值'], name='先验仓位净值', line=dict(color='grey')), 1, 2)

    fig.add_trace(go.Scatter(
        x=df_res.index, 
        y=df_res['超额净值'], 
        name='超额净值', 
        line=dict(color='blue', width=1.5)
    ), 2, 1)

    # 再画信号背景
    # 技巧：把信号 y 轴放大到超额净值的范围，或者直接用 yaxis2
    fig.add_trace(go.Scatter(
        x=df_res.index, 
        y=df_res['信号触发'], 
        name='触发脉冲', 
        fill='tozeroy', 
        line=dict(width=0),
        fillcolor='rgba(255, 165, 0, 0.2)', # 浅橙色背景
    ), 2, 1)

    fig.add_trace(
        go.Scatter(
            x=df_res.index, 
            y=df_res['超额净值'], 
            name='超额净值', 
            line=dict(color='blue', width=2),
            hovertemplate='日期: %{x}<br>超额净值: %{y:.4f}<extra></extra>'
        ), 
        row=2, col=2, secondary_y=False
    )

    # 2. 绘制仓位（作为次 Y 轴阴影，使用阶梯线）
    fig.add_trace(
        go.Scatter(
            x=df_res.index, 
            y=df_res['仓位'], 
            name='策略仓位', 
            fill='tozeroy', 
            # 核心优化：使用阶梯线（hv），真实还原调仓的离散跳变
            line_shape='hv', 
            line=dict(color='rgba(255, 165, 0, 0.8)', width=1), 
            # 浅橙色填充，不遮挡背景净值线
            fillcolor='rgba(255, 165, 0, 0.2)', 
            hovertemplate='日期: %{x}<br>当前仓位: %{y:.2f}<extra></extra>'
        ), 
        row=2, col=2, secondary_y=True
    )

    # 3. 更新 Y 轴设置，确保尺度专业
    fig.update_yaxes(title_text="净值水平", secondary_y=False, row=2, col=2)
    fig.update_yaxes(title_text="仓位权重", range=[0, 1.1], secondary_y=True, row=2, col=2)

    fig.update_layout(height=700, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)