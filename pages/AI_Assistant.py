import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from openai import OpenAI
from tools import AIStrategyBacktester, save_strategy_results



st.set_page_config(
    page_title="AI 策略助手",
    layout="wide",
    page_icon="🤖"
)

# 布局
col_main, col_sidebar = st.columns([3, 1])

result = {}
strategies = []

with col_main:
    use_feature_pool = ('feature_pool' in st.session_state) and (not st.session_state.feature_pool.empty)
    if not use_feature_pool:
        if not ('features' in st.session_state):
            st.warning("⚠️ 请先在 特征 页面生成特征或添加特征到特征池。")
            st.stop()
    if use_feature_pool:
        feature_data = st.session_state.feature_pool
        feature_columns = st.session_state.feature_pool.columns.tolist()
        feature_source = "特征池"
    else:
        feature_data = st.session_state.features
        feature_columns = st.session_state.features.columns.tolist() if st.session_state.features is not None else []
        feature_source = "单个特征"
        
    system_prompt = f"""
            # Role
你是一名精通 Python 和金融工程的量化开发专家。具备以下核心能力：因子逻辑解构能力：能精准识别不同类型因子（量价、基本面、情绪、宏观等）的金融含义，拆解因子背后的供需、资金、市场预期逻辑；
多维度策略构建能力：可基于单一因子、多因子共振 / 背离、因子趋势 / 水平、因子分位数 / 极值等维度，设计从简单到复杂的交易逻辑；
代码工程化能力：生成的 pandas 代码严格贴合回测框架要求，逻辑严谨、计算高效，兼容时间序列处理的常见场景（如去极值、中性化、滚动计算）；
策略解释专业性：对每个交易逻辑的解释需贴合金融市场实际，明确逻辑背后的经济学 / 金融学原理，而非单纯的数学规则描述。你的任务是将用户选择的因子池进行筛选并转化为符合特定回测框架的 pandas 代码。

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
    if st.button("生成回测逻辑（修改了特征列会报错，重新生成就好）"):
      client = OpenAI(
          api_key="sk-83579366c15a4f60a71c9f8d4628e8de", 
          base_url="https://api.deepseek.com"
      )
      with st.status("正在生成结果...", expanded=True) as status:
        response = client.chat.completions.create(
                    model="deepseek-chat",  # 或者使用 "deepseek-reasoner" (即 R1 模型)
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "请输出回测逻辑"},
                    ],
                    stream=False
                )
      st.session_state.response_content = response.choices[0].message.content
      #st.code(response_content, language="python")
     
    if 'response_content' in st.session_state:
      response_content = st.session_state.response_content
       
      #try:
      #  with open('example.json', 'r', encoding='utf-8') as f:
      #        # 解析 JSON 内容
      #        strategies = json.load(f)
      #except:
      #    st.error("⚠️ 生成回测逻辑失败，请检查输入是否符合要求。")
      #    st.stop()
      
            
      strategies = json.loads(response_content)
            
      for idx, strategy in enumerate(strategies, 1):
            # 提取策略关键信息（带默认值，避免字段缺失报错）
            strategy_name = strategy.get("strategy_name")
            long_condition = strategy.get("long_condition")
            short_condition = strategy.get("short_condition")
            description = strategy.get("description")
            
            st.divider()
            # 3. 输出/处理单个策略信息
            st.subheader(f"{strategy_name}")
            st.write(f"  做多条件：{long_condition}")
            st.write(f"  做空条件：{short_condition}")
            st.write(f"  策略描述：{description}")
            
            backtester = AIStrategyBacktester(stock_data=st.session_state.stock_data, feature_data=feature_data)
            df_res = backtester.run_backtest(feature_cols=feature_columns, long_condition=long_condition, short_condition=short_condition)
            

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

            result[strategy_name] = {
                "description": description,
                "long_condition": long_condition,
                "short_condition": short_condition,
                "final_nav": final_nav,
                "prior_nav": prior_nav,
                "excess_gain": excess_gain,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio
            }

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
            
with col_sidebar:
    # 侧边栏标题
    st.subheader("📈 策略回测结果")
    if strategies:
      st.subheader(f"共生成 {len(strategies)} 个交易逻辑（按超额收益排序）")
      st.divider()

      # 按超额收益降序排序
      sorted_results = sorted(
          result.items(),
          key=lambda x: x[1]['excess_gain'],
          reverse=True
      )

      # 遍历排序后的结果，卡片式展示
      for idx, (strategy_name, metrics) in enumerate(sorted_results, 1):
          description = metrics["description"]
          # 带边框的卡片容器
          with st.container(border=True):
              # 策略排名+名称
              st.markdown(f"### {strategy_name}")
              st.caption(description)  # 用caption显示策略描述，更紧凑

              # 三列展示指标
              col1, col2, col3 = st.columns(3)
              
              # 超额收益（红正绿负）
              excess_gain = metrics['excess_gain']
              gain_color = "red" if excess_gain > 0 else "green"
              col1.markdown(
                  f"<span style='color:{gain_color};'>超额收益</span><br>{excess_gain:.2%}",
                  unsafe_allow_html=True
              )
              
              # 最大回撤
              col2.markdown(
                  f"最大回撤<br>{metrics['max_drawdown']:.2%}",
                  unsafe_allow_html=True
              )
              
              # 夏普比率
              sharpe = metrics['sharpe_ratio']
              sharpe_color = "blue" if sharpe > 1 else "gray"
              col3.markdown(
                  f"<span style='color:{sharpe_color};'>夏普比率</span><br>{sharpe:.2f}",
                  unsafe_allow_html=True
              )
          
          st.divider()

      if st.button("保存为Excel", use_container_width=True):
                file_path, error = save_strategy_results(sorted_results, "xlsx")
                if file_path:
                    # 读取文件供下载
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="📥 下载Excel文件",
                            data=f,
                            file_name=os.path.basename(file_path),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    st.success(f"✅ 已保存：{file_path}")
                else:
                    st.error(error)
    
      
    else:
      st.error("⚠️ 未生成结果")
      st.stop()
        
        
