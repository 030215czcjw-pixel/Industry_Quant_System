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

# 布局
col_main, col_sidebar = st.columns([3, 1])


with col_main:
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
    if st.button("生成回测逻辑"):
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
      response_content = response.choices[0].message.content
      st.code(response_content, language="python")
      

       