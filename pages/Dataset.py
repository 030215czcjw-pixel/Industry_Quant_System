import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
            page_title="数据管理", 
            layout="wide",                 # 布局模式 ("centered" 或 "wide")
        )

stocks = ["中国神华", "综合交易价_CCTD秦皇岛动力煤(Q5500)", "招商轮船", "南方航空"]
bases = ["沪深300"]

cols = st.columns([1, 1])
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)
top_right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    stock_chosen = st.pills(
        "选择标的",
        options=stocks,
        selection_mode="single"
    )
    st.session_state.stock_chosen = stock_chosen
    if stock_chosen:
        try:
            stock_data = pd.read_excel('data/stock_data.xlsx', sheet_name=stock_chosen, index_col='日期', parse_dates=True)
            st.session_state.stock_data = stock_data
        except Exception as e:
            st.error(f"无法加载数据: {e}")
            st.stop()
    else:
        st.warning("⚠️ 请先选择标的。")
        st.stop()

with top_right_cell:
    base_chosen = st.pills(
        "选择基准",
        options=bases,
        default=bases[0],
        selection_mode="single"
    )
    st.session_state.base_chosen = base_chosen
    try:
        base_data = pd.read_excel('data/stock_data.xlsx', sheet_name=base_chosen, index_col='date', parse_dates=True)
        st.session_state.base_data = base_data
    except Exception as e:
        st.error(f"无法加载数据: {e}")
        st.stop()

cols = st.columns(1)
bottom_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with bottom_cell:
    # 1. 开启双Y轴模式
    fig = make_subplots(
        rows=1, cols=1, 
        specs=[[{"secondary_y": True}]] 
    )

    # 2. 第一条线：个股 (使用左轴 secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x=st.session_state.stock_data.index,
            y=st.session_state.stock_data['收盘'],
            mode='lines',
            line=dict(color='red'),
            name=f"{stock_chosen} (左轴)"
        ),
        secondary_y=False,
    )

    # 3. 第二条线：基准 (使用右轴 secondary_y=True)
    fig.add_trace(
        go.Scatter(
            x=st.session_state.base_data.index,
            y=st.session_state.base_data['close'],
            mode='lines',
            line=dict(color='grey'),
            name=f"{base_chosen} (右轴)"
        ),
        secondary_y=True,
    )

    # 4. 调整布局
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02, # 放在顶部，防止遮挡
            xanchor="left",
            x=0
        ),
        # 设置左右轴的标题（可选）
        yaxis=dict(title=stock_chosen),
        yaxis2=dict(title=base_chosen, showgrid=False) # showgrid=False 防止网格线打架
    )
    fig.update_xaxes(
        rangeslider_visible=True,  # 显示底部滑动条
        row=1, col=1               # 指定应用到哪个子图
    )
    st.plotly_chart(fig, width='stretch')
