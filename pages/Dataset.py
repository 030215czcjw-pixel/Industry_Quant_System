import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
            page_title="数据管理", 
            layout="wide",                 # 布局模式 ("centered" 或 "wide")
        )

stocks = ["中国神华", "秦皇岛5500K动力末煤平仓价", "动力煤（中信）", "招商轮船", "南方航空", "太平洋航运", "网易-S", "哔哩哔哩"]
bases = ["沪深300", "恒生综指", "零基准"]

cols = st.columns([1, 1])
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)
top_right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    if st.session_state.get('stock_chosen') is not None:
        default_stock = st.session_state['stock_chosen']
    else:
        default_stock = None
    stock_chosen = st.pills(
        "选择标的",
        options=stocks,
        default=default_stock,
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
    
    if base_chosen == "零基准":
        st.caption("零基准下的超额序列存在一定计算精度问题, 超额序列和原始序列不完全吻合")
        # 如果选择零基准，创建一个与标的价格索引相同、收盘价恒定为1.0的数据框
        # 使用1.0可以确保 pct_change() 计算结果恒为 0，不会产生 NaN
        if 'stock_data' in st.session_state:
            base_data = pd.DataFrame(index=st.session_state.stock_data.index)
            base_data['close'] = 1.0 
            st.session_state.base_data = base_data
        else:
            st.warning("请先选择标的以对齐日期。")
            st.stop()
    else:
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

# 计算超额收益
# 零基准存在计算精度问题
st.session_state.stock_data['收益率'] = st.session_state.stock_data['收盘'].pct_change()
if st.session_state.base_chosen == "零基准":
    # 直接使用原始序列进行归一化
    st.session_state.stock_data['超额收益率'] = st.session_state.stock_data['收益率'].fillna(0)
    st.session_state.stock_data['累计超额收益'] = st.session_state.stock_data['收盘'] / st.session_state.stock_data['收盘'].iloc[0]
else:
    # 正常基准下，计算相对于基准的超额收益率，再累乘生成超额净值
    st.session_state.base_data['收益率'] = st.session_state.base_data['close'].pct_change()
    st.session_state.stock_data['超额收益率'] = (st.session_state.stock_data['收益率'].fillna(0) - 
                                             st.session_state.base_data['收益率'].fillna(0))
    st.session_state.stock_data['累计超额收益'] = (1 + st.session_state.stock_data['超额收益率']).cumprod()

with bottom_cell:
    st.caption("走势对比", text_alignment="center")
    # 初始化双轴图 (左轴+右轴)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # trace1: 个股 (左轴 y1)
    fig.add_trace(
        go.Scatter(
            x=st.session_state.stock_data.index,
            y=st.session_state.stock_data['收盘'],
            name=f"{stock_chosen} (左轴)",
            line=dict(color='red', width=1.5),
            opacity=0.8,
            connectgaps=True  # 连接缺失值之间的点
        ),
        secondary_y=False
    )

    # trace2: 基准 (右轴 y2)
    fig.add_trace(
        go.Scatter(
            x=st.session_state.base_data.index,
            y=st.session_state.base_data['close'],
            name=f"{base_chosen} (右轴1)",
            line=dict(color='grey', width=1.5),
            opacity=0.5,
            connectgaps=True  # 连接缺失值之间的点
        ),
        secondary_y=True
    )

    # trace3: 累计超额 (右轴2 - 新增的第三轴)
    # 注意：这里不使用 secondary_y 参数，而是直接指定 yaxis='y3'
    fig.add_trace(
        go.Scatter(
            x=st.session_state.stock_data.index,
            y=st.session_state.stock_data['累计超额收益'],
            name="累计超额收益 (右轴2)",
            line=dict(color='#ff7f0e', width=1.5),
            opacity=0.5,
            fill='tozeroy',                      # 填充颜色，更醒目
            fillcolor='rgba(255, 127, 14, 0.1)', # 半透明橙色
            yaxis="y3",                          # 关键：指定挂载到 y3 轴
            connectgaps=True  # 连接缺失值之间的点
        )
    )

    # --- 3. 布局调整 (核心难点) ---
    fig.update_layout(
        xaxis=dict(domain=[0, 0.9]), # 压缩X轴绘图区域，给右侧留出空间放第三个轴
        
        # 左轴配置
        yaxis=dict(
            title=dict(text=stock_chosen, font=dict(color="red"))
        ),
        
        # 右轴1配置 (基准)
        yaxis2=dict(
            title=dict(text=base_chosen, font=dict(color="grey")),
            showgrid=False # 关掉网格防乱
        ),
        
        # 右轴2配置 (超额收益 - 第三轴)
        yaxis3=dict(
            title=dict(text="累计超额收益", font=dict(color="#ff7f0e")),
            anchor="free",     # 轴的位置自由移动
            overlaying="y",    # 覆盖在主图上
            side="right",      # 放在右边
            position=1.0,      # 1.0 是紧贴右侧图表边缘，此轴会和 y2 重叠，下面调整位置
            showgrid=False,
            tickformat='.2%'
        ),
        
        legend=dict(x=0, y=1.1, orientation='h'), # 图例放顶上
        hovermode="x unified" # 统一悬停显示
    )

    # 调整 y2 和 y3 的位置，避免重叠
    # 实际上 Plotly 的 secondary_y 默认在右侧。我们需要把 y3 推得更远一点。
    fig.update_layout(
        yaxis2=dict(position=0.9),  # 把基准轴稍微往左挪一点（配合 domain=[0, 0.9]）
        yaxis3=dict(position=1.0)   # 把超额轴放在最右边
    )
    
    # 更好的方案：保持 domain=[0, 0.85]，y2 在 0.85, y3 在 0.92
    fig.update_layout(xaxis=dict(domain=[0, 0.88])) # 图表画板占 88% 宽度
    fig.update_layout(yaxis2=dict(position=0.88))   # 第一右轴贴着画板
    fig.update_layout(yaxis3=dict(position=0.96))   # 第二右轴再往右偏离一点

    # --- 4. 添加下方拖动时间轴 & 快捷按钮 ---
    fig.update_xaxes(
        # 1. 开启下方拖动条
        rangeslider_visible=True,
        rangeslider=dict(
            visible=True,
            thickness=0.1,  # 设置拖动条的高度（占比 10%）
            bgcolor='rgba(230,230,230,0.5)' # 拖动条背景色
        ),
        
        # 2. 时间范围选择按钮
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6月", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(count=5, label="5年", step="year", stepmode="backward"),
                dict(step="all", label="全部")
            ]),
            x=0.73,     # 按钮位置 X
            y=1.1,  # 按钮位置 Y (放在图表上方)
            bgcolor='rgba(255,255,255,0.8)' # 按钮背景
        )
    )

    st.plotly_chart(fig, width='stretch')
