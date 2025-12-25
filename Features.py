import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from filterpy.kalman import KalmanFilter


def apply_kalman(series, Q_val=0.01, R_val=0.1):
    # 确保数据无空值
    vals = series.ffill().bfill().to_numpy()
    
    # 
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[vals[0]]]) # 初始状态
    kf.F = np.array([[1.]])      # 状态转移矩阵
    kf.H = np.array([[1.]])      # 观测矩阵
    kf.P *= 10.                  # 初始协方差
    kf.R = R_val                 # 测量噪声
    kf.Q = Q_val                 # 过程噪声
    
    filtered_results = []
    for z in vals:
        kf.predict()
        kf.update(z)
        filtered_results.append(kf.x[0, 0])
        
    return filtered_results

def generate_features(data, n_lag, n_MA, n_D, n_yoy, use_kalman):
    df = pd.DataFrame(index=data.index)
    # 强制转换为 float64
    df['原始数据'] = data.iloc[:, 0].astype('float64')

    # 1. 是否应用卡尔曼滤波
    if use_kalman:
        df['卡尔曼滤波'] = apply_kalman(df['原始数据'])
        data_source = df['卡尔曼滤波'] # 后续计算基于滤波后的数据
    else:
        data_source = df['原始数据']

    # 2. 循环生成特征 
    if n_lag > 0:
        df[f'滞后{n_lag}'] = data_source.shift(n_lag)
    
    if n_MA > 0:
        df[f'移动平均{n_MA}'] = data_source.shift(n_lag).rolling(window=n_MA).mean()
                
    if n_D > 0:
        df[f'差分{n_D}'] = data_source.shift(n_lag).diff(n_D)
    
    if n_yoy > 1:
        df[f'同比{n_yoy}'] = data_source.shift(n_lag).pct_change(n_yoy) 
    
    if n_yoy == 1:
        df[f'环比'] = data_source.shift(n_lag).pct_change(1)
            
    return df

def load_and_clean_feature(xl_obj, sheet_name):
    try:
        df = xl_obj.parse(sheet_name)
        # 自动寻找日期列并设为索引
        for col in df.columns:
            if '日期' in str(col) or 'Date' in str(col) or 'time' in str(col).lower():
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                return df # 找到后直接返回
        return df
    except Exception as e:
        st.error(f"读取数据出错: {e}")
        return pd.DataFrame()

# --- 页面配置 ---
st.set_page_config(page_title="特征工程", layout="wide")

# --- 布局结构 ---
cols = st.columns([1, 3], vertical_alignment="top")
top_left_cell = cols[0].container(border=True, height=250)
top_right_cell = cols[1].container(border=True, height=250)

Industry_list = ["交运", "煤炭"]
SHEET_LIST = {
    "交运": "1VVTAG1ixDe50ysjMZEAAZyvYkUbiHBvolh0oaYn8Mxw", 
    "煤炭": "1P3446_9mBi-7qrAMi78F1gHDHGIOCjw-"
} 

# --- 初始化 Session State ---
if 'xl_object' not in st.session_state:
    st.session_state['xl_object'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = pd.DataFrame()

# --- 左上角：数据源加载 ---
with top_left_cell:
    st.session_state['Industry_selected'] = st.selectbox("选择行业", Industry_list)
    SHEET_ID = SHEET_LIST[st.session_state['Industry_selected']]

    #@st.cache_resource(show_spinner=False) # 缓存Excel对象，避免重复下载
    def fetch_xl_object(sheet_id):
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
        return pd.ExcelFile(url)

    if st.button("同步云端表", use_container_width=True):
        with st.spinner("正在下载并解析数据..."):
            try:
                st.session_state['xl_object'] = fetch_xl_object(SHEET_ID)
                st.success("同步成功！")
            except Exception as e:
                st.error(f"同步失败: {e}")

# --- 右上角：选择具体特征 ---
with top_right_cell:
    xl = st.session_state['xl_object'] # 获取对象
    
    if xl is None:
        st.warning("⚠️ 请先在左侧点击“同步云端表”以加载数据。")
        st.stop() # 停止运行下面的代码，防止报错
        
    feature_selected = st.pills("选择特征", xl.sheet_names, selection_mode="single")
    st.session_state.feature_selected = feature_selected

# --- 中间区域布局 ---
cols_middle = st.columns([3, 1])
middle_left_cell = cols_middle[0].container(border=True) # 图表区
middle_right_cell = cols_middle[1].container(border=True) # 参数区

# --- 右侧：参数控制 ---
with middle_right_cell:
    st.caption("特征参数")
    use_kalman = st.checkbox("卡尔曼滤波", value=True)
    n_lag = st.slider("滞后期数", 0, 365, 1)
    n_MA = st.slider("移动平均窗口", 0, 365, 1)
    n_D = st.slider("差分期数", 0, 365, 1)
    n_yoy = st.selectbox("同比期数(1即为环比)", [0, 1, 12, 52, 252])
    
    st.divider()
    
    # 只有当选择了特征时才显示生成按钮
    if feature_selected:
        if st.button("生成/更新特征", type="primary", use_container_width=True):
            # 加载原始数据
            raw_df = load_and_clean_feature(xl, feature_selected)
            if not raw_df.empty:
                # 计算特征
                st.session_state.features = generate_features(
                    raw_df, n_lag, n_MA, n_D, n_yoy, use_kalman
                )
            else:
                st.error("所选Sheet数据为空或无法解析日期。")

# --- 左侧：结果展示 (表格 + 绘图) ---
with middle_left_cell:
    if not st.session_state.features.empty:
        df_res = st.session_state.features
        st.subheader(f"分析对象: {feature_selected}")
        
        # 显示数据概览
        with st.expander("查看详细数据表"):
            st.dataframe(df_res, use_container_width=True)
        
        # 绘图 (Plotly)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        safe_colors = [
            '#636EFA', # 蓝色 (Blue)
            '#00CC96', # 绿色 (Green)
            '#AB63FA', # 紫色 (Purple)
            '#FFA15A', # 橙色 (Orange)
            '#19D3F3', # 青色 (Cyan)
            '#FF6692', # 粉色 (Pink)
            '#B6E880', # 浅绿 (Light Green)
            '#FEF0D9', # 浅黄
        ]
        
        # 绘制所有列
        for i, col in enumerate(df_res.columns):
            # 轮流从颜色池中取色
            line_color = safe_colors[i % len(safe_colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=df_res.index, 
                    y=df_res[col], 
                    name=col, 
                    mode='lines',
                    # 显式设置颜色
                    line=dict(color=line_color, width=1.5) 
                ),
                secondary_y=False
            )
        
        if 'stock_data' in st.session_state and st.session_state.stock_data is not None:
            stock_df = st.session_state.stock_data
            stock_col_name = st.session_state.get('stock_chosen') 
            
            fig.add_trace(
                go.Scatter(
                    x=stock_df.index,
                    y=stock_df['收盘'], # 取选中的股票列数据
                    name=f"标的: {stock_col_name}", # 显示股票名字
                    mode='lines',
                    line=dict(color='red', width=2), # 标的通常用显眼的颜色(如红色)
                    opacity=0.8
                ),
                secondary_y=True # 股票价格放在右轴
            )
        
        # 4. 统一布局设置
        fig.update_layout(
            height=500, 
            title_text="特征 vs 标的走势对比",
            hovermode="x unified", # 鼠标悬停时同时显示所有数值
            legend=dict(
                orientation="h",   # 图例水平排列
                yanchor="bottom",
                y=1.02,            # 放在图表顶部
                xanchor="right",
                x=1
            ),
            yaxis=dict(title="特征数值"),
            yaxis2=dict(title="股价", showgrid=False) # 右轴不显示网格，防止太乱
        )
        fig.update_xaxes(
            rangeslider_visible=True # 显示底部滑动条
        )
        st.plotly_chart(fig, use_container_width=True)

            
    else:
        st.info("请在右侧设置参数后，点击“生成/更新特征”按钮以查看结果。")
