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
cols = st.columns([1, 1, 2], vertical_alignment="top")
top_left_cell = cols[0].container(border=True, height=280)
top_right1_cell = cols[1].container(border=True, height=280)
top_right2_cell = cols[2].container(border=True, height=280)

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

    if st.button("同步云端表", width='stretch'):
        with st.spinner("正在下载并解析数据..."):
            try:
                st.session_state['xl_object'] = fetch_xl_object(SHEET_ID)
                st.success("同步成功！")
            except Exception as e:
                st.error(f"同步失败: {e}")

# --- 右上角：选择具体特征 ---
with top_right1_cell:
    xl = st.session_state['xl_object'] # 获取对象
    
    if xl is None:
        st.warning("请先在左侧点击“同步云端表”以加载数据。")
        st.stop() # 停止运行下面的代码，防止报错
    
    if st.session_state.get('feature_selected') is not None:
        default_feature = st.session_state['feature_selected']
    else:
        default_feature = None
    
    try:
        feature_selected = st.pills("选择特征", xl.sheet_names, selection_mode="single", default=default_feature)
        st.session_state.feature_selected = feature_selected
    except:
        feature_selected = st.pills("选择特征", xl.sheet_names, selection_mode="single")
        st.session_state.feature_selected = feature_selected
    
    if not feature_selected:
        st.warning("请先选择一个特征。")

# --- 右侧：参数控制 ---
with top_right2_cell:
    st.caption("特征参数")
    
    # 1. 创建内部两列布局
    col_param_1, col_param_2 = st.columns(2)
    
    # --- 左侧列 ---
    with col_param_1:
        use_kalman = st.checkbox("卡尔曼滤波", value=True)
        n_lag = st.slider("滞后期数", 0, 365, 0)
        n_MA = st.slider("移动平均窗口", 0, 365, 0)

    # --- 右侧列 ---
    with col_param_2:
        # 为了让右侧第一行和左侧对齐，可以加个空的 write 或者直接放控件
        # 这里直接放 Selectbox，视觉上它比较高，能大致对齐
        n_yoy = st.select_slider("同比期数(1即为环比)", [0, 1, 12, 52, 252])
        n_D = st.slider("差分期数", 0, 365, 0)
    
        # --- 按钮区域 (保持通栏) ---
        if feature_selected:
            if st.button("生成/更新特征", type="primary", width='stretch'):
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

if not st.session_state.features.empty:
    df_res = st.session_state.features
    st.subheader(f"分析对象: {st.session_state.feature_selected}")
    
    # 显示数据概览
    with st.expander("查看详细数据表"):
        st.dataframe(df_res, use_container_width=True)
    
    if st.session_state.get('stock_chosen') is None:
        st.warning("请先在 数据 页面选择标的和基准，以便绘制股价和超额收益。")
    # --- 绘图开始 ---
    # 1. 初始化 (开启双轴模式，第三轴需手动配置)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    safe_colors = [
        '#636EFA', '#00CC96', '#AB63FA', '#FFA15A', 
        '#19D3F3', '#FF6692', '#B6E880', '#FEF0D9'
    ]
    
    # 2. 绘制特征线 (左轴 y1)
    for i, col in enumerate(df_res.columns):
        line_color = safe_colors[i % len(safe_colors)]
        fig.add_trace(
            go.Scatter(
                x=df_res.index, 
                y=df_res[col], 
                name=f"特征: {col}", 
                mode='lines',
                line=dict(color=line_color, width=1.5) 
            ),
            secondary_y=False
        )
    
    # 3. 检查并读取 session_state 中的数据
    has_stock = 'stock_data' in st.session_state and st.session_state.stock_data is not None
    # 
    target_col = '累计超额收益' 
    has_excess = has_stock and (target_col in st.session_state.stock_data.columns)

    if has_stock:
        stock_df = st.session_state.stock_data
        stock_col_name = st.session_state.get('stock_chosen', '标的') 
        
        # 绘制股价 (右轴1: y2)
        fig.add_trace(
            go.Scatter(
                x=stock_df.index,
                y=stock_df['收盘'],
                name=f"股价: {stock_col_name}",
                mode='lines',
                line=dict(color='red', width=2),
                opacity=0.6 # 设为半透明，避免遮挡
            ),
            secondary_y=True
        )
        
        # 4. 直接调用已计算的超额收益 (右轴2: y3)
        if has_excess:
            fig.add_trace(
                go.Scatter(
                    x=stock_df.index,
                    y=stock_df[target_col], # 直接调用 session_state 里的列
                    name="累计超额收益 (右轴2)",
                    mode='lines',
                    line=dict(color='#ff7f0e', width=2), # 橙色
                    fillcolor='rgba(255, 127, 14, 0.1)', # 极淡橙色
                    yaxis="y3"                           # 【关键】挂载到第三轴
                )
            )

    # 5. 布局设置 (三轴适配)
    # 动态调整 X 轴宽度：如果有超额收益数据，需要给右侧腾出放第三个轴的空间
    domain_end = 0.88 if has_excess else 1.0
    
    layout_config = dict(
        height=600,
        hovermode="x unified",
        xaxis=dict(
            domain=[0, domain_end] # 收缩绘图区
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        
        # 左轴：特征
        yaxis=dict(title="特征数值"),
        
        # 右轴1：股价
        yaxis2=dict(
            title="股价", 
            showgrid=False,
            side="right",
            position=domain_end # 紧贴绘图区右侧
        )
    )

    # 如果有超额收益，配置第三个轴
    if has_excess:
        layout_config['yaxis3'] = dict(
            title=dict(text="累计超额", font=dict(color="#ff7f0e")),
            tickfont=dict(color="#ff7f0e"),
            anchor="free",     # 自由定位
            overlaying="y",    # 叠加在主图上
            side="right",      # 放在右侧
            position=0.96,     # 放在比 yaxis2 更右侧的位置
            showgrid=False
        )

    fig.update_layout(**layout_config)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6月", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(count=5, label="5年", step="year", stepmode="backward"),
                dict(step="all", label="全部")
            ]),
            x=0.83,     # 按钮位置 X
            y=1.1,  # 按钮位置 Y (放在图表上方)
            bgcolor='rgba(255,255,255,0.8)' # 按钮背景
        )
    )
    # 底部滑动条
    fig.update_xaxes(rangeslider_visible=True)
    
    st.plotly_chart(fig, width='stretch')

else:
    st.info("请在右侧设置参数后，点击“生成/更新特征”按钮以查看结果。")