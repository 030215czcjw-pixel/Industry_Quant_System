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

    # --- 1. 处理1：滤波层 ---
    # 这一层是所有后续处理的基础
    if use_kalman:
        df['卡尔曼滤波'] = apply_kalman(df['原始数据'])
        base_series = df['卡尔曼滤波'] 
    else:
        base_series = df['原始数据']

    # --- 2. 处理2：转换层 ---
    # 基于滤波后的 base_series 进行同环比、差分变换
    # 如果没有任何变换，我们将 base_series 本身存入一个工作序列
    working_df = pd.DataFrame(index=df.index)
    has_transform = False
    
    if n_D > 0:
        working_df[f'差分{n_D}'] = base_series.diff(n_D)
        has_transform = True
    
    if n_yoy:
        for yoy in n_yoy:
            col_name = f'同比{yoy}' if yoy > 1 else '环比'
            working_df[col_name] = base_series.pct_change(yoy)
            has_transform = True
            
    if not has_transform:
        # 如果没有选任何变换，后续步骤（滞后、MA）作用于 base_series
        # 为了区分，我们给它起个临时名字
        working_df['数值'] = base_series

    # --- 3. 处理3：滞后层 ---
    # 对转换层产生的所有特征进行统一滞后
    if n_lag > 0:
        for col in working_df.columns:
            working_df[col] = working_df[col].shift(n_lag)
            # 重命名以体现滞后
            working_df.rename(columns={col: f"{col}_Lag{n_lag}"}, inplace=True)

    # --- 4. 处理4：均线层 ---
    # 在滞后后的基础上，再次进行移动平均平滑
    if n_MA > 0:
        for col in list(working_df.columns): # 使用 list 避免在迭代时修改
            working_df[f'{col}_MA{n_MA}'] = working_df[col].rolling(window=n_MA).mean()
            
    # 合并结果，保留原始数据和过程数据，其余为最终生成的特征
    return pd.concat([df, working_df], axis=1)

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
top_left_cell = cols[0].container(border=True, height=300)
top_right1_cell = cols[1].container(border=True, height=300)
top_right2_cell = cols[2].container(border=True, height=300)

Industry_list = ["煤炭", "交运"]
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
    st.caption("特征处理")
    
    # 采用并排的四列布局展示四个步骤
    c1, c2, c3, c4 = st.columns([1, 1.2, 1, 1])
    
    with c1:
        st.write("**1. 滤波**")
        use_kalman = st.checkbox("卡尔曼滤波", value=True, help="对原始数据去噪")
        
    with c2:
        st.write("**2. 同比环比差分**")
        if 'yoy_val' not in st.session_state:
            st.session_state['yoy_val'] = 0

        # 快速选择回调逻辑
        def update_yoy_slider():
            if st.session_state.get('yoy_pills'):
                st.session_state.yoy_val = st.session_state.yoy_pills

        st.pills("同环比周期", [1, 12, 52, 252], selection_mode="single", key="yoy_pills", on_change=update_yoy_slider)
        n_yoy_val = st.slider("", 0, 365, key='yoy_val')
        n_D = st.number_input("差分期", 0, 365, 0)
            
    with c3:
        st.write("**3. 滞后**")
        n_lag = st.slider("滞后期", 0, 365, 0, help="特征整体向后平移")
        n_scan = st.number_input("预判跨度", 1, 60, 20, help="向下探测相关性的期数范围")

    with c4:
        st.write("**4. 移动平均**")
        n_MA = st.number_input("MA窗口", 0, 365, 0, help="对处理后的序列做平滑")

    # --- 按钮区域 ---
    if feature_selected:
        if st.button("生成/更新特征", type="primary", width='stretch'):
            # 加载原始数据
            raw_df = load_and_clean_feature(xl, feature_selected)
            if not raw_df.empty:
                # 计算特征
                st.session_state.features = generate_features(
                    raw_df, n_lag, n_MA, n_D, [n_yoy_val] if n_yoy_val > 0 else [], use_kalman
                )
            else:
                st.error("所选Sheet数据为空或无法解析日期。")


# --- 左侧：结果展示 (表格 + 绘图) ---

if not st.session_state.features.empty:
    df_res = st.session_state.features
    st.subheader(f"分析对象: {st.session_state.get('feature_selected', '未选择')}")

    with st.expander("查看详细数据表"):
        st.dataframe(df_res, use_container_width=True)

    # --- 1. 绘图初始化 ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    safe_colors = [
        '#636EFA', '#00CC96', '#AB63FA', '#FFA15A', 
        '#19D3F3', '#FF6692', '#B6E880', '#FEF0D9'
    ]

    # 检查是否需要第三个轴 (y3)
    # 如果有“同比/环比”列，或者有“超额收益”，都需要启用 y3
    ratio_cols = [c for c in df_res.columns if '同比' in c or '环比' in c]

    # 检查超额收益
    stock_chosen = st.session_state.get('stock_chosen')
    has_stock = ('stock_data' in st.session_state) and (st.session_state.stock_data is not None) and (stock_chosen is not None)
    target_col = '累计超额收益' 
    has_excess = has_stock and (target_col in st.session_state.stock_data.columns)

    # 只要有比率特征 OR 有超额收益，就开启 y3
    use_y3 = (len(ratio_cols) > 0) or has_excess

    # --- 2. 绘制特征线 (智能分轴) ---
    for i, col in enumerate(df_res.columns):
        line_color = safe_colors[i % len(safe_colors)]
        
       
        is_ratio = '同比' in col or '环比' in col
        
        if is_ratio:
            # 挂载到 y3 (右侧独立轴)，不和原始数据挤在一起
            fig.add_trace(
                go.Scatter(
                    x=df_res.index, 
                    y=df_res[col], 
                    name=f"特征: {col} (右轴2)", 
                    mode='lines',
                    line=dict(color=line_color, width=1.5),
                    yaxis="y3" # 强制指定到 y3
                )
            )
        else:
            # 原始数据、均线等 -> 留在左轴 (y1)
            fig.add_trace(
                go.Scatter(
                    x=df_res.index, 
                    y=df_res[col], 
                    name=f"特征: {col} (左轴)", 
                    mode='lines',
                    line=dict(color=line_color, width=1.5) 
                ),
                secondary_y=False
            )

    # --- 3. 绘制股价与超额收益 ---
    if has_stock:
        stock_df = st.session_state.stock_data
        
        # (1) 累计超额收益 -> 挂载到右轴1 (y2)
        if has_excess:
            fig.add_trace(
                go.Scatter(
                    x=stock_df.index,
                    y=stock_df[target_col], 
                    name="累计超额收益 (右轴1)",
                    mode='lines',
                    line=dict(color='#ff7f0e', width=2),
                    fillcolor='rgba(255, 127, 14, 0.1)'
                ),
                secondary_y=True
            )
    else:
        st.warning("提示：在“数据”页面选择标的后，此处可叠加显示超额收益。")

    # --- 4. 布局设置 (三轴适配) ---
    # 如果启用了 y3，需要缩短 X 轴给右侧留空间
    domain_end = 0.88 if use_y3 else 1.0

    layout_config = dict(
        height=600,
        hovermode="x unified",
        xaxis=dict(
            domain=[0, domain_end] # 收缩绘图区
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        
        # 左轴 (y1)：原始数据
        yaxis=dict(
            title=dict(text="特征数值", font=dict(color="#636EFA"))
        ),
        
        # 右轴1 (y2)：累计超额收益
        yaxis2=dict(
            title=dict(text="累计超额收益", font=dict(color="#ff7f0e")),
            showgrid=False,
            side="right",
            position=domain_end 
        )
    )

    # 配置第三个轴 (y3)：专门用于 特征变换 (同比/环比)
    if use_y3:
        layout_config['yaxis3'] = dict(
            title=dict(text="同比/环比", font=dict(color="#00CC96")),
            anchor="free",     
            overlaying="y",    
            side="right",      
            position=0.96, # 放在最右边
            showgrid=False,
            tickformat='.2%' # 自动格式化为百分比
        )

    fig.update_layout(**layout_config)

    # 交互组件
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6月", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(step="all", label="全部")
            ]),
            x=0,     
            y=1.15,  
            bgcolor='rgba(255,255,255,0.8)' 
        ),
        rangeslider_visible=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 🔍 滞后期相关性预判
    # ==========================================
    st.divider()
    st.subheader("🔍 滞后期相关性预判")
    
    if has_stock:
        # 1. 准备统计数据
        # 使用超额收益率（或基准模式下的收益率）进行相关性分析，而非直接使用累计价格
        if '累计超额收益' in st.session_state.stock_data.columns:
            # 计算平稳的收益率序列进行相关性分析
            price_series = st.session_state.stock_data['累计超额收益'].pct_change().dropna()
        else:
            price_series = st.session_state.stock_data['收盘'].pct_change().dropna()
            
        # 排除非特征列进行特征选择
        analysis_features = [c for c in df_res.columns if c not in ['原始数据', '卡尔曼滤波']]
        if not analysis_features:
            analysis_features = [c for c in df_res.columns if c in ['卡尔曼滤波', '原始数据']]
        
        if not analysis_features:
            st.info("尚未生成特征，请先点击'生成/更新特征'。")
        else:
            # 选择要分析的单一特征
            target_feat = st.selectbox("选择分析特征", analysis_features)
            
            # --- 数据对齐与同频化处理 ---
            # 获取特征数据并处理频率
            f_data_raw = df_res[target_feat].dropna()
            
            # 将特征数据和收益率数据合并到同一个 DataFrame 以确保日期一一对应
            comparison_df = pd.DataFrame({'feature': f_data_raw, 'target': price_series})
            
            # 处理不同频数据：使用前向填充对齐特征数据（例如月频特征对齐日频收益率）
            # 然后删除仍然存在 NaN 的行（通常是开头部分）
            comparison_df = comparison_df.ffill().dropna()
            
            # 2. 计算 相关系数 (IC & Rank IC)
            lags = range(-5, n_scan + 1)
            ic_list = []
            rank_ic_list = []
            
            for k in lags:
                s_feat = comparison_df['feature'].shift(k)
                # 只有在特征领先/滞后后仍有重叠数据的部分进行计算
                valid_mask = s_feat.notna()
                if valid_mask.sum() > 20: # 提高有效样本阈值
                    # IC (Pearson)
                    ic = s_feat[valid_mask].corr(comparison_df.loc[valid_mask, 'target'], method='pearson')
                    ic_list.append(ic if not np.isnan(ic) else 0)
                    # Rank IC (Spearman)
                    rank_ic = s_feat[valid_mask].corr(comparison_df.loc[valid_mask, 'target'], method='spearman')
                    rank_ic_list.append(rank_ic if not np.isnan(rank_ic) else 0)
                else:
                    ic_list.append(0)
                    rank_ic_list.append(0)
            
            # 3. 绘制热力图
            # 仅从非负滞后 (Lag >= 0) 中筛选最优滞后期数 (默认使用 Rank IC 寻找)
            rank_ic_np = np.array(rank_ic_list)
            lags_np = np.array(list(lags))
            non_neg_mask = lags_np >= 0
            
            if non_neg_mask.any():
                sub_corrs = rank_ic_np[non_neg_mask]
                sub_lags = lags_np[non_neg_mask]
                best_sub_idx = np.argmax(np.abs(sub_corrs))
                best_lag = sub_lags[best_sub_idx]
                best_rank_ic = sub_corrs[best_sub_idx]
                best_ic = np.array(ic_list)[lags_np == best_lag][0]
            else:
                best_lag = 0
                best_rank_ic = 0
                best_ic = 0
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=[rank_ic_list, ic_list],
                x=list(lags),
                y=['Rank IC', 'IC'],
                colorscale='RdBu_r', 
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in rank_ic_list], [f"{v:.2f}" for v in ic_list]],
                texttemplate="%{text}",
                showscale=True
            ))
            fig_heatmap.update_layout(
                title=f"{target_feat} 分析：Rank IC 与 IC 热力图",
                height=300,
                xaxis_title="滞后期数 (Lag)",
                margin=dict(l=50, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.write(f"💡 **预判结果**：在滞后 **{best_lag}** 期时相关性最强。")
            st.write(f"📊 **Rank IC**: {best_rank_ic:.4f} | **IC**: {best_ic:.4f}")
            
            # 4. 绘制对比折线图 (标准化处理)
            # 注意：此处对比图表为了直观依然展示累计趋势，但最优滞后期已由收益率相关性决定
            def standard_norm(s): return (s - s.mean()) / s.std()
            
            # 获取累计价格/收益用于展示
            if '累计超额收益' in st.session_state.stock_data.columns:
                p_display_raw = st.session_state.stock_data['累计超额收益']
            else:
                p_display_raw = st.session_state.stock_data['收盘']
            
            # 使用 intersection 确保索引匹配，防止 KeyError
            common_idx = comparison_df.index.intersection(p_display_raw.index)
            p_display_matched = p_display_raw.loc[common_idx]
            f_display_matched = comparison_df.loc[common_idx, 'feature']
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=common_idx, y=standard_norm(p_display_matched),
                name=f"累计收益形态 (归一化)",
                line=dict(color='red', width=1, dash='dot'),
                opacity=0.5
            ))
            fig_trend.add_trace(go.Scatter(
                x=common_idx, y=standard_norm(f_display_matched.shift(best_lag)),
                name=f"{target_feat} (滞后{best_lag}期, 归一化)",
                line=dict(color='#636EFA', width=2)
            ))
            
            fig_trend.update_layout(
                title=f"最优滞后走势对比 (Lag={best_lag})",
                height=400,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("滞后期预判需要配合股价数据，请先在“数据管理”页面选择标的。")

else:
    st.info("请在右侧设置参数后，点击“生成/更新特征”按钮以查看结果。")