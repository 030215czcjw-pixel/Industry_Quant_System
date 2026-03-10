import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from filterpy.kalman import KalmanFilter

# --- 配置与常量 ---
Industry_list = ["煤炭", "焦煤", "交运", "干散货", "游戏"]
SHEET_LIST = {
    "交运": "1VVTAG1ixDe50ysjMZEAAZyvYkUbiHBvolh0oaYn8Mxw", 
    "焦煤": "1_ye6vJegv8KMzDQLjaBDlf-GDkD7s9dfr3p2Yyr0dNw",
    "煤炭": "1P3446_9mBi-7qrAMi78F1gHDHGIOCjw-",
    "干散货": "1_5szzrppATh1SFSk3MtxmkSo6XKUsV6x8WcJ6kOUsNc",
    "游戏": "1jV7SLw7fOrKmH8SkGYOMtYZL-4ig1E-DxK77jb32N5Y"
} 

def apply_kalman(series, Q_val=0.01, R_val=0.1):
    if isinstance(series, pd.DataFrame):
        target = series.iloc[:, 0]
    else:
        target = series
    vals = target.ffill().bfill().to_numpy()
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[vals[0]]])
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.P *= 10.
    kf.R = R_val
    kf.Q = Q_val
    filtered_results = []
    for z in vals:
        kf.predict()
        kf.update(z)
        filtered_results.append(kf.x[0, 0])
    return pd.Series(filtered_results, index=target.index, name=getattr(target, 'name', 'filtered'))

def apply_feature_transforms(series, mode="无"):
    """
    数值变换处理 (单选互斥)
    - 5%缩尾: 基于分位数
    - 3-Sigma: 基于均值标准差
    - Log Transform: 对数处理 (针对特征变换后的数据量级进行处理)
    - Z-score: 标准化
    - Robust Scaling: 基于中位数和MAD的标准化
    """
    if series.empty or mode == "无":
        return series
    
    s = series.copy()
    if mode == "5%缩尾":
        s = s.clip(s.quantile(0.05), s.quantile(0.95))
    elif mode == "3-Sigma缩尾":
        mu, sigma = s.mean(), s.std()
        s = s.clip(mu - 3 * sigma, mu + 3 * sigma)
    elif mode == "Log Transform":
        # 对数变换，处理正负值 (针对特征变换后的数据量级进行处理)
        s = np.sign(s) * np.log1p(np.abs(s))
    elif mode == "Z-score标准化":
        s = (s - s.mean()) / (s.std() + 1e-9)
    elif mode == "Robust Scaling":
        median = s.median()
        mad = (s - median).abs().median()
        # 1.4826 为标准正态分布下的缩放因子
        s = (s - median) / (1.4826 * mad + 1e-9)
    return s

def generate_features(data, n_lag, n_MA, n_D, n_yoy, use_kalman, transform_method="无"):
    df = pd.DataFrame(index=data.index)
    df['原始数据'] = data.iloc[:, 0].astype('float64')
    if use_kalman:
        df['卡尔曼滤波'] = apply_kalman(df['原始数据'])
        base_series = df['卡尔曼滤波'] 
    else:
        base_series = df['原始数据']
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
        working_df['原始数据'] = base_series
    
    if transform_method != "无":
        for col in working_df.columns:
            working_df[col] = apply_feature_transforms(working_df[col], transform_method)

    if n_lag > 0:
        for col in working_df.columns:
            working_df[col] = working_df[col].shift(n_lag)
            working_df.rename(columns={col: f"{col}_Lag{n_lag}"}, inplace=True)
    if n_MA > 0:
        for col in list(working_df.columns):
            working_df[f'{col}_MA{n_MA}'] = working_df[col].rolling(window=n_MA).mean()
    return pd.concat([working_df], axis=1)

def load_and_clean_feature(xl_obj, sheet_name):
    try:
        df = xl_obj.parse(sheet_name)
        for col in df.columns:
            if '日期' in str(col) or 'Date' in str(col) or 'time' in str(col).lower():
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                return df
        return df
    except Exception as e:
        st.error(f"读取数据出错: {e}")
        return pd.DataFrame()

def fetch_xl_object(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    return pd.ExcelFile(url)

# --- 页面配置 ---
st.set_page_config(page_title="特征工程", layout="wide")

# --- 布局结构 ---
cols = st.columns([0.5, 1, 2], vertical_alignment="top")
top_left_cell = cols[0].container(border=True, height=500)
top_right1_cell = cols[1].container(border=True, height=500)
top_right2_cell = cols[2].container(border=True, height=500)

# --- 初始化 Session State ---
if 'xl_object' not in st.session_state:
    st.session_state['xl_object'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = pd.DataFrame()
if 'feature_pool' not in st.session_state:
    st.session_state['feature_pool'] = pd.DataFrame()

# --- 左上角：数据源加载 ---
with top_left_cell:
    st.session_state['Industry_selected'] = st.selectbox("选择行业", Industry_list)
    SHEET_ID = SHEET_LIST[st.session_state['Industry_selected']]

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
    
    # 采用并排的五列布局展示五个步骤
    c1, c2, c3, c3_5, c4 = st.columns([1, 1.2, 0.8, 1, 1])
    
    with c1:
        st.write("**1. 滤波**")
        use_kalman = st.checkbox("卡尔曼滤波", value=False, help="对原始数据去噪")
        
    with c2:
        st.write("**2. 同比环比差分**")
        if 'yoy_val' not in st.session_state:
            st.session_state['yoy_val'] = 0

        st.pills("同环比周期（可多选）", [1, 12, 52, 252], selection_mode="multi", key="yoy_pills")
        n_yoy_val = st.slider("", 0, 365, key='yoy_val')
        n_D = st.number_input("差分期", 0, 365, 0)
            
    with c3:
        st.write("**3. 滞后**")
        n_lag = st.slider("滞后期", 0, 365, 0, help="特征整体向后平移")

    with c3_5:
        st.write("**4. 数值变换**")
        transform_options = ["无", "5%缩尾", "3-Sigma缩尾", "Log Transform", "Z-score标准化", "Robust Scaling"]
        transform_method = st.selectbox(
            "选择变换", 
            options=transform_options,
            index=0,
            label_visibility="collapsed"
        )

    with c4:
        st.write("**5. 移动平均**")
        n_MA = st.number_input("MA窗口", 0, 365, 0, help="对处理后的序列做平滑")

    # --- 按钮区域 ---
    if feature_selected:
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("📊 生成特征", type="primary", use_container_width=True):
                # 加载原始数据
                raw_df = load_and_clean_feature(xl, feature_selected)
                if not raw_df.empty:
                    # 汇总所有同比环比周期
                    yoy_list = []
                    if st.session_state.get('yoy_pills'):
                        yoy_list.extend(st.session_state.yoy_pills)
                    if n_yoy_val > 0 and n_yoy_val not in yoy_list:
                        yoy_list.append(n_yoy_val)

                    # 计算特征
                    features = generate_features(
                        raw_df, n_lag, n_MA, n_D, yoy_list, use_kalman,
                        transform_method=transform_method
                    )
                    # 对所有特征列应用shift(1)以防止未来数据
                    features_shifted = features.copy()
                    for col in features_shifted.columns:
                        features_shifted[col] = features_shifted[col].shift(1)
                    features_shifted.dropna(inplace=True)
                    st.session_state.features = features_shifted
                    st.success("✅ 特征已生成，请查看下方预览")
                else:
                    st.error("所选Sheet数据为空或无法解析日期。")

        with col_btn2:
            if st.button("➕ 添加到特征池", use_container_width=True, disabled=st.session_state.features.empty):
                if not st.session_state.features.empty:
                    # 给列名添加特征源标识
                    features_copy = st.session_state.features.copy()

                    # 重命名所有列（包括原始数据和卡尔曼滤波）
                    rename_dict = {}
                    for col in features_copy.columns:
                        new_name = f"{feature_selected}_{col}"
                        rename_dict[col] = new_name

                    features_copy.rename(columns=rename_dict, inplace=True)

                    # 合并到特征池（按日期索引对齐）
                    if st.session_state.feature_pool.empty:
                        st.session_state.feature_pool = features_copy
                    else:
                        # 只添加不在特征池中的列
                        new_cols = [c for c in features_copy.columns if c not in st.session_state.feature_pool.columns]
                        if new_cols:
                            st.session_state.feature_pool = pd.concat(
                                [st.session_state.feature_pool, features_copy[new_cols]],
                                axis=1
                            )
                            st.success(f"✅ 已添加 {len(new_cols)} 个特征到特征池")
                        else:
                            st.warning("⚠️ 这些特征已存在于特征池中")


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
            # 与特征数据索引对齐，并处理缺失值
            common_index = df_res.index.intersection(stock_df.index)
            if len(common_index) > 0:
                excess_series = stock_df.loc[common_index, target_col]

                fig.add_trace(
                    go.Scatter(
                        x=common_index,
                        y=excess_series,
                        name="累计超额收益 (右轴1)",
                        mode='lines',
                        line=dict(color='#ff7f0e', width=2),
                        connectgaps=True  # 连接缺失值之间的点
                    ),
                    secondary_y=True
                )
            else:
                st.warning("⚠️ 股价数据与特征数据的日期没有交集，无法显示超额收益")
    else:
        st.warning('提示：在"数据"页面选择标的后，此处可叠加显示超额收益。')

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
else:
    st.info('请在右侧设置参数后，点击"📊 生成特征"按钮以查看结果。')

# --- 特征池管理区域 ---
st.divider()
st.header("🗂️ 特征池管理", divider="rainbow")

if not st.session_state.feature_pool.empty:
    pool_df = st.session_state.feature_pool

    # 统计信息
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("📊 特征数量", len(pool_df.columns))
    col_stat2.metric("📅 数据行数", len(pool_df))
    col_stat3.metric("📆 起始日期", pool_df.index.min().strftime('%Y-%m-%d') if not pool_df.empty else 'N/A')
    col_stat4.metric("📆 结束日期", pool_df.index.max().strftime('%Y-%m-%d') if not pool_df.empty else 'N/A')

    # 操作按钮
    col_op1, col_op2, col_op3 = st.columns([1, 1, 4])

    with col_op1:
        if st.button("🗑️ 清空特征池", type="secondary", use_container_width=True):
            st.session_state.feature_pool = pd.DataFrame()
            st.rerun()

    with col_op2:
        # 下载按钮
        csv = pool_df.to_csv(index=True)
        st.download_button(
            label="📥 下载CSV",
            data=csv,
            file_name=f"feature_pool_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 特征选择和删除
    with st.expander("🔧 管理特征列", expanded=False):
        st.caption("选择要删除的特征（可多选）")

        # 按特征来源分组显示
        feature_groups = {}
        for col in pool_df.columns:
            # 所有特征都应该有前缀格式：特征名_列名
            if '_' in col:
                group = col.split('_')[0]
            else:
                group = '其他'

            if group not in feature_groups:
                feature_groups[group] = []
            feature_groups[group].append(col)

        # 显示分组
        cols_to_delete = []
        for group_name, cols in feature_groups.items():
            with st.expander(f"📁 {group_name} ({len(cols)})", expanded=True):
                group_cols = st.columns(3)
                for i, col in enumerate(cols):
                    if group_cols[i % 3].checkbox(col, key=f"delete_{col}"):
                        cols_to_delete.append(col)

        if cols_to_delete:
            if st.button(f"🗑️ 删除选中的 {len(cols_to_delete)} 个特征", type="secondary"):
                st.session_state.feature_pool.drop(columns=cols_to_delete, inplace=True)
                st.success(f"✅ 已删除 {len(cols_to_delete)} 个特征")
                st.rerun()

    # 数据预览
    with st.expander("📋 查看特征池数据", expanded=False):
        st.dataframe(
            pool_df,
            use_container_width=True,
            height=400
        )

        # 数据统计
        st.caption("数据统计信息")
        st.dataframe(
            pool_df.describe(),
            use_container_width=True
        )

else:
    st.info('💡 特征池为空，请先生成特征并点击"➕ 添加到特征池"按钮')
