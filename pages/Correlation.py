import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.processing import (
    Industry_list, SHEET_LIST, fetch_xl_object, 
    load_and_clean_feature, generate_features
)

# --- 页面配置 ---
st.set_page_config(page_title="滞后期预判", layout="wide")

st.title("🔍 滞后期相关性预判")

# --- 初始化 Session State (独立于特征工程页面) ---
if 'corr_xl_object' not in st.session_state:
    st.session_state['corr_xl_object'] = None
if 'corr_features' not in st.session_state:
    st.session_state['corr_features'] = pd.DataFrame()
if 'z_matrix_ic' not in st.session_state:
    st.session_state['z_matrix_ic'] = None
if 'z_matrix_rank' not in st.session_state:
    st.session_state['z_matrix_rank'] = None
if 'lags_list' not in st.session_state:
    st.session_state['lags_list'] = []
if 'windows_list' not in st.session_state:
    st.session_state['windows_list'] = []

# --- 检查股价数据 ---
if 'stock_data' not in st.session_state or st.session_state.stock_data is None:
    st.warning("⚠️ 滞后期预判需要配合股价数据，请先在“数据管理”页面选择标的。")
    st.stop()

stock_df = st.session_state.stock_data

# --- 布局结构 ---
cols_ui = st.columns([1, 1, 2], vertical_alignment="top")
top_left_cell = cols_ui[0].container(border=True, height=300)
top_right1_cell = cols_ui[1].container(border=True, height=300)
top_right2_cell = cols_ui[2].container(border=True, height=300)

# --- 1. 数据源加载 ---
with top_left_cell:
    industry_selected = st.selectbox("选择行业", Industry_list, key="corr_ind")
    SHEET_ID = SHEET_LIST[industry_selected]

    if st.button("同步云端表", width='stretch', key="corr_sync"):
        with st.spinner("正在下载并解析数据..."):
            try:
                st.session_state['corr_xl_object'] = fetch_xl_object(SHEET_ID)
                st.success("同步成功！")
            except Exception as e:
                st.error(f"同步失败: {e}")

# --- 2. 选择具体特征 ---
with top_right1_cell:
    xl = st.session_state['corr_xl_object']
    if xl is None:
        st.warning("请先点击“同步云端表”。")
    else:
        feature_selected = st.pills("选择特征", xl.sheet_names, selection_mode="single", key="corr_feat_pill")
        if not feature_selected:
            st.warning("请先选择一个特征。")

# --- 3. 参数控制 ---
with top_right2_cell:
    st.caption("特征处理与分析范围")
    c1, c2, c3, c4 = st.columns([1, 1.2, 1, 1])
    
    with c1:
        st.write("**1. 滤波**")
        use_kalman = st.checkbox("卡尔曼滤波", value=True, key="corr_kalman")
        
    with c2:
        st.write("**2. 变换**")
        n_yoy_pills = st.pills("同环比", [0, 1, 12, 52, 252], selection_mode="single", default=0, key="corr_yoy_pills")
        n_D = st.number_input("差分期", 0, 365, 0, key="corr_d")
            
    with c3:
        st.write("**3. 预判范围**")
        n_scan = st.number_input("特征滞后预判跨度", 0, 252, 20, key="corr_scan")
        m_target = st.number_input("累计收益同比跨度", 0, 252, 20, key="corr_m_target")

    with c4:
        st.write("**4. 平滑**")
        n_MA = st.number_input("MA窗口", 0, 365, 0, key="corr_ma")

    if st.session_state.get("corr_feat_pill"):
        if st.button("生成特征并分析", type="primary", width='stretch', key="corr_gen"):
            raw_df = load_and_clean_feature(xl, st.session_state.corr_feat_pill)
            if not raw_df.empty:
                # 1. 生成特征
                st.session_state.corr_features = generate_features(
                    raw_df, 0, n_MA, n_D, [n_yoy_pills] if n_yoy_pills > 0 else [], use_kalman
                )
                
                # 2. 立即触发热力图矩阵计算
                df_res = st.session_state.corr_features
                target_feat_tmp = [c for c in df_res.columns if c not in ['原始数据', '卡尔曼滤波', '数值']][-1]
                
                price_raw = stock_df['累计超额收益'] if '累计超额收益' in stock_df.columns else stock_df['收盘']
                f_data_raw = df_res[target_feat_tmp].dropna()
                common_idx = price_raw.index.intersection(f_data_raw.index)
                
                if len(common_idx) >= 20:
                    p_data = price_raw.loc[common_idx]
                    f_data = f_data_raw.loc[common_idx]
                    lags = list(range(1, n_scan + 1))
                    windows = list(range(1, m_target + 1))
                    
                    z_ic, z_rank = [], []
                    with st.spinner("正在全局扫描相关性矩阵..."):
                        for w in windows:
                            target_w = p_data if w == 0 else p_data.pct_change(w)
                            row_ic, row_rank = [], []
                            for k in lags:
                                feat_k = f_data.shift(k)
                                mask = target_w.notna() & feat_k.notna()
                                if mask.sum() > 20:
                                    row_ic.append(feat_k[mask].corr(target_w[mask], method='pearson'))
                                    row_rank.append(feat_k[mask].corr(target_w[mask], method='spearman'))
                                else:
                                    row_ic.append(0); row_rank.append(0)
                            z_ic.append(row_ic); z_rank.append(row_rank)
                    
                    st.session_state.z_matrix_ic = z_ic
                    st.session_state.z_matrix_rank = z_rank
                    st.session_state.lags_list = lags
                    st.session_state.windows_list = windows
            else:
                st.error("无法解析数据。")

# --- 结果展示 ---
if not st.session_state.corr_features.empty:
    df_res = st.session_state.corr_features
    
    # 排除过程数据列，供用户选择分析对象
    analysis_features = [c for c in df_res.columns if c not in ['原始数据', '卡尔曼滤波', '数值']]
    if not analysis_features:
        analysis_features = [c for c in df_res.columns]
        
    target_feat = st.selectbox("选择分析特征", analysis_features, key="corr_target_feat")
    
    st.divider()
    
    # 1. 获取原始价格序列 (用于计算向前步长收益率)
    price_raw = stock_df['累计超额收益'] if '累计超额收益' in stock_df.columns else stock_df['收盘']
    f_data_raw = df_res[target_feat].dropna()
    
    # --- 交互区域：手动选择参数以绘图 ---
    st.subheader("📈 组合走势")
    row_plot = st.columns([1, 1, 3])
    with row_plot[0]:
        manual_lag = st.number_input("特征滞后期", 0, n_scan, 0, key="manual_lag")
    with row_plot[1]:
        manual_window = st.number_input("累计超额收益同比处理步长", 0, m_target, 0, key="manual_win")

    # 2. 初始日期对齐
    common_idx = price_raw.index.intersection(f_data_raw.index)
    if len(common_idx) < 20:
        st.error("特征数据与股价数据重叠范围过小，无法分析。")
    else:
        p_data = price_raw.loc[common_idx]
        f_data = f_data_raw.loc[common_idx]

        # ==========================================
        # 📈 第一部分：组合走势
        # ==========================================
        def standard_norm(s): return (s - s.mean()) / s.std()
        
        if manual_window == 0:
            target_selected = p_data
            target_label = "超额净值"
        else:
            target_selected = p_data.pct_change(manual_window)
            target_label = f"{manual_window}期向前同比"
        
        feat_selected = f_data.shift(manual_lag)
        plot_mask = target_selected.notna() & feat_selected.notna()
        plot_idx = common_idx[plot_mask]
        
        if not plot_idx.empty:
            curr_ic = target_selected.loc[plot_idx].corr(feat_selected.loc[plot_idx], method='pearson')
            curr_rank = target_selected.loc[plot_idx].corr(feat_selected.loc[plot_idx], method='spearman')
            
            fig_manual = go.Figure()
            fig_manual.add_trace(go.Scatter(x=plot_idx, y=standard_norm(target_selected.loc[plot_idx]), name=target_label, line=dict(color='red', width=1.5, dash='dot'), opacity=0.7))
            fig_manual.add_trace(go.Scatter(x=plot_idx, y=standard_norm(feat_selected.loc[plot_idx]), name=f"{target_feat}(Lag={manual_lag})", line=dict(color='#636EFA', width=2)))
            fig_manual.update_layout(title=f"走势验证: {target_feat} (Rank IC: {curr_rank:.4f}, IC: {curr_ic:.4f})", height=450, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_manual, use_container_width=True)
        else:
            st.warning("该组合无重叠数据。")

        # ==========================================
        # 🔥 第二部分：热力图
        # ==========================================
        if st.session_state.z_matrix_ic is not None:
            st.divider()
            st.subheader("🔥 相关性热力图")
            
            lags = st.session_state.lags_list
            windows = st.session_state.windows_list
            z_ic = st.session_state.z_matrix_ic
            z_rank = st.session_state.z_matrix_rank
            
            col_ic, col_rank = st.columns(2)
            with col_ic:
                fig_ic = go.Figure(data=go.Heatmap(z=z_ic, x=lags, y=windows, colorscale='RdBu_r', zmin=-1, zmax=1, colorbar=dict(title="IC")))
                fig_ic.update_layout(title="Pearson IC (2D Scan)", xaxis_title="Lag", yaxis_title="Window", height=600)
                st.plotly_chart(fig_ic, use_container_width=True)
            with col_rank:
                fig_rank = go.Figure(data=go.Heatmap(z=z_rank, x=lags, y=windows, colorscale='RdBu_r', zmin=-1, zmax=1, colorbar=dict(title="Rank IC")))
                fig_rank.update_layout(title="Spearman Rank IC (2D Scan)", xaxis_title="Lag", yaxis_title="Window", height=600)
                st.plotly_chart(fig_rank, use_container_width=True)
                
            # 寻找并提示全局最优
            z_rank_np = np.array(z_rank)
            flat_idx = np.argmax(np.abs(z_rank_np))
            w_idx, k_idx = np.unravel_index(flat_idx, z_rank_np.shape)
            st.success(f"💡 **矩阵扫描结果**: 当 Lag={lags[k_idx]}, Window={windows[w_idx]} 时，绝对相关性达到峰值 (Rank IC: {z_rank_np[w_idx, k_idx]:.4f})")
else:
    st.info("请在左边配置参数并点击“生成特征并分析”。")

