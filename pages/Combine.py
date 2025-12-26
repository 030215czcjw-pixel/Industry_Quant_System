import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==========================================
# 配置常量 - 放在文件开头
# ==========================================

# 本地全市场数据默认路径（CSV，与 baysain_factor_analysis_app 一致）
DEFAULT_PRICE_PATH = r"D:\Quant\data\all_stock_data_ts_20140102_20251231.csv"

# 基准指数Google Sheets ID（每个基准一个表格）
BENCHMARK_SHEET_IDS = {
    "沪深300": "1UeNchI2Lh3dycY_6q0xHRKQlgxWMqRn9wFe31JJBrro",
    "中证500": "1_0qA4Gb-xXvsR3q5DIgoyB6qS8DiNvimOPTNEuUZLnw", 
    "上证指数": "1HAyXzomKMupAGiwUdt4qCn61zvdtyIi16Xs1BHeEcCA"
}

# ==========================================
# 工具函数定义
# ==========================================

def fetch_online_sheet(sheet_id):
    """
    从Google Sheets获取数据并返回ExcelFile对象
    参数：sheet_id - Google Sheets的ID字符串
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    return pd.ExcelFile(url)

def fetch_online_dataframe(sheet_id, sheet_name=0):
    """
    从Google Sheets直接获取DataFrame
    参数：
        sheet_id - Google Sheets的ID字符串
        sheet_name - 工作表名称或索引（默认为0，即第一个sheet）
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    return pd.read_excel(url, sheet_name=sheet_name)

def get_data_path(default_path):
    """
    智能查找数据路径：
    1. 优先查找硬编码的绝对路径 (本机开发环境)
    2. 其次查找当前目录下的 data 文件夹 (便于打包/部署)
    3. 最后查找当前目录
    """
    if os.path.exists(default_path):
        return default_path
    filename = os.path.basename(default_path)
    data_subpath = os.path.join("data", filename)
    if os.path.exists(data_subpath):
        return data_subpath
    if os.path.exists(filename):
        return filename
    return None


def normalize_market_df(df, close_target='收盘'):
    """通用日期/收盘列规范化：设日期为索引，生成指定收盘列名称。"""
    if df is None or df.empty:
        return df
    date_col = next((c for c in df.columns if '日期' in str(c) or 'date' in str(c).lower() or 'time' in str(c).lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.rename(columns={date_col: '日期'}).sort_values('日期').set_index('日期')
    close_col = next((c for c in df.columns if 'close' in str(c).lower() or '收盘' in str(c) or 'price' in str(c).lower()), None)
    if close_col and close_target not in df.columns:
        df[close_target] = df[close_col]
    return df

# ==========================================
# 数据处理函数
# ==========================================
def apply_filterpy_kalman(series, Q_val=0.01, R_val=0.1):
    """卡尔曼滤波"""
    from filterpy.kalman import KalmanFilter
    # 确保传入的是 numpy 数组且无空值
    vals = series.fillna(method='ffill').fillna(method='bfill').values
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
    return filtered_results

def FE(original_feature, n_MA, n_D, Y_window, Q_window, feature_name, use_kalman, selected_col=None):
    """
    特征工程：智能识别数值列，避开日期列导致的编码错误
    """
    # 1. 自动筛选数值列 (避开日期类型)
    numeric_df = original_feature.select_dtypes(include=[np.number])
    if numeric_df.empty:
        # 如果没有识别出数字列，尝试暴力转换
        numeric_df = original_feature.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    
    if numeric_df.empty:
        st.error("无法在所选表格中找到数值列，请检查数据格式。")
        return pd.DataFrame()

    if selected_col and selected_col in numeric_df.columns:
        target_col = selected_col
    else:
        target_col = numeric_df.columns[0]
    df = pd.DataFrame(index=original_feature.index)
    # 强制转换为 float64，防止 Timestamp 混入
    df['原始数据'] = numeric_df[target_col].astype(float).ffill().bfill()

    if use_kalman:
        df['卡尔曼滤波'] = apply_filterpy_kalman(df['原始数据'])
        data = df['卡尔曼滤波']
    else:
        data = df['原始数据']
        
    for op in feature_name:
        if op == "移动平均":
            for ma in n_MA:
                if ma > 0:
                    df[f'移动平均{ma}'] = data.rolling(window=ma).mean()
        if op == "差分":
            for d in n_D:
                if d > 0:
                    df[f'差分{d}'] = data.pct_change(periods=d)
        if op == "一阶导数":
            df['一阶导数'] = data.diff(1)
        if op == "二阶导数":
            df['二阶导数'] = data.diff(1).diff(1)
    
    return df

# ==========================================
# 超额收益计算函数
# ==========================================
def set_price_data(stock_data, baselinedata, feature_data, holding_period):
    """计算价格数据和超额收益"""
    # 确保索引对齐
    common_dates = stock_data.index.intersection(baselinedata.index).intersection(feature_data.index).sort_values()
    
    price_data = pd.DataFrame({
        '股价': stock_data.loc[common_dates, '收盘'],
        '基准': baselinedata.loc[common_dates, 'close'],
    }, index=common_dates)
    
    price_data['股价收益率'] = price_data['股价'].pct_change()
    price_data['基准收益率'] = price_data['基准'].pct_change()
    price_data['超额收益率'] = price_data['股价收益率'] - price_data['基准收益率']
    
    # 计算净值
    price_data['超额净值'] = (1 + price_data['超额收益率'].fillna(0)).cumprod()
    price_data['持有期超额收益率'] = price_data['超额净值'].shift(-holding_period) / price_data['超额净值'] - 1
    
    return price_data

# ==========================================
# 核心算法逻辑
# ==========================================
def bayesian_analysis(price_data, feature_data, profit_setted, observation_periods, holding_period, f, s):
    """贝叶斯择时分析"""
    common_dates = price_data.index.intersection(feature_data.index).sort_values()
    df = price_data.loc[common_dates].copy()
    
    for col in f:
        df[col] = feature_data.loc[common_dates, col]
    
    df['胜率触发'] = (df['持有期超额收益率'] > profit_setted).astype(int)
    df['胜率不触发'] = 1 - df['胜率触发']
    
    # 贝叶斯核心计算
    pw_early = df['胜率触发'].rolling(window=observation_periods).mean().shift(holding_period + 1)
    pw_late = df['胜率触发'].rolling(window=observation_periods).mean().shift(holding_period + 1)
    cutoff = observation_periods + holding_period
    df['P(W)'] = pw_early
    if len(df) > cutoff:
        df.iloc[cutoff:, df.columns.get_loc('P(W)')] = pw_late.iloc[cutoff:]
    
    # 安全执行策略逻辑
    try:
        df['信号触发'] = eval(s).astype(int)
    except Exception as e:
        st.error(f"策略表达式错误: {e}")
        df['信号触发'] = 0

    # 条件概率 P(C|W) 和 P(C|not W)
    shift_n = holding_period + 1
    df['W_and_C'] = ((df['胜率触发'] == 1) & (df['信号触发'] == 1)).astype(int)
    df['notW_and_C'] = ((df['胜率触发'] == 0) & (df['信号触发'] == 1)).astype(int)
    
    p_c_w = (df['W_and_C'].rolling(observation_periods).sum().shift(shift_n) / 
             df['胜率触发'].rolling(observation_periods).sum().shift(shift_n))
    p_c_notw = (df['notW_and_C'].rolling(observation_periods).sum().shift(shift_n) / 
                df['胜率不触发'].rolling(observation_periods).sum().shift(shift_n))
    
    df['P(W|C)'] = (p_c_w * df['P(W)']) / (p_c_w * df['P(W)'] + p_c_notw * (1 - df['P(W)']))
    
    # 信号生成与仓位
    df['买入信号'] = np.where(
        (df['P(W|C)'] > df['P(W)']) & (df['信号触发'] == 1) & 
        ((df['P(W|C)'] > 0.5) | (df['P(W|C)'] > df['P(W|C)'].shift(1)*0.9)), 1, 0
    )
    df['仓位'] = np.where(df['买入信号'] == 1, 
                        df['信号触发'].rolling(holding_period).sum() / holding_period, 0)
    
    pos_prev = df['仓位'].fillna(0).shift(1).fillna(0)
    prior_prev = df['P(W)'].fillna(0).shift(1).fillna(0)
    df['仓位净值'] = (1 + (pos_prev * df['超额收益率'].fillna(0))).cumprod()
    df['先验仓位净值'] = (1 + (prior_prev * df['超额收益率'].fillna(0))).cumprod()
    
    return df


# ==========================================
# Streamlit 界面
# ==========================================

st.set_page_config(page_title="贝叶斯择时回测平台", layout="wide")
st.title("贝叶斯择时回测平台")

# 初始化会话状态
if 'feature_data_after' not in st.session_state:
    st.session_state['feature_data_after'] = None
if 'market_preview' not in st.session_state:
    st.session_state['market_preview'] = None

# ==========================================
# 侧边栏：数据源配置
# ==========================================

st.sidebar.header("📁 因子文件上传")

# 1. 因子文件上传
factor_file = st.sidebar.file_uploader("上传因子数据 (Excel)", type=['xlsx', 'xls', 'csv'])
if factor_file is not None:
    try:
        # 根据文件类型选择读取方式
        if factor_file.name.endswith('.csv'):
            df_factor = pd.read_csv(factor_file)
        else:
            df_factor = pd.read_excel(factor_file)
        
        # 自动寻找日期列并设为索引
        for col in df_factor.columns:
            if '日期' in str(col) or 'Date' in str(col) or 'time' in str(col).lower():
                try:
                    df_factor[col] = pd.to_datetime(df_factor[col])
                    df_factor = df_factor.set_index(col)
                except Exception:
                    pass
                break
        st.session_state['raw_feature_df'] = df_factor
        st.sidebar.success("✅ 已上传因子文件")
        st.sidebar.caption(f"列数: {len(df_factor.columns)}")
    except Exception as e:
        st.sidebar.error(f"❌ 读取因子文件失败: {e}")

# 选择用于特征工程的因子列
base_factor_col = None
if 'raw_feature_df' in st.session_state:
    numeric_cols = st.session_state['raw_feature_df'].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        default_base = st.session_state.get('base_factor_col', numeric_cols[0])
        if default_base not in numeric_cols:
            default_base = numeric_cols[0]
        base_factor_col = st.sidebar.selectbox("选择用于特征工程的因子列", numeric_cols, index=numeric_cols.index(default_base))
        st.session_state['base_factor_col'] = base_factor_col
    else:
        st.sidebar.warning("未找到数值型因子列，无法执行特征工程。")

# 2. 本地获取市场数据
if 'price_path' not in st.session_state:
    try:
        with st.spinner("🔄 正在检测本地市场数据路径..."):
            real_price_path = get_data_path(DEFAULT_PRICE_PATH)
            if real_price_path:
                st.session_state['price_path'] = real_price_path
            else:
                st.sidebar.warning("⚠️ 未找到本地市场数据文件")
    except Exception as e:
        st.sidebar.warning(f"⚠️ 自动检测市场数据路径失败: {e}")

st.sidebar.divider()
st.sidebar.subheader("本地市场数据加载")
if st.sidebar.button("加载本地市场数据", use_container_width=True):
    price_path = st.session_state.get('price_path') or get_data_path(DEFAULT_PRICE_PATH)
    if price_path:
        st.session_state['price_path'] = price_path
        try:
            try:
                df_preview = pd.read_csv(price_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df_preview = pd.read_csv(price_path, encoding='gbk')
            st.session_state['market_preview'] = df_preview.head(5)
            st.sidebar.success("✅ 本地市场数据已加载")
        except Exception as e:
            st.sidebar.error(f"❌ 本地市场数据加载失败: {e}")
    else:
        st.sidebar.warning("⚠️ 未检测到本地市场数据文件，请检查路径")

# 3. 输入标的股票代码
stock_selected = st.sidebar.text_input(
    "输入标的股票代码", 
    value="601919.SH",
    placeholder="例如: 601919",
    help="请输入股票代码，如 601919.SH 或 601919"
)

# 4. 选择基准指数
baseline_selected = st.sidebar.selectbox(
    "选择基准指数", 
    list(BENCHMARK_SHEET_IDS.keys()),
    index=0
)

# ==========================================
# 侧边栏：参数配置
# ==========================================

st.sidebar.divider()
st.sidebar.subheader("胜利条件参数")
profit_target = st.sidebar.number_input("胜率阈值（目标超额收益）", value=0.0, step=0.01)

st.sidebar.divider()
st.sidebar.subheader("数据处理参数")
use_kalman = st.sidebar.checkbox("启用卡尔曼滤波", value=False)
features_op = st.sidebar.multiselect(
    "操作算子", 
    ["移动平均", "差分", "一阶导数", "二阶导数"], 
    default=["差分"]
)
n_MA = st.sidebar.slider("移动平均窗口", 0, 60, 0)
n_D = st.sidebar.slider("差分期数", 0, 365, 0)

# 选择需要绘制的因子（可选展示参数）
if st.session_state.get('feature_data_after') is not None:
    available_factors = st.session_state['feature_data_after'].columns.tolist()
    default_factors = st.session_state.get('selected_plot_factors', available_factors)
    selected_factors = st.sidebar.multiselect(
        "选择绘制的因子", 
        available_factors, 
        default=default_factors
    )
    st.session_state['selected_plot_factors'] = selected_factors
else:
    st.sidebar.caption("执行特征工程后可选择绘制的因子。")

st.sidebar.divider()
st.sidebar.subheader("贝叶斯统计参数")
hp = st.sidebar.slider("持有期（以数据频率为单位）", 1, 365, 5)
op = st.sidebar.slider("观察期（以数据频率为单位）", 1, 365, 60)

# 信号选择：从处理后的因子列中选择，再选择简单逻辑
signal_factor_col = st.session_state.get('signal_factor_col')
if st.session_state.get('feature_data_after') is not None:
    factor_cols_for_signal = st.session_state['feature_data_after'].columns.tolist()
    if factor_cols_for_signal:
        default_signal = signal_factor_col if signal_factor_col in factor_cols_for_signal else factor_cols_for_signal[0]
        signal_factor_col = st.sidebar.selectbox("选择信号因子列", factor_cols_for_signal, index=factor_cols_for_signal.index(default_signal))
        st.session_state['signal_factor_col'] = signal_factor_col
else:
    st.sidebar.caption("执行特征工程后可选择信号因子列；暂用默认列。")

logic_option = st.sidebar.selectbox(
    "后验信号逻辑",
    ["因子大于阈值", "因子上升(>前一期)", "因子高于均值窗口", "自定义表达式"],
    index=0
)
threshold_val = None
ma_window_signal = None
custom_logic = None
if logic_option == "因子大于阈值":
    threshold_val = st.sidebar.number_input("阈值", value=0.0, step=0.1)
elif logic_option == "因子高于均值窗口":
    ma_window_signal = st.sidebar.slider("均值窗口", 1, 120, 20)
elif logic_option == "自定义表达式":
    custom_logic = st.sidebar.text_area("自定义表达式 (Python)", value="df > 0")

# ==========================================
# 主界面：数据加载状态
# ==========================================
if st.session_state.get('market_preview') is not None:
    st.success(f"本地市场数据已加载: {st.session_state.get('price_path', '')}")
    with st.expander("本地市场数据预览", expanded=False):
        st.dataframe(st.session_state['market_preview'])
else:
    st.warning("尚未加载本地市场数据")

# 若已生成特征，提前展示因子折线
if st.session_state.get('feature_data_after') is not None:
    st.subheader("因子折线预览")
    preview_df = st.session_state['feature_data_after']
    st.line_chart(preview_df)

# ==========================================
# 主界面：执行按钮
# ==========================================

# 一键执行：特征工程 + 回测分析
if st.button("执行回测分析", use_container_width=True):
    if 'raw_feature_df' not in st.session_state:
        st.error("❌ 请先在左侧上传因子数据！")
    else:
        with st.spinner('🔄 执行回测分析中...'):
            raw_f = st.session_state['raw_feature_df']
            if base_factor_col and base_factor_col in raw_f.columns:
                fe_input = raw_f[[base_factor_col]]
            else:
                fe_input = raw_f
            processed_fe = FE(fe_input, [n_MA], [n_D], 12, 12, features_op, use_kalman, selected_col=base_factor_col)
            st.session_state['feature_data_after'] = processed_fe
            stock_raw = None
            baseline_raw = None
            fe_data = st.session_state['feature_data_after']
            
            try:
                # ========== 读取标的股票数据（本地CSV：全市场） ==========
                price_path = st.session_state.get('price_path')
                if price_path:
                    try:
                        try:
                            df_all = pd.read_csv(price_path, encoding='utf-8-sig')
                        except UnicodeDecodeError:
                            df_all = pd.read_csv(price_path, encoding='gbk')

                        # 识别日期列并统一为 '日期'
                        date_col = next((c for c in df_all.columns if 'date' in str(c).lower() or '日期' in str(c) or 'time' in str(c).lower()), df_all.columns[0])
                        df_all.rename(columns={date_col: '日期'}, inplace=True)
                        df_all['日期'] = pd.to_datetime(df_all['日期'], errors='coerce')
                        df_all = df_all.dropna(subset=['日期'])

                        # 识别代码列并筛选标的
                        code_col = next((c for c in df_all.columns if 'code' in str(c).lower() or 'symbol' in str(c).lower() or '代码' in str(c)), None)
                        stock_df = df_all.copy()
                        if code_col:
                            stock_code = str(stock_selected)
                            stock_df = df_all[df_all[code_col].astype(str) == stock_code]
                            if len(stock_df) == 0 and '.' in stock_code:
                                short_code = stock_code.split('.')[0]
                                stock_df = df_all[df_all[code_col].astype(str) == short_code]
                            if len(stock_df) == 0:
                                short_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
                                if short_code.isdigit():
                                    no_zero_code = str(int(short_code))
                                    stock_df = df_all[df_all[code_col].astype(str) == no_zero_code]
                            if len(stock_df) == 0:
                                st.warning(f"未在全市场数据中找到代码 {stock_selected}，将使用全部数据。")
                                stock_df = df_all.copy()

                        stock_df = normalize_market_df(stock_df, close_target='收盘')
                        if stock_df is None or stock_df.empty or '收盘' not in stock_df.columns:
                            st.error("未找到收盘价列！")
                            raise ValueError("Missing close column")
                        stock_raw = stock_df
                    except Exception as e:
                        st.error(f"读取本地全市场CSV失败: {e}")
                        raise
                else:
                    st.error("未检测到本地市场数据路径，无法读取标的数据。")
                    raise RuntimeError("price_path missing")

                # ========== 读取基准指数数据 ==========
                if baseline_selected in BENCHMARK_SHEET_IDS:
                    benchmark_sheet_id = BENCHMARK_SHEET_IDS[baseline_selected]
                    
                    if benchmark_sheet_id and benchmark_sheet_id != "请替换为沪深300数据表的URL或ID":
                        try:
                            # 从独立的基准指数表格获取数据
                            baseline_df = fetch_online_dataframe(benchmark_sheet_id, sheet_name=0)
                            
                            baseline_df = normalize_market_df(baseline_df, close_target='close')
                            baseline_raw = baseline_df.copy()
                        except Exception as e:
                            st.warning(f"⚠️ 从在线表格加载基准 {baseline_selected} 失败: {e}，尝试本地文件...")
                            # 回退到本地文件
                            baseline_raw = pd.read_excel('stock_data.xlsx', sheet_name=baseline_selected, parse_dates=True)
                            baseline_raw = normalize_market_df(baseline_raw, close_target='close')
                    else:
                        # 基准ID未配置，尝试本地文件
                        st.info(f"ℹ️ 基准指数 {baseline_selected} 未配置在线链接，尝试本地文件...")
                        baseline_raw = pd.read_excel('stock_data.xlsx', sheet_name=baseline_selected, parse_dates=True)
                        baseline_raw = normalize_market_df(baseline_raw, close_target='close')
                else:
                    # 如果不在预设基准中，尝试本地文件
                    baseline_raw = pd.read_excel('stock_data.xlsx', sheet_name=baseline_selected, parse_dates=True)
                    baseline_raw = normalize_market_df(baseline_raw, close_target='close')

            except Exception as e:
                st.error(f"❌ 市场数据读取失败: {e}")
                st.stop()

            # 统一日期索引为日期粒度，避免时间戳不对齐
            for df_tmp in [stock_raw, baseline_raw, fe_data]:
                df_tmp.index = pd.to_datetime(df_tmp.index).normalize()

            # 根据因子频率对价格与基准重采样
            resample_rule = None
            try:
                freq_guess = pd.infer_freq(fe_data.index)
                if freq_guess:
                    if freq_guess.startswith('W'):
                        resample_rule = 'W-FRI'
                    elif freq_guess.startswith('M'):
                        resample_rule = 'M'
                    elif freq_guess.startswith('Q'):
                        resample_rule = 'Q'
            except Exception:
                resample_rule = None

            def resample_price(df_price, price_col, bench_col=None, rule=None):
                if rule is None:
                    return df_price
                df_price = df_price.sort_index()
                agg = {price_col: 'last'}
                if bench_col:
                    agg[bench_col] = 'last'
                df_resampled = df_price.resample(rule).agg(agg)
                df_resampled[price_col] = df_resampled[price_col].ffill()
                if bench_col:
                    df_resampled[bench_col] = df_resampled[bench_col].ffill()
                df_resampled['股价收益率'] = df_resampled[price_col].pct_change()
                if bench_col:
                    df_resampled['基准收益率'] = df_resampled[bench_col].pct_change()
                    df_resampled['超额收益率'] = df_resampled['股价收益率'] - df_resampled['基准收益率']
                return df_resampled

            if resample_rule:
                # 重采样标的价格
                if '股价收益率' in stock_raw.columns:
                    stock_raw = stock_raw.drop(columns=['股价收益率'], errors='ignore')
                stock_raw_resampled = resample_price(stock_raw, '收盘', None, resample_rule)

                # 重采样基准
                if '基准收益率' in baseline_raw.columns:
                    baseline_raw = baseline_raw.drop(columns=['基准收益率'], errors='ignore')
                baseline_raw_resampled = resample_price(baseline_raw, 'close', None, resample_rule)

                stock_raw = stock_raw_resampled
                baseline_raw = baseline_raw_resampled

            # 预检查日期区间
            stock_range = (stock_raw.index.min(), stock_raw.index.max()) if stock_raw is not None else (None, None)
            base_range = (baseline_raw.index.min(), baseline_raw.index.max()) if baseline_raw is not None else (None, None)
            fe_range = (fe_data.index.min(), fe_data.index.max()) if fe_data is not None else (None, None)

            # ========== 数据完整性检查 ==========
            if stock_raw is None or stock_raw.empty:
                st.error("❌ 标的股票数据为空，请检查代码是否正确或本地市场文件是否包含该代码。")
                st.stop()
            if baseline_raw is None or baseline_raw.empty:
                st.error("❌ 基准指数数据为空，请检查基准配置或本地/在线数据。")
                st.stop()
            if fe_data is None or fe_data.empty:
                st.error("❌ 特征工程结果为空，请检查因子数据或参数设置。")
                st.stop()

            # 构建后验信号逻辑
            # 选择用于信号的因子列，若未预先选择则默认第一列
            if signal_factor_col and signal_factor_col in fe_data.columns:
                sig_col = signal_factor_col
            else:
                sig_col = fe_data.columns[0]

            if logic_option == "因子大于阈值":
                val = threshold_val if threshold_val is not None else 0
                s_input = f"(df['{sig_col}'] > {val})"
            elif logic_option == "因子上升(>前一期)":
                s_input = f"(df['{sig_col}'] > df['{sig_col}'].shift(1))"
            elif logic_option == "因子高于均值窗口":
                win = ma_window_signal if ma_window_signal is not None else 20
                s_input = f"(df['{sig_col}'] > df['{sig_col}'].rolling({win}).mean())"
            else:
                s_input = custom_logic if custom_logic else f"(df['{sig_col}'] > 0)"

            # ========== 执行回测计算 ==========
            p_data = set_price_data(stock_raw, baseline_raw, fe_data, hp)
            if p_data.empty:
                st.error(
                    "❌ 价格与因子日期交集为空，无法计算，请检查数据日期范围。\n"
                    f"标的数据区间: {stock_range[0]} ~ {stock_range[1]}\n"
                    f"基准数据区间: {base_range[0]} ~ {base_range[1]}\n"
                    f"因子数据区间: {fe_range[0]} ~ {fe_range[1]}"
                )
                st.stop()
            df_res = bayesian_analysis(p_data, fe_data, profit_target, op, hp, fe_data.columns.tolist(), s_input)
            if df_res.empty:
                st.error("❌ 回测结果为空，可能因参数或数据导致无有效样本。")
                st.stop()

            # ========== 结果展示 ==========
            final_nav = df_res['仓位净值'].iloc[-1]
            prior_nav = df_res['先验仓位净值'].iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("策略净值", f"{final_nav:.3f}", f"{(final_nav-1):.2%}")
            c2.metric("先验净值", f"{prior_nav:.3f}", f"{(prior_nav-1):.2%}", delta_color="off")
            c3.metric("超额增益", f"{(final_nav-prior_nav):.2%}")

            # ========== 因子与超额收益走势图 ==========
            st.subheader("因子与超额收益走势")
            fig_factor = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 左轴：超额净值
            fig_factor.add_trace(
                go.Scatter(x=df_res.index, y=df_res['超额净值'], name='超额净值', line=dict(color='blue', width=2)),
                secondary_y=False
            )
            
            # 右轴：因子
            exclude_cols = ['股价', '基准', '股价收益率', '基准收益率', '超额收益率', '超额净值', '持有期超额收益率', 
                          '胜率触发', '胜率不触发', 'P(W)', '信号触发', 'W_and_C', 'notW_and_C', 'P(W|C)', 
                          '买入信号', '仓位', '仓位净值', '先验仓位净值']
            selected_factors = st.session_state.get('selected_plot_factors', [])
            if selected_factors:
                feature_cols = [c for c in selected_factors if c in df_res.columns and c not in exclude_cols]
            else:
                feature_cols = [c for c in df_res.columns if c not in exclude_cols]
            
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
            for i, col in enumerate(feature_cols):
                color = colors[i % len(colors)]
                fig_factor.add_trace(
                    go.Scatter(x=df_res.index, y=df_res[col], name=f'因子: {col}', 
                              line=dict(color=color, width=1, dash='dot')),
                    secondary_y=True
                )
                
            fig_factor.update_yaxes(title_text="超额净值", secondary_y=False)
            fig_factor.update_yaxes(title_text="因子值", secondary_y=True)
            fig_factor.update_layout(height=500, template="plotly_white", hovermode="x unified")
            
            st.plotly_chart(fig_factor, use_container_width=True)

            # ========== 贝叶斯分析结果图 ==========
            fig = make_subplots(
                rows=2, cols=2, 
                subplot_titles=("胜率修正", "净值表现", "信号触发", "实时仓位"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": True}]]
            )
            
            # 子图1: 胜率修正
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W)'], name='先验', 
                                    line=dict(color='orange')), 1, 1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W|C)'], name='后验', 
                                    line=dict(color='grey', dash='dot')), 1, 1)
            
            # 子图2: 净值表现
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['仓位净值'], name='策略仓位净值', 
                                    line=dict(color='red')), 1, 2)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['先验仓位净值'], name='先验仓位净值', 
                                    line=dict(color='grey')), 1, 2)

            # 子图3: 信号触发
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['超额净值'], name='超额净值', 
                                    line=dict(color='blue', width=1.5)), 2, 1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['信号触发'], name='触发脉冲', 
                                    fill='tozeroy', line=dict(width=0),
                                    fillcolor='rgba(255, 165, 0, 0.2)'), 2, 1)
            
            # 子图4: 实时仓位
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['超额净值'], name='超额净值', 
                                    line=dict(color='blue', width=2),
                                    hovertemplate='日期: %{x}<br>超额净值: %{y:.4f}<extra></extra>'), 
                         row=2, col=2, secondary_y=False)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['仓位'], name='策略仓位', 
                                    fill='tozeroy', line_shape='hv', 
                                    line=dict(color='rgba(255, 165, 0, 0.8)', width=1), 
                                    fillcolor='rgba(255, 165, 0, 0.2)', 
                                    hovertemplate='日期: %{x}<br>当前仓位: %{y:.2f}<extra></extra>'), 
                         row=2, col=2, secondary_y=True)
            
            fig.update_yaxes(title_text="净值水平", secondary_y=False, row=2, col=2)
            fig.update_yaxes(title_text="仓位权重", range=[0, 1.1], secondary_y=True, row=2, col=2)
            
            fig.update_layout(height=700, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
