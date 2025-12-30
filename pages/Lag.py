"""
滞后关系分析工具 - Web版
使用Streamlit构建
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import io
import base64

# 设置页面配置
st.set_page_config(
    page_title="滞后关系分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入滞后分析工具
from lag_analysis_tool import LagAnalyzer

# 页面标题
st.title("📊 滞后关系分析工具")
st.markdown("---")
st.markdown("""
这个工具可以帮助您分析两个时间序列之间的滞后关系。
上传您的数据文件，选择要分析的两列，即可获得详细的滞后分析结果。
""")

# 侧边栏 - 文件上传和参数设置
st.sidebar.header("📁 数据上传")
uploaded_file = st.sidebar.file_uploader(
    "选择数据文件",
    type=['csv', 'xlsx', 'xls'],
    help="支持CSV和Excel文件格式"
)

# 参数设置
st.sidebar.header("⚙️ 分析参数")
max_lag = st.sidebar.slider(
    "最大滞后期（月）",
    min_value=1,
    max_value=24,
    value=12,
    help="分析的最大滞后期数"
)

min_points = st.sidebar.slider(
    "最少数据点数",
    min_value=5,
    max_value=50,
    value=10,
    help="计算相关系数所需的最少数据点数"
)

# 主内容区域
if uploaded_file is not None:
    try:
        # 读取数据
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ 成功加载数据文件: {uploaded_file.name}")
        
        # 显示数据预览
        with st.expander("📋 数据预览", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        # 数据列选择
        st.markdown("---")
        st.header("🔍 选择分析指标")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("指标1")
            column1 = st.selectbox(
                "选择第一个指标",
                options=df.columns.tolist(),
                key="col1"
            )
            name1 = st.text_input("指标1名称", value=column1, key="name1")
        
        with col2:
            st.subheader("指标2")
            column2 = st.selectbox(
                "选择第二个指标",
                options=df.columns.tolist(),
                key="col2"
            )
            name2 = st.text_input("指标2名称", value=column2, key="name2")
        
        # 日期列选择（可选）
        st.markdown("---")
        st.subheader("📅 日期设置（可选）")
        
        # 选择是否使用同一日期列
        use_same_date = st.checkbox("两个指标使用同一日期列", value=True)
        
        if use_same_date:
            # 使用同一日期列
            date_column = st.selectbox(
                "选择共同日期列（如果数据中没有日期列，将自动生成）",
                options=["无"] + df.columns.tolist(),
                key="date_col"
            )
            date_column1 = date_column
            date_column2 = date_column
        else:
            # 为每个指标选择单独的日期列
            col_date1, col_date2 = st.columns(2)
            
            with col_date1:
                st.subheader("指标1的日期列")
                date_column1 = st.selectbox(
                    "选择指标1的日期列",
                    options=["无"] + df.columns.tolist(),
                    key="date_col1"
                )
            
            with col_date2:
                st.subheader("指标2的日期列")
                date_column2 = st.selectbox(
                    "选择指标2的日期列",
                    options=["无"] + df.columns.tolist(),
                    key="date_col2"
                )
        
        # 开始分析按钮
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
            if column1 == column2:
                st.error("❌ 请选择两个不同的指标进行分析！")
            else:
                with st.spinner("正在分析数据，请稍候..."):
                    try:
                        # 准备数据
                        data1 = df[column1]
                        data2 = df[column2]
                        
                        # 处理日期
                        dates1 = pd.to_datetime(df[date_column1], errors='coerce') if date_column1 != "无" else None
                        dates2 = pd.to_datetime(df[date_column2], errors='coerce') if date_column2 != "无" else None
                        
                        # 创建分析器
                        analyzer = LagAnalyzer(
                            data1=data1,
                            data2=data2,
                            dates1=dates1,
                            dates2=dates2,
                            name1=name1,
                            name2=name2
                        )
                        
                        # 执行分析
                        lag_df, best_lag, best_corr = analyzer.analyze(
                            max_lag=max_lag,
                            min_points=min_points,
                            output_file=None,  # 不在文件中保存，而是在内存中生成
                            save_results=False
                        )
                        
                        # 检查是否有有效的分析结果
                        if lag_df.empty:
                            st.markdown("---")
                            st.error("❌ 分析失败")
                            st.markdown("\n" + "="*60)
                            st.markdown("**分析无法完成：没有足够的数据点来计算相关系数。**")
                            st.markdown("\n请尝试以下解决方案：")
                            st.markdown("1. **降低最少数据点数参数**（当前值：{}\）".format(min_points))
                            st.markdown("2. **确保选择的指标包含足够的有效数据点**")
                            st.markdown("3. **检查日期列格式是否正确**")
                            st.markdown("4. **确保数据包含足够的时间跨度**")
                            st.markdown("="*60)
                        else:
                            # 显示结果
                            st.markdown("---")
                            st.header("📈 分析结果")
                            
                            # 关键指标
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("最优滞后期", f"{best_lag} 个月")
                            with col2:
                                st.metric("最大相关系数", f"{best_corr:.4f}")
                            with col3:
                                st.metric("有效数据点数", f"{len(analyzer.data1_clean)}")
                            
                            # 解释结果
                            st.markdown("---")
                            st.subheader("💡 结果解释")
                            
                            if best_lag < 0:
                                explanation = f"""
                                - **{name1}滞后 {abs(best_lag)} 个月**时，与{name2}相关性最强（r={best_corr:.4f}）
                                - 这意味着：**{name2}的变化领先于{name1} {abs(best_lag)} 个月**
                                - **预测建议**：可以使用{name2}来预测未来 {abs(best_lag)} 个月的{name1}
                                """
                            elif best_lag > 0:
                                explanation = f"""
                                - **{name2}滞后 {best_lag} 个月**时，与{name1}相关性最强（r={best_corr:.4f}）
                                - 这意味着：**{name1}的变化领先于{name2} {best_lag} 个月**
                                - **预测建议**：可以使用{name1}来预测未来 {best_lag} 个月的{name2}
                                """
                            else:
                                explanation = f"""
                                - 两个指标**同步性最强**（r={best_corr:.4f}）
                                - 这意味着：两个指标几乎同时变化
                                - **预测建议**：可以使用任一指标来预测另一个指标的同期值
                                """
                            
                            st.markdown(explanation)
                            
                            # 可视化图表
                            st.markdown("---")
                            st.subheader("📊 可视化图表")
                            
                            # 生成图表
                            fig = analyzer.visualize(lag_df, best_lag, best_corr, output_file=None)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 详细数据表
                            st.markdown("---")
                            st.subheader("📋 详细数据")
                            
                            # 显示前10个最高相关性
                            st.write("**不同滞后期的相关系数（绝对值前10个最高）**")
                            top_abs_corrs = lag_df.reindex(lag_df['correlation'].abs().nlargest(10).index)
                            st.dataframe(
                                top_abs_corrs[['lag', 'correlation', 'p_value', 'n_points']],
                                use_container_width=True
                            )
                            
                            # 完整数据表
                            with st.expander("查看完整数据表"):
                                st.dataframe(lag_df, use_container_width=True)
                            
                            # 下载结果
                            st.markdown("---")
                            st.subheader("💾 下载结果")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 下载CSV
                                csv = lag_df.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 下载分析结果 (CSV)",
                                    data=csv,
                                    file_name=f"滞后分析_{name1}_vs_{name2}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # 下载图表（重用已生成的fig对象）
                                try:
                                    # 尝试保存为PNG（需要kaleido）
                                    img_bytes = fig.to_image(format="png", width=1800, height=900, scale=2)
                                    st.download_button(
                                        label="📥 下载图表 (PNG)",
                                        data=img_bytes,
                                        file_name=f"滞后分析_{name1}_vs_{name2}.png",
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    # 如果保存PNG失败，提供HTML下载
                                    html_bytes = fig.to_html()
                                    st.download_button(
                                        label="📥 下载图表 (HTML)",
                                        data=html_bytes.encode('utf-8'),
                                        file_name=f"滞后分析_{name1}_vs_{name2}.html",
                                        mime="text/html"
                                    )
                                    st.caption("💡 提示：如需下载PNG格式，请安装 kaleido: pip install kaleido")
                            
                            st.success("✅ 分析完成！")
                        
                    except Exception as e:
                        st.error(f"❌ 分析过程中出现错误: {str(e)}")
                        st.exception(e)
    
    except Exception as e:
        st.error(f"❌ 读取文件时出现错误: {str(e)}")
        st.exception(e)

else:
    # 显示使用说明
    st.info("👈 请在左侧上传数据文件开始分析")
    
    st.markdown("---")
    st.header("📖 使用说明")
    
    st.markdown("""
    ### 使用步骤：
    
    1. **上传数据文件**
       - 支持CSV和Excel格式
       - 确保数据包含至少两列数值数据
    
    2. **选择分析指标**
       - 从下拉菜单中选择要分析的两个指标
       - 可以为指标设置自定义名称
    
    3. **设置日期列（可选）**
       - 如果数据中有日期列，请选择它
       - 如果没有，系统会自动生成日期索引
    
    4. **调整分析参数**
       - 最大滞后期：分析的最大时间滞后范围
       - 最少数据点数：计算相关系数所需的最少数据点
    
    5. **开始分析**
       - 点击"开始分析"按钮
       - 系统会自动计算并显示结果
    
    ### 结果说明：
    
    - **最优滞后期**：相关性最强的滞后期数
      - 负值：表示第一个指标滞后
      - 正值：表示第二个指标滞后
      - 0：表示两个指标同步
    
    - **最大相关系数**：最优滞后期下的相关系数（-1到1之间）
    
    - **可视化图表**：包含6个子图，展示不同角度的分析结果
    
    ### 示例数据：
    
    如果您没有数据文件，可以使用以下示例数据格式：
    
    ```csv
    日期,指标1,指标2
    2020-01-01,100,200
    2020-02-01,105,210
    2020-03-01,110,220
    ...
    ```
    """)
    
    # 提供示例数据下载
    st.markdown("---")
    st.subheader("📥 下载示例数据")
    
    # 生成示例数据
    dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='ME')
    example_data = pd.DataFrame({
        '日期': dates,
        '指标1': np.random.randn(len(dates)).cumsum() + 100,
        '指标2': np.random.randn(len(dates)).cumsum() + 200
    })
    
    csv_example = example_data.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下载示例CSV文件",
        data=csv_example,
        file_name="示例数据.csv",
        mime="text/csv"
    )

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>滞后关系分析工具 | 使用 Streamlit 构建</p>
    </div>
    """,
    unsafe_allow_html=True
)

