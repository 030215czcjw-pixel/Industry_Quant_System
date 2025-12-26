# 文件名: Home.py
import streamlit as st

st.set_page_config(
    page_title="行业择时系统",
)

st.title("行业择时系统")
#st.sidebar.success("请在上方选择一个功能页面。")

st.markdown("""
使用方法：
- 数据管理
    - 选择标的和基准
- 特征工程
    - 选择行业
    - 同步云端表格数据
    - 选择并生成特征
- 策略回测
    - 设置策略逻辑
    - 运行回测
""")

st.divider()

st.markdown("""
    注意事项：
    - 表格
        - 表格名称为数据名称
        - 表格内第一列放日期，第二列放原始值（如果加载错误试试把第一列列名改为 日期、Date或time）
        - 煤炭 https://docs.google.com/spreadsheets/d/1P3446_9mBi-7qrAMi78F1gHDHGIOCjw-/edit?usp=drive_link&ouid=116487443839473589964&rtpof=true&sd=true 在这更新！
        - 交运 https://docs.google.com/spreadsheets/d/1VVTAG1ixDe50ysjMZEAAZyvYkUbiHBvolh0oaYn8Mxw/edit?usp=drive_link 在这更新！
        - 上面的要挂🪜
    - 数据日期是取特征、标的、基准的交集
    - 选取卡尔曼滤波之后，后续计算均基于滤波后的数据
""")

st.divider()

st.markdown("""
    - 源码：https://github.com/030215czcjw-pixel/Industry_Quant_System
""")
