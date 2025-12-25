import streamlit as st

# 这里设置网页的标题和图标
st.set_page_config(page_title="行业择时系统", layout="wide")

# 定义所有的页面路径和对应的中文名称
# 注意：路径要相对于 app.py 的位置
pages = {
    "系统功能": [
        st.Page("Home.py", title="主页"),
        st.Page("pages/Dataset.py", title="数据管理"),
        st.Page("pages/Features.py", title="特征工程"),
        st.Page("pages/Backtest.py", title="策略回测"),
    ]
}

# 启动导航栏
pg = st.navigation(pages)
pg.run()
