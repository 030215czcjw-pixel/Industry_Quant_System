app.py 设置菜单栏
Home.py 主页
pages 里面的是每一页的代码
data 中存放股票和沪深300的数据，但后续尝试更新在线表中

每一页中间传递的变量存在st.session_state中
st.session_state.stock_chosen ：选择股票的名称
st.session_state.base_chosen ：选择基准名称
st.session_state.stock_data ：选择股票的具体数据
st.session_state.base_data ：选择基准的具体数据
st.session_state.xl_object ：获取的在线表格
st.session_state.Industry_selected ：选择的行业
st.session_state.feature_selected ：选择的特征
st.session_state.features ：计算过后的特征，存在df表格里
st.session_state.strategy_expression ：策略表达式
st.session_state.holding_period ：持有期
st.session_state.observation_period ：观察期
st.session_state.profit_target ：目标超额收益
