import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os

class AIStrategyBacktester:
    def __init__(self, stock_data, feature_data):
        """
        初始化回测器，执行数据对齐和基础收益率计算。
        """
        # 1. 数据对齐 (Intersection)
        common_dates = stock_data.index.intersection(feature_data.index).sort_values()
        
        # 保存原始数据副本，以便后续使用
        self.feature_data_aligned = feature_data.loc[common_dates].copy()
        
        # 2. 构建基础价格DataFrame
        self.df = pd.DataFrame({
            '价格': stock_data.loc[common_dates, '收盘']
        }, index=common_dates)
        
        # 3. 计算收益率指标 (预处理)
        self.df['收益率'] = self.df['价格'].pct_change()
        
        # 4. 计算净值曲线
        self.df['净值'] = (1 + self.df['收益率'].fillna(0)).cumprod()
        
        self.df['仓位'] = 0
        
    def run_backtest(self, feature_cols, long_condition, short_condition):
        """
        运行回测，计算未来持有期收益率。
        """
        # 使用副本以免污染原始数据
        df = self.df.copy()
        
        # 合并指定的特征列
        for col in feature_cols:
            if col in self.feature_data_aligned.columns:
                df[col] = self.feature_data_aligned[col]
            else:
                print(f"警告: 特征 {col} 不存在于特征数据中")

        long_signal = eval(long_condition).tolist()
        if short_condition == "0":
            short_signal = [False] * len(long_signal)  # 与做多信号长度一致
        else:
            short_signal = eval(short_condition).tolist()

        df['多单开仓信号触发'] = np.where(long_signal, 1, 0).astype(int)
        df['空单开仓信号触发'] = np.where(short_signal, 1, 0).astype(int)
        
        # 第一步：标记多空信号同时触发的行
        df['多空信号冲突'] = np.where((df['多单开仓信号触发'] == 1) & (df['空单开仓信号触发'] == 1), 1, 0)
        
        # 第二步：重新计算仓位（冲突时赋值0，其余逻辑不变）
        df['仓位'] = np.where(
            df['多空信号冲突'] == 1,  # 优先判断冲突
            0,  # 冲突时仓位为0（无效信号）
            np.where(df['多单开仓信号触发'] == 1, 1,  # 无冲突则按原逻辑
                    np.where(df['空单开仓信号触发'] == 1, -1, 0))
        )
        df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['收益率'].fillna(0))).cumprod()
        
        return df
    
def save_strategy_results(sorted_results, save_format="xlsx"):
    """
    保存策略结果到文件
    
    Args:
        sorted_results: 排序后的策略结果列表
        save_format: 保存格式（xlsx/csv）
    """
    # 整理保存的数据
    save_data = []
    for idx, (strategy_name, metrics) in enumerate(sorted_results, 1):
        save_data.append({
            "策略排名": idx,
            "策略名称": strategy_name,
            "策略描述": metrics["description"],
            "多头条件": metrics["long_condition"],
            "空头条件": metrics["short_condition"],
            "超额收益(%)": round(metrics['excess_gain'] * 100, 2),
            "最大回撤(%)": round(metrics['max_drawdown'] * 100, 2),
            "夏普比率": round(metrics['sharpe_ratio'], 2)
        })
    
    # 转换为DataFrame
    df_save = pd.DataFrame(save_data)
    
    # 生成带时间戳的文件名（避免重复）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"策略回测结果_{timestamp}.{save_format}"
    
    # 保存文件
    try:
        if save_format == "xlsx":
            df_save.to_excel(filename, index=False, engine="openpyxl")
        else:
            df_save.to_csv(filename, index=False, encoding="utf-8-sig")
        
        # 返回文件路径，用于下载
        return filename, None
    except Exception as e:
        return None, f"保存失败：{str(e)}"