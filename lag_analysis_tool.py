import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from pandas.tseries.frequencies import to_offset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class LagAnalyzer:
    """
    滞后关系分析工具类
    用于分析两个时间序列之间的滞后关系
    支持处理未对齐的日期数据，包括不同频率、日期空值和长短不匹配
    """
    
    def __init__(self, data1, data2, dates1=None, dates2=None, name1='指标1', name2='指标2', 
                 resample_freq=None, interpolation_method='linear'):
        """
        初始化滞后分析器
        
        参数:
        data1: 第一个时间序列（pandas Series 或 numpy array）
        data2: 第二个时间序列（pandas Series 或 numpy array）
        dates1: 第一个时间序列的日期索引（可选，pandas DatetimeIndex或可转换为日期的序列）
        dates2: 第二个时间序列的日期索引（可选，pandas DatetimeIndex或可转换为日期的序列）
        name1: 第一个指标的名称
        name2: 第二个指标的名称
        resample_freq: 重采样频率（可选，如'M'表示月度，'D'表示日度）
        interpolation_method: 插值方法（可选，默认为'linear'）
        """
        self.name1 = name1
        self.name2 = name2
        self.resample_freq = resample_freq
        self.interpolation_method = interpolation_method
        
        # 处理第一个时间序列
        self.series1 = self._create_time_series(data1, dates1, name1)
        
        # 处理第二个时间序列
        self.series2 = self._create_time_series(data2, dates2, name2)
        
        # 对齐两个时间序列
        self.aligned_series1, self.aligned_series2 = self._align_time_series()
        
        # 移除对齐后的缺失值
        valid_mask = ~(np.isnan(self.aligned_series1.values) | np.isnan(self.aligned_series2.values))
        self.data1_clean = self.aligned_series1.values[valid_mask]
        self.data2_clean = self.aligned_series2.values[valid_mask]
        self.dates_clean = self.aligned_series1.index[valid_mask]
        
        print(f"数据1 ({self.name1}): {len(self.series1)} 个原始数据点")
        print(f"数据2 ({self.name2}): {len(self.series2)} 个原始数据点")
        print(f"对齐后有效数据点: {len(self.data1_clean)} 个")
    
    def _create_time_series(self, data, dates, name):
        """
        创建时间序列对象，处理日期和数据类型
        
        参数:
        data: 时间序列数据
        dates: 日期索引
        name: 序列名称
        
        返回:
        pandas Series: 时间序列对象
        """
        if isinstance(data, pd.Series):
            # 如果数据已经是Series，直接使用
            series = data.copy()
            if dates is not None:
                # 如果提供了新的日期，替换索引
                series.index = pd.to_datetime(dates, errors='coerce')
        else:
            # 转换为numpy数组
            data_array = np.array(data)
            
            if dates is not None:
                # 使用提供的日期，强制转换并处理非日期值
                index = pd.to_datetime(dates, errors='coerce')
                # 创建临时DataFrame来处理非日期值
                temp_df = pd.DataFrame({'data': data_array, 'date': index})
                # 移除日期为空的行
                temp_df = temp_df[temp_df['date'].notna()]
                # 重新创建Series
                series = pd.Series(temp_df['data'].values, index=temp_df['date'].values, name=name)
            else:
                # 自动生成日期索引
                index = pd.date_range(start='2000-01-01', periods=len(data_array), freq='M')
                series = pd.Series(data_array, index=index, name=name)
        
        # 移除日期为空的行
        series = series[series.index.notna()]
        
        # 按日期排序
        series = series.sort_index()
        
        return series
    
    def _align_time_series(self):
        """
        对齐两个时间序列，处理不同频率、日期空值和长短不匹配的情况
        
        返回:
        tuple: 对齐后的两个时间序列
        """
        # 合并两个序列的日期范围
        start_date = max(self.series1.index.min(), self.series2.index.min())
        end_date = min(self.series1.index.max(), self.series2.index.max())
        
        # 如果没有重叠日期，使用两个序列的全部日期范围
        if start_date > end_date:
            start_date = min(self.series1.index.min(), self.series2.index.min())
            end_date = max(self.series1.index.max(), self.series2.index.max())
        
        # 确定重采样频率
        if self.resample_freq is None:
            # 自动检测频率
            freq1 = self._detect_frequency(self.series1)
            freq2 = self._detect_frequency(self.series2)
            
            # 选择较高的频率（较短的时间间隔）
            if freq1 and freq2:
                self.resample_freq = self._get_higher_frequency(freq1, freq2)
            elif freq1:
                self.resample_freq = freq1
            elif freq2:
                self.resample_freq = freq2
            else:
                # 默认使用月度频率
                self.resample_freq = 'M'
            
            print(f"自动检测并重采样为: {self.resample_freq}")
        
        # 创建统一的日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.resample_freq)
        
        # 重采样和插值
        aligned1 = self._resample_and_interpolate(self.series1, date_range)
        aligned2 = self._resample_and_interpolate(self.series2, date_range)
        
        return aligned1, aligned2
    
    def _detect_frequency(self, series):
        """
        检测时间序列的频率
        
        参数:
        series: 时间序列
        
        返回:
        str: 检测到的频率（标准频率单位）
        """
        try:
            # 优先使用inferred_freq
            freq = series.index.inferred_freq
            if freq is not None:
                return freq
            
            # 如果没有inferred_freq，计算平均间隔并转换为标准频率
            diffs = series.index.to_series().diff().dropna()
            if len(diffs) > 0:
                avg_diff = diffs.mean()
                
                # 转换为天数
                avg_days = avg_diff.total_seconds() / (24 * 3600)
                
                # 根据平均天数判断标准频率
                if avg_days < 1.5:
                    return 'D'  # 日度
                elif avg_days < 8:
                    return 'W'  # 周度
                elif avg_days < 35:
                    return 'M'  # 月度
                elif avg_days < 100:
                    return 'Q'  # 季度
                else:
                    return 'A'  # 年度
        except Exception as e:
            print(f"频率检测失败: {e}")
            return None
    
    def _get_higher_frequency(self, freq1, freq2):
        """
        比较两个频率，返回较高的频率（较短的时间间隔）
        
        参数:
        freq1: 第一个频率
        freq2: 第二个频率
        
        返回:
        str: 较高的频率
        """
        # 频率优先级（从高到低）
        freq_priority = ['D', 'B', 'W', 'M', 'Q', 'A']
        
        # 提取主要频率单位
        unit1 = freq1[-1] if freq1[-1] in freq_priority else freq1
        unit2 = freq2[-1] if freq2[-1] in freq_priority else freq2
        
        if unit1 in freq_priority and unit2 in freq_priority:
            if freq_priority.index(unit1) < freq_priority.index(unit2):
                return freq1
            else:
                return freq2
        else:
            # 默认返回第一个频率
            return freq1
    
    def _resample_and_interpolate(self, series, date_range):
        """
        对时间序列进行重采样和插值
        
        参数:
        series: 原始时间序列
        date_range: 目标日期范围
        
        返回:
        pandas Series: 重采样和插值后的时间序列
        """
        try:
            # 首先尝试直接与目标日期范围对齐并插值
            # 这种方法更适合处理不同频率的时间序列
            aligned = series.reindex(date_range)
            
            # 如果对齐后缺失值过多，尝试先重采样
            if aligned.isna().sum() > len(aligned) * 0.5:
                try:
                    # 重采样到目标频率
                    resampled = series.resample(self.resample_freq).mean()
                    aligned = resampled.reindex(date_range)
                except:
                    # 重采样失败，继续使用直接对齐的结果
                    pass
            
            # 插值处理
            interpolated = aligned.interpolate(method=self.interpolation_method, limit_direction='both')
            
            return interpolated
        except Exception as e:
            print(f"重采样和插值过程中出现错误: {e}")
            # 最后尝试简单的对齐和线性插值
            return series.reindex(date_range).interpolate(method='linear', limit_direction='both')
    
    def calculate_lag_correlations(self, max_lag=12, min_points=10):
        """
        计算不同滞后期的相关系数
        
        参数:
        max_lag: 最大滞后期（月）
        min_points: 计算相关系数所需的最少数据点数
        
        返回:
        lag_df: DataFrame，包含滞后期、相关系数、p值等信息
        """
        lag_correlations = []
        
        print(f"\n计算滞后期范围: -{max_lag} 到 +{max_lag} 个月")
        print(f"负值表示{self.name1}滞后，正值表示{self.name2}滞后\n")
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # 数据1滞后（向前移动）
                data1_shifted = self.data1_clean[-lag:]
                data2_aligned = self.data2_clean[:len(data1_shifted)]
                data1_aligned = data1_shifted
            elif lag > 0:
                # 数据2滞后（向前移动）
                data2_shifted = self.data2_clean[lag:]
                data1_aligned = self.data1_clean[:len(data2_shifted)]
                data2_aligned = data2_shifted
            else:
                # 无滞后
                data1_aligned = self.data1_clean
                data2_aligned = self.data2_clean
            
            # 确保长度一致
            min_len = min(len(data1_aligned), len(data2_aligned))
            data1_aligned = data1_aligned[:min_len]
            data2_aligned = data2_aligned[:min_len]
            
            # 计算相关系数
            if len(data1_aligned) >= min_points:
                corr, p_value = pearsonr(data1_aligned, data2_aligned)
                lag_correlations.append({
                    'lag': lag,
                    'correlation': corr,
                    'p_value': p_value,
                    'n_points': len(data1_aligned)
                })
        
        lag_df = pd.DataFrame(lag_correlations)
        
        if lag_df.empty:
            print(f"警告：没有足够的数据点来计算相关系数。请调整min_points参数（当前值：{min_points}）或增加数据量。")
        
        return lag_df
    
    def find_best_lag(self, lag_df):
        """
        找出最优滞后期
        
        参数:
        lag_df: 滞后相关性DataFrame
        
        返回:
        best_lag: 最优滞后期
        best_corr: 最大相关系数
        """
        if lag_df.empty or 'correlation' not in lag_df.columns:
            # 如果没有足够的数据点计算相关系数，返回默认值
            return 0, 0
        
        max_corr_idx = lag_df['correlation'].abs().idxmax()
        best_lag = lag_df.loc[max_corr_idx, 'lag']
        best_corr = lag_df.loc[max_corr_idx, 'correlation']
        return best_lag, best_corr
    
    def align_data_by_lag(self, lag):
        """
        根据滞后期对齐数据
        
        参数:
        lag: 滞后期
        
        返回:
        data1_aligned, data2_aligned, dates_aligned
        """
        if lag < 0:
            # 数据1滞后
            data1_aligned = self.data1_clean[-lag:]
            data2_aligned = self.data2_clean[:len(data1_aligned)]
            dates_aligned = self.dates_clean[:len(data1_aligned)]
        elif lag > 0:
            # 数据2滞后
            data1_aligned = self.data1_clean[:len(self.data2_clean)-lag]
            data2_aligned = self.data2_clean[lag:]
            dates_aligned = self.dates_clean[lag:]
        else:
            # 无滞后
            data1_aligned = self.data1_clean
            data2_aligned = self.data2_clean
            dates_aligned = self.dates_clean
        
        # 确保长度一致
        min_len = min(len(data1_aligned), len(data2_aligned))
        return data1_aligned[:min_len], data2_aligned[:min_len], dates_aligned[:min_len]
    
    def visualize(self, lag_df, best_lag, best_corr, output_file=None):
        """
        生成可视化图表
        
        参数:
        lag_df: 滞后相关性DataFrame
        best_lag: 最优滞后期
        best_corr: 最大相关系数
        output_file: 输出文件名（可选）
        """
        fig = plt.figure(figsize=(18, 12))
        
        lags = lag_df['lag'].values
        corrs = lag_df['correlation'].values
        
        # 1. 滞后相关性柱状图
        ax1 = plt.subplot(2, 3, 1)
        colors = ['red' if c < 0 else 'blue' for c in corrs]
        bars = ax1.bar(lags, corrs, color=colors, alpha=0.6, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax1.axvline(x=best_lag, color='green', linestyle='--', linewidth=2, label=f'最优滞后期={best_lag}')
        ax1.set_xlabel('滞后期（月）', fontsize=12)
        ax1.set_ylabel('相关系数', fontsize=12)
        ax1.set_title('不同滞后期的相关系数', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 时间序列对比（原始数据，标准化）
        ax2 = plt.subplot(2, 3, 2)
        scaler = StandardScaler()
        data1_scaled = scaler.fit_transform(self.data1_clean.reshape(-1, 1)).flatten()
        data2_scaled = scaler.fit_transform(self.data2_clean.reshape(-1, 1)).flatten()
        
        ax2.plot(self.dates_clean, data1_scaled, 'b-', linewidth=1.5, label=f'{self.name1}（标准化）', alpha=0.7)
        ax2.plot(self.dates_clean, data2_scaled, 'r-', linewidth=1.5, label=f'{self.name2}（标准化）', alpha=0.7)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('标准化值', fontsize=12)
        ax2.set_title('时间序列对比（标准化）', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. 最优滞后期对齐后的对比
        ax3 = plt.subplot(2, 3, 3)
        data1_aligned, data2_aligned, dates_aligned = self.align_data_by_lag(best_lag)
        
        data1_aligned_scaled = scaler.fit_transform(data1_aligned.reshape(-1, 1)).flatten()
        data2_aligned_scaled = scaler.fit_transform(data2_aligned.reshape(-1, 1)).flatten()
        
        lag1_label = f'{self.name1}（滞后{abs(best_lag) if best_lag<0 else 0}月）'
        lag2_label = f'{self.name2}（滞后{best_lag if best_lag>0 else 0}月）'
        
        ax3.plot(dates_aligned, data1_aligned_scaled, 'b-', linewidth=1.5, label=lag1_label, alpha=0.7)
        ax3.plot(dates_aligned, data2_aligned_scaled, 'r-', linewidth=1.5, label=lag2_label, alpha=0.7)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_ylabel('标准化值', fontsize=12)
        ax3.set_title(f'最优滞后期对齐后的对比（滞后期={best_lag}月）', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. 散点图（无滞后）
        ax4 = plt.subplot(2, 3, 4)
        min_len = min(len(self.data1_clean), len(self.data2_clean))
        ax4.scatter(self.data1_clean[:min_len], self.data2_clean[:min_len], alpha=0.5, s=30)
        corr_0, _ = pearsonr(self.data1_clean[:min_len], self.data2_clean[:min_len])
        ax4.set_xlabel(self.name1, fontsize=12)
        ax4.set_ylabel(self.name2, fontsize=12)
        ax4.set_title(f'散点图（无滞后，r={corr_0:.4f}）', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(self.data1_clean[:min_len], self.data2_clean[:min_len], 1)
        p = np.poly1d(z)
        ax4.plot(self.data1_clean[:min_len], p(self.data1_clean[:min_len]), "r--", alpha=0.8, linewidth=2, label='趋势线')
        ax4.legend()
        
        # 5. 散点图（最优滞后期）
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(data1_aligned, data2_aligned, alpha=0.5, s=30, color='green')
        corr_best, _ = pearsonr(data1_aligned, data2_aligned)
        ax5.set_xlabel(self.name1, fontsize=12)
        ax5.set_ylabel(self.name2, fontsize=12)
        ax5.set_title(f'散点图（滞后期={best_lag}月，r={corr_best:.4f}）', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(data1_aligned, data2_aligned, 1)
        p = np.poly1d(z)
        ax5.plot(data1_aligned, p(data1_aligned), "r--", alpha=0.8, linewidth=2, label='趋势线')
        ax5.legend()
        
        # 6. 相关性变化趋势
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(lag_df['lag'], lag_df['correlation'], 'o-', linewidth=2, markersize=6, color='purple')
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax6.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax6.axvline(x=best_lag, color='green', linestyle='--', linewidth=2, label=f'最优滞后期={best_lag}')
        ax6.set_xlabel('滞后期（月）', fontsize=12)
        ax6.set_ylabel('相关系数', fontsize=12)
        ax6.set_title('相关系数随滞后期变化', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.suptitle(f'{self.name1} 与 {self.name2} 滞后关系分析', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n可视化图表已保存到: {output_file}")
        
        return fig
    
    def analyze(self, max_lag=12, min_points=10, output_file=None, save_results=True):
        """
        执行完整的滞后分析
        
        参数:
        max_lag: 最大滞后期
        min_points: 最少数据点数
        output_file: 输出图表文件名（可选）
        save_results: 是否保存结果到CSV
        
        返回:
        lag_df: 滞后相关性DataFrame
        best_lag: 最优滞后期
        best_corr: 最大相关系数
        """
        print("="*80)
        print(f"{self.name1} 与 {self.name2} 滞后关系分析")
        print("="*80)
        
        # 计算滞后相关性
        lag_df = self.calculate_lag_correlations(max_lag=max_lag, min_points=min_points)
        
        # 找出最优滞后期
        best_lag, best_corr = self.find_best_lag(lag_df)
        
        # 检查是否有有效的分析结果
        if lag_df.empty:
            print(f"\n分析无法完成：没有足够的数据点来计算相关系数。")
            print(f"请尝试以下解决方案：")
            print(f"1. 降低min_points参数值（当前值：{min_points}）")
            print(f"2. 确保数据包含足够的有效数据点")
            print(f"3. 检查日期列格式是否正确")
            return lag_df, best_lag, best_corr
        
        print(f"\n最优滞后期: {best_lag} 个月")
        print(f"最大相关系数: {best_corr:.4f}")
        
        if best_lag < 0:
            print(f"解释: {self.name1}滞后 {abs(best_lag)} 个月时，与{self.name2}相关性最强")
            print(f"这意味着: {self.name2}的变化领先于{self.name1} {abs(best_lag)} 个月")
        elif best_lag > 0:
            print(f"解释: {self.name2}滞后 {best_lag} 个月时，与{self.name1}相关性最强")
            print(f"这意味着: {self.name1}的变化领先于{self.name2} {best_lag} 个月")
        else:
            print(f"解释: 两个指标同步性最强")
        
        # 生成可视化
        self.visualize(lag_df, best_lag, best_corr, output_file=output_file)
        
        # 保存结果
        if save_results:
            # 清理文件名中的不允许字符
            def clean_filename(filename):
                invalid_chars = '\\/:*?"<>|'
                for char in invalid_chars:
                    filename = filename.replace(char, '_')
                return filename
            
            # 创建安全的文件名
            safe_name1 = clean_filename(self.name1)
            safe_name2 = clean_filename(self.name2)
            results_file = f'滞后分析_{safe_name1}_vs_{safe_name2}.csv'
            
            try:
                lag_df.to_csv(results_file, index=False, encoding='utf-8-sig')
                print(f"\n分析结果已保存到: {results_file}")
            except Exception as e:
                print(f"\n保存结果失败: {e}")
        
        # 打印详细统计
        print(f"\n不同滞后期的相关系数（绝对值前10个最高）:")
        top_abs_corrs = lag_df.reindex(lag_df['correlation'].abs().nlargest(10).index)
        print(top_abs_corrs[['lag', 'correlation', 'p_value', 'n_points']].to_string(index=False))
        
        return lag_df, best_lag, best_corr


# ========== 便捷函数 ==========

def analyze_lag(data1, data2, dates=None, dates1=None, dates2=None, name1='指标1', name2='指标2', 
                 max_lag=12, min_points=10, output_file=None):
    """
    便捷函数：分析两个时间序列的滞后关系
    
    参数:
    data1: 第一个时间序列
    data2: 第二个时间序列
    dates: 日期索引（可选，当两个序列日期相同时使用）
    dates1: 第一个时间序列的日期索引（可选）
    dates2: 第二个时间序列的日期索引（可选）
    name1: 第一个指标的名称
    name2: 第二个指标的名称
    max_lag: 最大滞后期
    min_points: 最少数据点数
    output_file: 输出图表文件名（可选）
    
    返回:
    lag_df, best_lag, best_corr
    """
    # 保持向后兼容性
    if dates is not None:
        analyzer = LagAnalyzer(data1, data2, dates1=dates, dates2=dates, name1=name1, name2=name2)
    else:
        analyzer = LagAnalyzer(data1, data2, dates1=dates1, dates2=dates2, name1=name1, name2=name2)
    return analyzer.analyze(max_lag=max_lag, min_points=min_points, output_file=output_file)


if __name__ == "__main__":
    # 示例用法
    print("滞后分析工具 - 使用示例")
    print("="*80)
    print("\n使用方法：")
    print("1. 导入模块: from lag_analysis_tool import LagAnalyzer, analyze_lag")
    print("2. 创建分析器: analyzer = LagAnalyzer(data1, data2, name1='指标1', name2='指标2')")
    print("3. 执行分析: lag_df, best_lag, best_corr = analyzer.analyze()")
    print("\n或者使用便捷函数：")
    print("lag_df, best_lag, best_corr = analyze_lag(data1, data2, name1='指标1', name2='指标2')")

