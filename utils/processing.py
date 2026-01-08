import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import streamlit as st

Industry_list = ["煤炭", "交运"]
SHEET_LIST = {
    "交运": "1VVTAG1ixDe50ysjMZEAAZyvYkUbiHBvolh0oaYn8Mxw", 
    "煤炭": "1P3446_9mBi-7qrAMi78F1gHDHGIOCjw-"
} 

def apply_kalman(series, Q_val=0.01, R_val=0.1):
    vals = series.ffill().bfill().to_numpy()
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

def generate_features(data, n_lag, n_MA, n_D, n_yoy, use_kalman):
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
        working_df['数值'] = base_series

    if n_lag > 0:
        for col in working_df.columns:
            working_df[col] = working_df[col].shift(n_lag)
            working_df.rename(columns={col: f"{col}_Lag{n_lag}"}, inplace=True)

    if n_MA > 0:
        for col in list(working_df.columns):
            working_df[f'{col}_MA{n_MA}'] = working_df[col].rolling(window=n_MA).mean()
            
    return pd.concat([df, working_df], axis=1)

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
