"""
洛书对宫检验：5×5偏相关矩阵 + 地理对宫空间相关

核心检验：
1. 五行相空间的洛书对宫约束
   - 水↔火(T轴Z₂): 应反相关
   - 木↔金(风轴): 应反相关
   - 土↔土(rhum轴Z₂): 自对宫
2. 地理对宫：以洛阳为中心重新划分九宫
   - 对宫和=10 → 同一气候变量在对宫位置应反相关
3. 洛阳中心 vs 原始中心 的对比
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = './data/ncep/'
CUG_FILE = './data/CUG-CMA_2.5grid.nc'

# ============================================================
# 编码
# ============================================================
ZY_NAMES = {0:'土', 1:'金', 2:'水', 3:'木', 4:'火'}
QI_NAMES = ['初之气', '二之气', '三之气', '四之气', '五之气', '终之气']
MONTH_TO_QI = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5}

# 洛书九宫→后天八卦→五行
LUOSHU_WUXING = {
    1: '水',  # 坎/北
    2: '土',  # 坤/西南
    3: '木',  # 震/东
    4: '木',  # 巽/东南
    5: '土',  # 中
    6: '金',  # 乾/西北
    7: '金',  # 兑/西
    8: '土',  # 艮/东北
    9: '火',  # 离/南
}

# 对宫对(和=10)
OPPOSITE_PAIRS = [(1,9), (2,8), (3,7), (4,6)]
OPPOSITE_NAMES = {
    (1,9): '水↔火',
    (2,8): '土↔土',
    (3,7): '木↔金',
    (4,6): '木↔金',
}

# 五行→气候变量映射
WUXING_CLIMATE = {
    '水': '寒(TMIN_neg)',
    '火': '暑(TMAX)',
    '木': '风(wspd)',
    '金': '燥(100-rhum)',
    '土': '湿(rhum)',
}

import datetime

# ============================================================
# Part 1: 加载NCEP数据，构造5行五行变量
# ============================================================
print("="*70)
print("PART 1: 构造5行五行气候变量")
print("="*70)

def load_ncep_var(varname, lat_range=(17.5, 42.5), lon_range=(97.5, 122.5)):
    """加载NCEP月均值，返回DataFrame"""
    ds = nc.Dataset(f'{DATA_DIR}{varname}.mon.mean.nc')
    time_var = np.array(ds.variables['time'][:])
    origin = datetime.datetime(1800, 1, 1)
    dates = [origin + datetime.timedelta(hours=float(t)) for t in time_var]
    
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])
    
    data = ds.variables[varname][:, lat_mask, lon_mask]
    # 缺失值过滤
    if varname == 'rhum':
        data = np.where(data < -9000, np.nan, data)
    
    ds.close()
    return data, dates, lat[lat_mask], lon[lon_mask]

# 加载4个基本变量
print("加载TMAX(CUG-CMA)...")
ds_cug = nc.Dataset(CUG_FILE)
cug_lat = ds_cug.variables['lat'][:]
cug_lon = ds_cug.variables['lon'][:]
cug_time = ds_cug.variables['time'][:]
# CUG时间处理
cug_years = np.floor(cug_time).astype(int)
cug_months = np.round((cug_time - cug_years) * 12 + 1).astype(int)
cug_months = np.clip(cug_months, 1, 12)

# 筛选范围
lat_mask_cug = (cug_lat >= 17.5) & (cug_lat <= 42.5)
lon_mask_cug = (cug_lon >= 97.5) & (cug_lon <= 122.5)

TMAX_all = ds_cug.variables['tmax'][:, lat_mask_cug, lon_mask_cug]
TMIN_all = ds_cug.variables['tmin'][:, lat_mask_cug, lon_mask_cug]
ds_cug.close()

print("加载NCEP变量...")
rhum_data, rhum_dates, rhum_lat, rhum_lon = load_ncep_var('rhum')
wspd_data, wspd_dates, wspd_lat, wspd_lon = load_ncep_var('wspd')
uwnd_data, _, _, _ = load_ncep_var('uwnd')
vwnd_data, _, _, _ = load_ncep_var('vwnd')

print(f"  CUG-CMA shape: {TMAX_all.shape}")
print(f"  NCEP shape: {rhum_data.shape}")

# ============================================================
# Part 2: 以洛阳为中心的九宫划分
# ============================================================
print("\n" + "="*70)
print("PART 2: 以洛阳为中心的九宫划分")
print("="*70)

LUOYANG_LAT = 34.62  # 洛阳纬度
LUOYANG_LON = 112.45  # 洛阳经度

# 旧映射（原始脚本中的）
# lat: 35为上下分界, 25为下中分界
# lon: 107.5为左中分界, 112.5为中外分界
OLD_LAT_BOUNDS = (25, 35)
OLD_LON_BOUNDS = (107.5, 112.5)

print(f"洛阳坐标: ({LUOYANG_LAT}°N, {LUOYANG_LON}°E)")
print(f"旧中心: (30°N, 110°E) — lat分界=[25,35], lon分界=[107.5,112.5]")

# 新映射：以洛阳为中心
# 洛阳应在宫5(中宫)，所以中宫区域应包含洛阳
# lat分界: LUOYANG_LAT ± Δlat
# lon分界: LUOYANG_LON ± Δlon
# 选择Δ使九宫面积大致相等且覆盖中国中东部

# 中国中东部范围: lat 17.5-42.5(25°), lon 97.5-122.5(25°)
# 九宫: 3×3, 每格约8.3°×8.3°
# 以洛阳为中心，中宫取5°范围

# 方案A: 以洛阳为中心，对称扩展
DELTA_LAT = 5.0  # 中宫南北5°
DELTA_LON = 5.0  # 中宫东西5°

NEW_LAT_BOUNDS = (LUOYANG_LAT - DELTA_LAT, LUOYANG_LAT + DELTA_LAT)  # (29.62, 39.62)
NEW_LON_BOUNDS = (LUOYANG_LON - DELTA_LON, LUOYANG_LON + DELTA_LON)  # (107.45, 117.45)

print(f"\n新中心(洛阳): lat分界={NEW_LAT_BOUNDS}, lon分界={NEW_LON_BOUNDS}")
print(f"  中宫范围: [{NEW_LAT_BOUNDS[0]}, {NEW_LAT_BOUNDS[1]}]°N × [{NEW_LON_BOUNDS[0]}, {NEW_LON_BOUNDS[1]}]°E")
print(f"  洛阳是否在中宫内: lat={NEW_LAT_BOUNDS[0]}<{LUOYANG_LAT}<{NEW_LAT_BOUNDS[1]}? {NEW_LAT_BOUNDS[0]<LUOYANG_LAT<NEW_LAT_BOUNDS[1]}")
print(f"  洛阳是否在中宫内: lon={NEW_LON_BOUNDS[0]}<{LUOYANG_LON}<{NEW_LON_BOUNDS[1]}? {NEW_LON_BOUNDS[0]<LUOYANG_LON<NEW_LON_BOUNDS[1]}")

# 方案B: 以洛阳为中宫中心，外宫按比例分配
# 中宫5°×5°，外宫按剩余空间3等分
lat_min, lat_max = 17.5, 42.5
lon_min, lon_max = 97.5, 122.5

lat_lo = LUOYANG_LAT - DELTA_LAT
lat_hi = LUOYANG_LAT + DELTA_LAT
lon_lo = LUOYANG_LON - DELTA_LON
lon_hi = LUOYANG_LON + DELTA_LON

# 修正确保在范围内
lat_lo = max(lat_lo, lat_min)
lat_hi = min(lat_hi, lat_max)
lon_lo = max(lon_lo, lon_min)
lon_hi = min(lon_hi, lon_max)

print(f"\n修正后中宫: [{lat_lo:.2f}, {lat_hi:.2f}]°N × [{lon_lo:.2f}, {lon_hi:.2f}]°E")

# ============================================================
# Part 3: 用NCEP数据按新九宫聚合
# ============================================================
print("\n" + "="*70)
print("PART 3: 按洛阳中心九宫聚合NCEP数据")
print("="*70)

def assign_palace_new(lat_arr, lon_arr, lat_lo, lat_hi, lon_lo, lon_hi):
    """以指定中宫范围划分九宫"""
    palaces = np.full(len(lat_arr), -1, dtype=int)
    
    for i in range(len(lat_arr)):
        la, lo = lat_arr[i], lon_arr[i]
        
        # 行: 上/中/下
        if la > lat_hi:
            row = '上'
        elif la < lat_lo:
            row = '下'
        else:
            row = '中'
        
        # 列: 左/中/右
        if lo < lon_lo:
            col = '左'
        elif lo > lon_hi:
            col = '右'
        else:
            col = '中'
        
        # 映射到宫号(标准洛书布局)
        palace_map = {
            ('上','左'): 6, ('上','中'): 1, ('上','右'): 8,
            ('中','左'): 7, ('中','中'): 5, ('中','右'): 3,
            ('下','左'): 2, ('下','中'): 9, ('下','右'): 4,
        }
        palaces[i] = palace_map.get((row, col), -1)
    
    return palaces

# NCEP格点→宫位(新方案)
ncep_lat = rhum_lat
ncep_lon = rhum_lon

# 创建格点网格
lat_grid, lon_grid = np.meshgrid(ncep_lat, ncep_lon, indexing='ij')
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()

palaces_new = assign_palace_new(lat_flat, lon_flat, lat_lo, lat_hi, lon_lo, lon_hi)
palaces_old = assign_palace_new(lat_flat, lon_flat, 25, 35, 107.5, 112.5)

print("新九宫(洛阳中心)格点分布:")
unique, counts = np.unique(palaces_new[palaces_new>0], return_counts=True)
for p, c in zip(unique, counts):
    print(f"  宫{p}({LUOSHU_WUXING.get(p,'?')}): {c}格点")

print("\n旧九宫格点分布:")
unique, counts = np.unique(palaces_old[palaces_old>0], return_counts=True)
for p, c in zip(unique, counts):
    print(f"  宫{p}({LUOSHU_WUXING.get(p,'?')}): {c}格点")

# ============================================================
# Part 4: 按宫位聚合时间序列(年度+六气)
# ============================================================
print("\n" + "="*70)
print("PART 4: 按宫位聚合时间序列")
print("="*70)

# NCEP时间范围: 1948-2020
# 提取年份和月份
ncep_years = np.array([d.year for d in rhum_dates])
ncep_months = np.array([d.month for d in rhum_dates])

# 筛选完整年
year_mask = (ncep_years >= 1948) & (ncep_years <= 2020)

# 对每个宫位，计算六气聚合的变量均值
def aggregate_by_palace_qi(data_3d, palaces_flat, years, months, 
                            lat_shape, lon_shape, year_range=(1948,2020)):
    """按宫位和六气聚合"""
    results = []
    n_lat, n_lon = lat_shape, lon_shape
    
    for palace_id in range(1, 10):
        # 找属于该宫的格点
        grid_mask = palaces_flat.reshape(n_lat, n_lon) == palace_id
        if not grid_mask.any():
            continue
        
        for year in range(year_range[0], year_range[1]+1):
            for qi in range(6):
                qi_months = [m for m, q in MONTH_TO_QI.items() if q == qi]
                mask = (years == year) & np.isin(months, qi_months)
                
                if not mask.any():
                    continue
                
                # 该宫该气的数据
                sub = data_3d[mask][:, grid_mask]
                val = np.nanmean(sub)
                
                results.append({
                    'palace': palace_id,
                    'year': year,
                    'qi': qi,
                    'value': val
                })
    
    return pd.DataFrame(results)

# 聚合各变量
n_lat_ncep, n_lon_ncep = len(ncep_lat), len(ncep_lon)

print("聚合rhum...")
df_rhum = aggregate_by_palace_qi(rhum_data, palaces_new, ncep_years, ncep_months,
                                   n_lat_ncep, n_lon_ncep)
df_rhum.rename(columns={'value': 'rhum'}, inplace=True)

print("聚合wspd...")
df_wspd = aggregate_by_palace_qi(wspd_data, palaces_new, ncep_years, ncep_months,
                                   n_lat_ncep, n_lon_ncep)
df_wspd.rename(columns={'value': 'wspd'}, inplace=True)

print("聚合uwnd...")
df_uwnd = aggregate_by_palace_qi(uwnd_data, palaces_new, ncep_years, ncep_months,
                                   n_lat_ncep, n_lon_ncep)
df_uwnd.rename(columns={'value': 'uwnd'}, inplace=True)

print("聚合vwnd...")
df_vwnd = aggregate_by_palace_qi(vwnd_data, palaces_new, ncep_years, ncep_months,
                                   n_lat_ncep, n_lon_ncep)
df_vwnd.rename(columns={'value': 'vwnd'}, inplace=True)

# CUG-CMA需要不同处理(不同格点)
print("聚合CUG-CMA TMAX/TMIN...")
cug_lat_vals = cug_lat[lat_mask_cug]
cug_lon_vals = cug_lon[lon_mask_cug]
cug_lat_grid, cug_lon_grid = np.meshgrid(cug_lat_vals, cug_lon_vals, indexing='ij')
cug_lat_flat = cug_lat_grid.flatten()
cug_lon_flat = cug_lon_grid.flatten()

palaces_cug_new = assign_palace_new(cug_lat_flat, cug_lon_flat, lat_lo, lat_hi, lon_lo, lon_hi)
n_lat_cug, n_lon_cug = len(cug_lat_vals), len(cug_lon_vals)

df_tmax = aggregate_by_palace_qi(TMAX_all, palaces_cug_new, cug_years, cug_months,
                                   n_lat_cug, n_lon_cug, year_range=(1948,2020))
df_tmax.rename(columns={'value': 'TMAX'}, inplace=True)

df_tmin = aggregate_by_palace_qi(TMIN_all, palaces_cug_new, cug_years, cug_months,
                                   n_lat_cug, n_lon_cug, year_range=(1948,2020))
df_tmin.rename(columns={'value': 'TMIN'}, inplace=True)

# 合并
print("合并数据...")
df_all = df_rhum.copy()
for df_other in [df_wspd, df_uwnd, df_vwnd, df_tmax, df_tmin]:
    df_all = df_all.merge(df_other, on=['palace','year','qi'], how='inner')

# 构造5行五行变量
df_all['寒'] = -df_all['TMIN']       # 水: 寒(负TMIN，越大越寒)
df_all['暑'] = df_all['TMAX']        # 火: 暑(TMAX)
df_all['风'] = df_all['uwnd']        # 木: 风(纬向风)
df_all['燥'] = 100 - df_all['rhum']  # 金: 燥
df_all['湿'] = df_all['rhum']        # 土: 湿

WUXING_COLS = ['寒(水)', '暑(火)', '风(木)', '燥(金)', '湿(土)']
WUXING_KEYS = ['寒', '暑', '风', '燥', '湿']

print(f"\n总记录数: {len(df_all)}")
print(f"宫位: {sorted(df_all['palace'].unique())}")
print(f"年范围: {df_all['year'].min()}-{df_all['year'].max()}")
print(f"六气: {sorted(df_all['qi'].unique())}")

# ============================================================
# Part 5: 5×5偏相关矩阵(全数据)
# ============================================================
print("\n" + "="*70)
print("PART 5: 5×5偏相关矩阵 — 五行变量间")
print("="*70)

from pingouin import partial_corr
import pingouin as pg

# 全数据偏相关(控制季节qi)
df_partial = df_all[WUXING_KEYS + ['qi']].dropna()

# 计算所有对的偏相关(控制qi)
print("\n偏相关矩阵(控制六气qi):")
partial_matrix = np.zeros((5, 5))
p_matrix = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        if i == j:
            partial_matrix[i, j] = 1.0
            p_matrix[i, j] = 0.0
        else:
            # 控制qi和其他3个五行变量
            covars = ['qi'] + [WUXING_KEYS[k] for k in range(5) if k != i and k != j]
            try:
                result = pg.partial_corr(data=df_partial, x=WUXING_KEYS[i], 
                                          y=WUXING_KEYS[j], covar=covars)
                partial_matrix[i, j] = result['r'].values[0]
                p_matrix[i, j] = result['p-val'].values[0]
            except:
                partial_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan

# 打印偏相关矩阵
print("\n5×5偏相关矩阵(控制qi+其余3个五行):")
print(f"{'':>8}", end='')
for k in WUXING_KEYS:
    print(f"{k:>8}", end='')
print()
for i in range(5):
    print(f"{WUXING_KEYS[i]:>8}", end='')
    for j in range(5):
        val = partial_matrix[i, j]
        sig = '*' if p_matrix[i, j] < 0.05 else ''
        print(f"{val:>7.3f}{sig}", end='')
    print()

# ============================================================
# Part 6: 洛书对宫检验 — 五行层面
# ============================================================
print("\n" + "="*70)
print("PART 6: 洛书对宫检验 — 五行变量层面")
print("="*70)

# 洛书对宫在五行空间的对应:
# 水(寒) ↔ 火(暑): indices 0↔1
# 木(风) ↔ 金(燥): indices 2↔3
# 土(湿) 自对宫: index 4

print("\n洛书对宫五行对:")
print(f"  水(寒)↔火(暑): 偏相关 = {partial_matrix[0,1]:.4f}, p = {p_matrix[0,1]:.4f}")
print(f"  木(风)↔金(燥): 偏相关 = {partial_matrix[2,3]:.4f}, p = {p_matrix[2,3]:.4f}")
print(f"  土(湿)自对宫: 偏相关 = {partial_matrix[4,4]:.4f}")

# 预测:
# 水↔火应反相关(寒暑对立): partial_matrix[0,1] < 0?
# 木↔金应反相关(风燥对立): partial_matrix[2,3] < 0?
print(f"\n水↔火反相关? {'✓' if partial_matrix[0,1] < 0 else '✗'} (r={partial_matrix[0,1]:.4f})")
print(f"木↔金反相关? {'✓' if partial_matrix[2,3] < 0 else '✗'} (r={partial_matrix[2,3]:.4f})")

# 扩展: 相生链(最近邻)和相克链(次近邻)
# 五行循环: 水→木→火→土→金→水
# 生链: 水→木, 木→火, 火→土, 土→金, 金→水
# 克链: 水→火, 火→金, 金→木, 木→土, 土→水

# 在Z₅编码: 土=0, 金=1, 水=2, 木=3, 火=4
# 生链: a→(a+4)%5 (或 a→(a-1)%5)
# 克链: a→(a+3)%5 (或 a→(a-2)%5)

# 在WUXING_KEYS顺序: 寒=水=2, 暑=火=4, 风=木=3, 燥=金=1, 湿=土=0
# 映射到Z₅: 寒→2, 暑→4, 风→3, 燥→1, 湿→0
z5_map = {'寒': 2, '暑': 4, '风': 3, '燥': 1, '湿': 0}

print("\n生链(最近邻, a→a-1)偏相关:")
sheng_pairs = [('暑','风'), ('风','寒'), ('寒','燥'), ('燥','湿'), ('湿','暑')]
for a, b in sheng_pairs:
    i, j = WUXING_KEYS.index(a), WUXING_KEYS.index(b)
    print(f"  {a}→{b}: r={partial_matrix[i,j]:.4f}, p={p_matrix[i,j]:.4f}")

print("\n克链(次近邻, a→a-2)偏相关:")
ke_pairs = [('暑','燥'), ('燥','风'), ('风','湿'), ('湿','寒'), ('寒','暑')]
for a, b in ke_pairs:
    i, j = WUXING_KEYS.index(a), WUXING_KEYS.index(b)
    print(f"  {a}克{b}: r={partial_matrix[i,j]:.4f}, p={p_matrix[i,j]:.4f}")

# ============================================================
# Part 7: 地理对宫检验 — 同一变量在对宫位置的相关性
# ============================================================
print("\n" + "="*70)
print("PART 7: 地理对宫检验 — 以洛阳为中心")
print("="*70)

# 对每个变量，计算对宫位置的年际相关
# 即: 宫1的值 vs 宫9的值(同一年、同一气)

for var in WUXING_KEYS:
    print(f"\n--- {var} ---")
    for p1, p2 in OPPOSITE_PAIRS:
        d1 = df_all[df_all['palace']==p1][['year','qi',var]].rename(columns={var: f'P{p1}'})
        d2 = df_all[df_all['palace']==p2][['year','qi',var]].rename(columns={var: f'P{p2}'})
        
        if len(d1) == 0 or len(d2) == 0:
            print(f"  宫{p1}↔宫{p2}: 数据缺失")
            continue
        
        merged = d1.merge(d2, on=['year','qi'])
        if len(merged) < 10:
            print(f"  宫{p1}↔宫{p2}: 样本不足(n={len(merged)})")
            continue
        
        r, p = stats.pearsonr(merged[f'P{p1}'], merged[f'P{p2}'])
        element_pair = OPPOSITE_NAMES.get((p1,p2), '')
        sign_ok = '✓反相关' if r < 0 else '✗正相关'
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        print(f"  宫{p1}↔宫{p2}({element_pair}): r={r:.4f}, p={p:.4f} {sig} {sign_ok}")

# ============================================================
# Part 8: 对宫互补检验 — 对宫和是否恒定
# ============================================================
print("\n" + "="*70)
print("PART 8: 对宫互补检验 — 对宫值之和是否近似恒定")
print("="*70)

for var in WUXING_KEYS:
    print(f"\n--- {var} ---")
    for p1, p2 in OPPOSITE_PAIRS:
        d1 = df_all[df_all['palace']==p1][['year','qi',var]].rename(columns={var: f'P{p1}'})
        d2 = df_all[df_all['palace']==p2][['year','qi',var]].rename(columns={var: f'P{p2}'})
        
        if len(d1) == 0 or len(d2) == 0:
            continue
        
        merged = d1.merge(d2, on=['year','qi'])
        if len(merged) < 10:
            continue
        
        s = merged[f'P{p1}'] + merged[f'P{p2}']
        cv = s.std() / s.mean() * 100  # 变异系数
        element_pair = OPPOSITE_NAMES.get((p1,p2), '')
        print(f"  宫{p1}+宫{p2}({element_pair}): mean={s.mean():.2f}, std={s.std():.2f}, CV={cv:.1f}%")

# ============================================================
# Part 9: 旧中心 vs 新中心(洛阳)对比
# ============================================================
print("\n" + "="*70)
print("PART 9: 旧中心 vs 洛阳中心 — 对宫相关对比")
print("="*70)

# 用旧映射重新聚合(仅对宫相关)
palaces_old_flat = palaces_old  # 已有

# 为旧映射也聚合
df_rhum_old = aggregate_by_palace_qi(rhum_data, palaces_old_flat, ncep_years, ncep_months,
                                       n_lat_ncep, n_lon_ncep)
df_rhum_old.rename(columns={'value': 'rhum'}, inplace=True)
df_wspd_old = aggregate_by_palace_qi(wspd_data, palaces_old_flat, ncep_years, ncep_months,
                                       n_lat_ncep, n_lon_ncep)
df_wspd_old.rename(columns={'value': 'wspd'}, inplace=True)
df_uwnd_old = aggregate_by_palace_qi(uwnd_data, palaces_old_flat, ncep_years, ncep_months,
                                       n_lat_ncep, n_lon_ncep)
df_uwnd_old.rename(columns={'value': 'uwnd'}, inplace=True)
df_vwnd_old = aggregate_by_palace_qi(vwnd_data, palaces_old_flat, ncep_years, ncep_months,
                                       n_lat_ncep, n_lon_ncep)
df_vwnd_old.rename(columns={'value': 'vwnd'}, inplace=True)

# CUG旧映射
palaces_cug_old = assign_palace_new(cug_lat_flat, cug_lon_flat, 25, 35, 107.5, 112.5)
df_tmax_old = aggregate_by_palace_qi(TMAX_all, palaces_cug_old, cug_years, cug_months,
                                       n_lat_cug, n_lon_cug, year_range=(1948,2020))
df_tmax_old.rename(columns={'value': 'TMAX'}, inplace=True)
df_tmin_old = aggregate_by_palace_qi(TMIN_all, palaces_cug_old, cug_years, cug_months,
                                       n_lat_cug, n_lon_cug, year_range=(1948,2020))
df_tmin_old.rename(columns={'value': 'TMIN'}, inplace=True)

df_old = df_rhum_old.copy()
for df_other in [df_wspd_old, df_uwnd_old, df_vwnd_old, df_tmax_old, df_tmin_old]:
    df_old = df_old.merge(df_other, on=['palace','year','qi'], how='inner')

df_old['寒'] = -df_old['TMIN']
df_old['暑'] = df_old['TMAX']
df_old['风'] = df_old['uwnd']
df_old['燥'] = 100 - df_old['rhum']
df_old['湿'] = df_old['rhum']

print(f"\n旧映射宫位: {sorted(df_old['palace'].unique())}")

print("\n旧中心对宫相关:")
for var in WUXING_KEYS:
    for p1, p2 in OPPOSITE_PAIRS:
        d1 = df_old[df_old['palace']==p1][['year','qi',var]].rename(columns={var: f'P{p1}'})
        d2 = df_old[df_old['palace']==p2][['year','qi',var]].rename(columns={var: f'P{p2}'})
        
        merged = d1.merge(d2, on=['year','qi'])
        if len(merged) < 10:
            continue
        
        r, p = stats.pearsonr(merged[f'P{p1}'], merged[f'P{p2}'])
        element_pair = OPPOSITE_NAMES.get((p1,p2), '')
        print(f"  {var}: 宫{p1}↔宫{p2}({element_pair}): r={r:.4f}, p={p:.4f}")

print("\n新中心(洛阳)对宫相关:")
for var in WUXING_KEYS:
    for p1, p2 in OPPOSITE_PAIRS:
        d1 = df_all[df_all['palace']==p1][['year','qi',var]].rename(columns={var: f'P{p1}'})
        d2 = df_all[df_all['palace']==p2][['year','qi',var]].rename(columns={var: f'P{p2}'})
        
        merged = d1.merge(d2, on=['year','qi'])
        if len(merged) < 10:
            continue
        
        r, p = stats.pearsonr(merged[f'P{p1}'], merged[f'P{p2}'])
        element_pair = OPPOSITE_NAMES.get((p1,p2), '')
        print(f"  {var}: 宫{p1}↔宫{p2}({element_pair}): r={r:.4f}, p={p:.4f}")

# ============================================================
# Part 10: 5×5偏相关按对宫五行分组汇总
# ============================================================
print("\n" + "="*70)
print("PART 10: 洛书对宫框架下的5×5偏相关汇总")
print("="*70)

# 洛书对宫五行配对:
# 宫1(水)↔宫9(火): 水↔火 = 寒↔暑
# 宫2(土)↔宫8(土): 土↔土 = 湿↔湿
# 宫3(木)↔宫7(金): 木↔金 = 风↔燥
# 宫4(木)↔宫6(金): 木↔金 = 风↔燥

# 对宫在五行层面的预测:
# 1. 水↔火: 洛书对宫和=10，预测寒暑互补 → 偏相关<0
# 2. 土↔土: 自对宫，无方向预测
# 3. 木↔金: 洛书对宫和=10，预测风燥互补 → 偏相关<0

# 已在Part 6计算，这里做系统性汇总
print("\n5×5偏相关矩阵对宫五行检验:")
print(f"  水↔火(寒↔暑): r={partial_matrix[0,1]:.4f}, p={p_matrix[0,1]:.6f}")
print(f"  木↔金(风↔燥): r={partial_matrix[2,3]:.4f}, p={p_matrix[2,3]:.6f}")
print(f"  土(湿)自对宫: r=1.000 (定义)")

# 简单相关(不控制其他五行)
print("\n简单相关矩阵(对照):")
simple_corr = df_all[WUXING_KEYS].corr()
print(simple_corr.round(4))

print("\n水↔火简单相关:", simple_corr.loc['寒','暑'].round(4))
print("木↔金简单相关:", simple_corr.loc['风','燥'].round(4))
print("水↔木简单相关:", simple_corr.loc['寒','风'].round(4))
print("火↔土简单相关:", simple_corr.loc['暑','湿'].round(4))

# ============================================================
# Part 11: 按六气分层检验
# ============================================================
print("\n" + "="*70)
print("PART 11: 按六气分层 — 对宫五行偏相关")
print("="*70)

for qi in range(6):
    print(f"\n--- {QI_NAMES[qi]} ---")
    df_qi = df_all[df_all['qi']==qi]
    
    # 简单相关就够了(分层后样本小，偏相关不稳定)
    corr_qi = df_qi[WUXING_KEYS].corr()
    
    r_wf = corr_qi.loc['寒','暑']
    r_mj = corr_qi.loc['风','燥']
    print(f"  水↔火(寒↔暑): r={r_wf:.4f} {'✓反相关' if r_wf<0 else '✗正相关'}")
    print(f"  木↔金(风↔燥): r={r_mj:.4f} {'✓反相关' if r_mj<0 else '✗正相关'}")

print("\n" + "="*70)
print("分析完成")
print("="*70)
