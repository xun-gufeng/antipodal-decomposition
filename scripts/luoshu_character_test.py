"""
洛书特质检验：九州季节循环是否具有洛书特有对称结构

核心逻辑：
- 不去季节！季节循环本身就是信号
- 洛书特质 = 对宫轴的互补对称（1↔9最反, 2↔8最似, 3↔7/4↔6居中）
- 检验：实际9宫季节循环的相似度矩阵，是否比随机排列更符合洛书结构
- 零假设：9宫气候只由纬度+海陆决定，不存在洛书特有的轴对称

具体检验：
1. 季节循环波形相似度矩阵（9×9）
2. 洛书轴对宫 vs 非轴对宫的相似度差异
3. 洛书排列的拟合优度 vs 全部排列的拟合优度分布
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from scipy import stats
from itertools import permutations
import datetime
import warnings
warnings.filterwarnings('ignore')

QI_NAMES = ['初之气', '二之气', '三之气', '四之气', '五之气', '终之气']
MONTH_TO_QI = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5}

PALACE_WUXING = {1:'水', 2:'土', 3:'木', 4:'木', 5:'土', 6:'金', 7:'金', 8:'土', 9:'火'}
PALACE_NAMES = {1:'坎(北)', 2:'坤(西南)', 3:'震(东)', 4:'巽(东南)',
                5:'中', 6:'乾(西北)', 7:'兑(西)', 8:'艮(东北)', 9:'离(南)'}

# 洛书轴对宫定义
LUOSHU_AXIS_PAIRS = [(1,9), (2,8), (3,7), (4,6)]  # 中心5不参与轴配对

# 洛书排列：位置(row,col) → 数字
# 上=北, 下=南, 左=西, 右=东 (面南而立)
LUOSHU_GRID = np.array([[6, 1, 8],
                         [7, 5, 3],
                         [2, 9, 4]])

# 洛书对宫的数值关系：和=10（1+9, 2+8, 3+7, 4+6, 5+5）
# 物理含义：对宫互补，形成平衡

PALACE_MAP = {
    ('上','左'): 6, ('上','中'): 1, ('上','右'): 8,
    ('中','左'): 7, ('中','中'): 5, ('中','右'): 3,
    ('下','左'): 2, ('下','中'): 9, ('下','右'): 4,
}

def lat_to_row(lat):
    if lat > 35: return '上'
    elif lat < 25: return '下'
    else: return '中'

def lon_to_col(lon):
    if lon < 107.5: return '左'
    elif lon > 112.5: return '右'
    else: return '中'

DATA_DIR = './data/ncep/'

# ============================================================
# 1. 数据提取
# ============================================================
print("=" * 70)
print("洛书特质检验：九州季节循环的洛书对称结构")
print("=" * 70)

def extract_local(varname, filepath):
    ds = nc.Dataset(filepath)
    time_var = np.array(ds.variables['time'][:])
    origin = datetime.datetime(1800, 1, 1)
    dates = [origin + datetime.timedelta(hours=float(t)) for t in time_var]
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    lat_mask = (lat >= 22.5) & (lat <= 42.5)
    lon_mask = (lon >= 97.5) & (lon <= 122.5)
    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    data = ds.variables[varname][:, lat_mask, lon_mask]
    ds.close()
    records = []
    for li, la in enumerate(lat_sel):
        for lj, lo in enumerate(lon_sel):
            row = lat_to_row(la)
            col = lon_to_col(lo)
            palace = PALACE_MAP.get((row, col), None)
            if palace is None: continue
            for t, dt in enumerate(dates):
                if dt.year < 1948 or dt.year > 2025: continue
                val = float(data[t, li, lj])
                if val < -9e30 or np.isnan(val): continue
                records.append((dt.year, dt.month, palace, val))
    df = pd.DataFrame(records, columns=['year','month','palace',varname])
    return df.groupby(['year','month','palace'])[varname].mean().reset_index()

local_vars = {
    'shtfl': f'{DATA_DIR}shtfl.sfc.mon.mean.nc',
    'lhtfl': f'{DATA_DIR}lhtfl.sfc.mon.mean.nc',
    'wspd': f'{DATA_DIR}wspd.mon.mean.nc',
    'rhum': f'{DATA_DIR}rhum.mon.mean.nc',
    'air': f'{DATA_DIR}air.mon.mean.nc',
}

dfs = {}
for varname, filepath in local_vars.items():
    print(f"  {varname}...", end=" ", flush=True)
    dfs[varname] = extract_local(varname, filepath)
    print(f"OK")

df = dfs['shtfl']
for varname in ['lhtfl', 'wspd', 'rhum', 'air']:
    df = df.merge(dfs[varname], on=['year','month','palace'], how='outer')
del dfs

opendap_vars = {
    'tmax': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/tmax.2m.mon.mean.nc',
    'tmin': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/tmin.2m.mon.mean.nc',
    'tcdc': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/other_gauss/tcdc.eatm.mon.mean.nc',
    'gflux': 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/surface_gauss/gflux.sfc.mon.mean.nc',
}

for varname, url in opendap_vars.items():
    print(f"  {varname} (OPeNDAP)...", end=" ", flush=True)
    try:
        ds = xr.open_dataset(url, engine='netcdf4')
        lat = ds.lat.values; lon = ds.lon.values
        lat_mask = (lat >= 22.5) & (lat <= 42.5)
        lon_mask = (lon >= 97.5) & (lon <= 122.5)
        lat_idx = np.where(lat_mask)[0]; lon_idx = np.where(lon_mask)[0]
        data = ds[varname].isel(lat=lat_idx, lon=lon_idx).values
        time_var = ds.time.values; ds.close()
        records = []
        for li, la in enumerate(lat[lat_mask]):
            for lj, lo in enumerate(lon[lon_mask]):
                row = lat_to_row(float(la)); col = lon_to_col(float(lo))
                palace = PALACE_MAP.get((row, col), None)
                if palace is None: continue
                for t in range(len(time_var)):
                    dt = pd.Timestamp(time_var[t])
                    if dt.year < 1948 or dt.year > 2025: continue
                    val = float(data[t, li, lj])
                    if val < -9e30 or np.isnan(val): continue
                    records.append((dt.year, dt.month, palace, val))
        df_v = pd.DataFrame(records, columns=['year','month','palace',varname])
        df_v = df_v.groupby(['year','month','palace'])[varname].mean().reset_index()
        df = df.merge(df_v, on=['year','month','palace'], how='outer')
        del df_v, data
        print(f"OK")
    except Exception as e:
        print(f"FAIL: {e}")

# 派生变量
df['bowen'] = np.where(np.abs(df['lhtfl']) > 1, df['shtfl'] / df['lhtfl'], np.nan)
df['dtr'] = df['tmax'] - df['tmin']
df['qi'] = df['month'].map(MONTH_TO_QI)
print(f"数据: {len(df)} rows")

# ============================================================
# 2. 季节循环气候态（12个月均值，不去季节！）
# ============================================================
print("\n" + "=" * 70)
print("计算季节循环气候态")
print("=" * 70)

variables = ['air', 'rhum', 'wspd', 'bowen', 'dtr', 'gflux', 'shtfl', 'lhtfl', 'tcdc']
# 每宫×每变量：12个月气候态 → 归一化波形

climatology = {}  # climatology[var][palace] = array(12)

for var in variables:
    climatology[var] = {}
    for p in range(1, 10):
        sub = df[(df['palace'] == p)][['month', var]].dropna()
        if len(sub) < 36:
            continue
        monthly = sub.groupby('month')[var].mean()
        if len(monthly) < 12:
            continue
        vals = monthly.reindex(range(1, 13)).values
        # 归一化为波形：减均值除以标准差（保留形状，去掉量纲）
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if sd > 0:
            climatology[var][p] = (vals - mu) / sd
        else:
            climatology[var][p] = vals - mu

# 打印各宫季节循环特征
print("\n各宫季节循环峰值月份（未归一化原始值）：")
for var in ['air', 'rhum', 'bowen', 'dtr']:
    print(f"\n  {var}:")
    for p in range(1, 10):
        sub = df[(df['palace'] == p)][['month', var]].dropna()
        monthly = sub.groupby('month')[var].mean()
        vals = monthly.reindex(range(1, 13)).values
        peak_month = np.nanargmax(vals) + 1
        trough_month = np.nanargmin(vals) + 1
        amp = np.nanmax(vals) - np.nanmin(vals)
        print(f"    宫{p} {PALACE_NAMES[p]:>8}: 峰值月={peak_month:2d}, 谷值月={trough_month:2d}, 振幅={amp:.2f}")

# ============================================================
# 3. 季节循环波形相似度矩阵
# ============================================================
print("\n" + "=" * 70)
print("季节循环波形相似度矩阵")
print("=" * 70)

def waveform_distance(v1, v2):
    """两个12月波形的相关距离 = 1 - |correlation|"""
    valid = ~(np.isnan(v1) | np.isnan(v2))
    if valid.sum() < 6:
        return np.nan
    r = np.corrcoef(v1[valid], v2[valid])[0, 1]
    return 1 - abs(r)  # 距离：0=完全相关, 2=完全反相关

def waveform_corr(v1, v2):
    """两个12月波形的相关系数"""
    valid = ~(np.isnan(v1) | np.isnan(v2))
    if valid.sum() < 6:
        return np.nan
    return np.corrcoef(v1[valid], v2[valid])[0, 1]

# 对每个变量计算9×9相关矩阵
for var in ['air', 'rhum', 'bowen', 'dtr']:
    print(f"\n{var} 季节循环相关矩阵:")
    palaces = sorted(climatology[var].keys())
    print(f"     ", end="")
    for p in palaces:
        print(f"  宫{p}  ", end="")
    print()
    
    for p1 in palaces:
        print(f"  宫{p1}:", end="")
        for p2 in palaces:
            if p1 == p2:
                print(f"  ---  ", end="")
            else:
                r = waveform_corr(climatology[var][p1], climatology[var][p2])
                if np.isnan(r):
                    print(f"  N/A  ", end="")
                else:
                    print(f" {r:+.3f} ", end="")
        print()

# ============================================================
# 4. 洛书特质检验A：轴对宫的波形互补性
# ============================================================
print("\n" + "=" * 70)
print("检验A：洛书轴对宫的波形互补性")
print("=" * 70)
print("洛书预测：对宫波形应最互补（相关最低或反号）")
print("  1↔9(水火): 最反号  2↔8(土土): 最相似")
print("  3↔7(木金): 互补    4↔6(木金): 互补")
print()

for var in variables:
    if not climatology.get(var):
        continue
    
    palaces = sorted(climatology[var].keys())
    
    # 所有可能的配对
    all_pairs = []
    for i, p1 in enumerate(palaces):
        for p2 in palaces[i+1:]:
            if p1 == 5 or p2 == 5: continue  # 排除中心宫
            r = waveform_corr(climatology[var][p1], climatology[var][p2])
            if not np.isnan(r):
                all_pairs.append((p1, p2, r))
    
    # 轴对宫
    axis_r = {}
    for pa, pb in LUOSHU_AXIS_PAIRS:
        if pa in climatology[var] and pb in climatology[var]:
            r = waveform_corr(climatology[var][pa], climatology[var][pb])
            axis_r[(pa, pb)] = r
    
    # 非轴对宫
    axis_set = set()
    for pa, pb in LUOSHU_AXIS_PAIRS:
        axis_set.add((min(pa,pb), max(pa,pb)))
    non_axis = [(p1, p2, r) for p1, p2, r in all_pairs if (min(p1,p2), max(p1,p2)) not in axis_set]
    
    print(f"  {var}:")
    print(f"    轴对宫相关:", end="")
    for (pa, pb), r in sorted(axis_r.items()):
        print(f" {pa}↔{pb}={r:+.3f}", end="")
    
    # 洛书预测：1↔9应最不相关（最互补），2↔8应最相关（最相似）
    axis_r_vals = [r for r in axis_r.values() if not np.isnan(r)]
    non_axis_r_vals = [abs(r) for _, _, r in non_axis]
    
    if axis_r_vals and non_axis_r_vals:
        # 检验：轴对宫的|相关|是否比非轴对宫更极端（更互补或更相似）
        axis_abs = [abs(r) for r in axis_r_vals]
        print(f"\n    轴|相关|均值={np.mean(axis_abs):.3f}, 非轴|相关|均值={np.mean(non_axis_r_vals):.3f}")
        
        # 更关键的检验：1↔9是否是最不相关的配对？
        all_r_sorted = sorted(all_pairs, key=lambda x: x[2])
        if axis_r.get((1,9)) is not None:
            r19 = axis_r[(1,9)]
            rank = sum(1 for _, _, r in all_pairs if r < r19) + 1
            print(f"    1↔9相关={r19:+.3f}, 在{len(all_pairs)}对中排名{rank}(1=最负)")
        
        # 2↔8是否是最相关的配对？
        if axis_r.get((2,8)) is not None:
            r28 = axis_r[(2,8)]
            rank = sum(1 for _, _, r in all_pairs if r > r28) + 1
            print(f"    2↔8相关={r28:+.3f}, 在{len(all_pairs)}对中排名{rank}(1=最正)")
    print()

# ============================================================
# 5. 洛书特质检验B：排列检验 — 洛书排列 vs 随机排列
# ============================================================
print("\n" + "=" * 70)
print("检验B：洛书排列拟合优度 vs 随机排列")
print("=" * 70)
print("思路：把1-9的数字分配到9宫位置，洛书是一种特定分配")
print("如果气候的互补模式与洛书分配最匹配，则具有洛书特质")
print()

# 定义"互补性得分"：对宫相关系数的加权平均
# 洛书预测权重：1↔9(和=10)最大，2↔8(和=10)相同，3↔7(和=10)，4↔6(和=10)
# 但物理预期：1↔9最互补(r最负), 2↔8最相似(r最正), 3↔7/4↔6居中

# 用季节循环波形定义"互补性"：
# 互补 = 对宫季节循环反相 → 相关为负
# 相似 = 对宫季节循环同相 → 相关为正

# 洛书特质得分(Luoshu Character Score, LCS)：
# LCS = -r(1,9) + r(2,8) - |r(3,7)| - |r(4,6)|
# 水火轴越反号越好，坤艮轴越同号越好，其余轴越互补越好

# 但这太主观了。更客观的做法：
# 对所有8!/2 = 20160种排列（9个位置分配1-9，减去等价排列）
# 计算每种排列的"对宫互补度"
# 看洛书排列是否优于大部分随机排列

# 简化：固定3×3网格位置，只排列1-9的数字分配
# 对宫定义：上下对+左右对+对角对 → 4对轴

# 用综合变量（多变量平均波形距离）来定义互补度

# 先构建综合季节循环特征：对每宫，拼接所有变量的归一化波形
print("构建综合季节循环特征向量...")
feature_vectors = {}
for p in range(1, 10):
    vecs = []
    for var in variables:
        if p in climatology.get(var, {}):
            vecs.append(climatology[var][p])
    if vecs:
        feature_vectors[p] = np.concatenate(vecs)  # 12 × n_vars 维

print(f"  特征维度: {len(next(iter(feature_vectors.values())))}")

# 计算9×9综合距离矩阵
n_palace = 9
dist_matrix = np.full((n_palace, n_palace), np.nan)
corr_matrix = np.full((n_palace, n_palace), np.nan)

for p1 in range(1, 10):
    for p2 in range(1, 10):
        if p1 == p2:
            dist_matrix[p1-1, p2-1] = 0
            corr_matrix[p1-1, p2-1] = 1.0
            continue
        if p1 in feature_vectors and p2 in feature_vectors:
            v1 = feature_vectors[p1]
            v2 = feature_vectors[p2]
            valid = ~(np.isnan(v1) | np.isnan(v2))
            if valid.sum() > 10:
                corr_matrix[p1-1, p2-1] = np.corrcoef(v1[valid], v2[valid])[0, 1]

print("\n综合波形相关矩阵:")
print("      ", end="")
for p in range(1, 10):
    print(f"  宫{p}  ", end="")
print()
for p1 in range(1, 10):
    print(f"  宫{p1}:", end="")
    for p2 in range(1, 10):
        r = corr_matrix[p1-1, p2-1]
        if np.isnan(r):
            print(f"  N/A  ", end="")
        elif p1 == p2:
            print(f"  1.00 ", end="")
        else:
            print(f" {r:+.3f} ", end="")
    print()

# 定义洛书对宫互补度得分
# 洛书特质：轴对宫应该最"互补"（距离最大/相关最低）
# 得分 = sum of (轴对宫距离) = sum of (1 - 轴对宫|相关|) 对反号轴
#        + sum of (1 - 轴对宫距离) 对同号轴
# 但我们不知道哪条轴该反号哪条该同号——用洛书结构预测

# 洛书预测的轴关系强度：
# 1↔9: 数值差8(最大)→最互补
# 3↔7: 数值差4 →中等互补
# 4↔6: 数值差2 →弱互补
# 2↔8: 数值差6 但同为土→可能相似

# 更简洁的得分：用洛书对宫的数值差作为权重
# 对宫数值差：|1-9|=8, |2-8|=6, |3-7|=4, |4-6|=2
# 洛书预测：数值差越大→对宫越互补(距离越大)

def luoshu_complementarity_score(assignment, corr_mat):
    """
    给定9宫位置到1-9数字的分配，计算洛书互补度得分
    assignment: dict mapping position_idx(0-8) → palace_number(1-9)
    即位置(row,col) → 宫号
    
    3×3网格的对宫定义（通过中心对称）：
    (0,0)↔(2,2), (0,1)↔(2,1), (0,2)↔(2,0), (1,0)↔(1,2)
    
    洛书预测：对宫的洛书数之和=10
    → 数值差越大 → 应该越互补
    → 得分 = Σ(对宫数值差 × 对宫距离)
    """
    # 位置对宫关系（中心对称）
    position_pairs = [(0,8), (1,7), (2,6), (3,5)]  # (r,c) linear index
    
    score = 0
    for pos_a, pos_b in position_pairs:
        palace_a = assignment[pos_a]  # 该位置分配的宫号
        palace_b = assignment[pos_b]
        luoshu_a = palace_a  # 宫号就是洛书数
        luoshu_b = palace_b
        # 数值差
        num_diff = abs(luoshu_a - luoshu_b)
        # 实际距离（1-相关）
        r = corr_mat[palace_a-1, palace_b-1]
        if np.isnan(r):
            continue
        distance = 1 - r  # 正相关→距离小，反相关→距离大
        # 加权：数值差越大，权重越大
        score += num_diff * distance
    
    return score

# 当前洛书排列的得分
luoshu_assignment = {0:6, 1:1, 2:8, 3:7, 4:5, 5:3, 6:2, 7:9, 8:4}
obs_score = luoshu_complementarity_score(luoshu_assignment, corr_matrix)
print(f"\n洛书排列互补度得分: {obs_score:.4f}")

# Permutation: 随机分配1-9到9个位置
N_PERM = 50000
np.random.seed(42)

perm_scores = np.zeros(N_PERM)
for k in range(N_PERM):
    perm = np.random.permutation(9) + 1  # 随机1-9
    perm_assign = {i: perm[i] for i in range(9)}
    perm_scores[k] = luoshu_complementarity_score(perm_assign, corr_matrix)

perm_mean = np.mean(perm_scores)
perm_std = np.std(perm_scores)
p_val = np.mean(perm_scores >= obs_score)
p_val = max(min(p_val, 1.0), 1.0/N_PERM)

print(f"随机排列得分: μ={perm_mean:.4f}, σ={perm_std:.4f}")
print(f"洛书排列百分位: {np.mean(perm_scores <= obs_score)*100:.1f}%")
print(f"p值(单侧: ≥洛书得分): {p_val:.4f}")

sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
print(f"判定: {sig if sig else '不显著'}")

# ============================================================
# 6. 洛书特质检验C：逐轴细节
# ============================================================
print("\n" + "=" * 70)
print("检验C：逐轴互补度细节")
print("=" * 70)

# 固定位置对宫关系，看4条轴各自的互补度
position_pairs_names = [
    ((0,0),(2,2), '左上↔右下(对角)'),
    ((0,1),(2,1), '上中↔下中(南北)'),
    ((0,2),(2,0), '右上↔左下(对角)'),
    ((1,0),(1,2), '中左↔中右(东西)'),
]

print("\n洛书分配下的轴对宫:")
print(f"  {'轴':>16} {'宫对':>6} {'洛书数':>6} {'数值差':>6} {'综合r':>8} {'距离':>6}")
print("  " + "-" * 55)
for (ra,ca),(rb,cb), name in position_pairs_names:
    p_a = luoshu_assignment[ra*3+ca]
    p_b = luoshu_assignment[rb*3+cb]
    r = corr_matrix[p_a-1, p_b-1]
    num_diff = abs(p_a - p_b)
    if not np.isnan(r):
        print(f"  {name:>16} {p_a}↔{p_b}  {p_a},{p_b}  {num_diff:6d} {r:+8.4f} {1-r:6.4f}")

# 检验：在洛书排列中，数值差最大的1↔9轴是否距离也最大？
axis_data = []
for (ra,ca),(rb,cb), name in position_pairs_names:
    p_a = luoshu_assignment[ra*3+ca]
    p_b = luoshu_assignment[rb*3+cb]
    r = corr_matrix[p_a-1, p_b-1]
    num_diff = abs(p_a - p_b)
    if not np.isnan(r):
        axis_data.append((name, p_a, p_b, num_diff, r, 1-r))

if axis_data:
    # Spearman相关：数值差 vs 距离
    num_diffs = [x[3] for x in axis_data]
    distances = [x[5] for x in axis_data]
    if len(num_diffs) >= 4:
        sp_r, sp_p = stats.spearmanr(num_diffs, distances)
        print(f"\n  洛书数值差 vs 实际距离 Spearman: ρ={sp_r:+.3f}, p={sp_p:.4f}")

# ============================================================
# 7. 逐变量分别检验洛书拟合
# ============================================================
print("\n" + "=" * 70)
print("检验D：逐变量洛书拟合")
print("=" * 70)

for var in variables:
    if not climatology.get(var):
        continue
    
    # 构建该变量的相关矩阵
    var_corr = np.full((9, 9), np.nan)
    for p1 in range(1, 10):
        for p2 in range(1, 10):
            if p1 == p2:
                var_corr[p1-1, p2-1] = 1.0
                continue
            if p1 in climatology[var] and p2 in climatology[var]:
                r = waveform_corr(climatology[var][p1], climatology[var][p2])
                var_corr[p1-1, p2-1] = r
    
    # 洛书得分
    obs_s = luoshu_complementarity_score(luoshu_assignment, var_corr)
    
    # Permutation
    perm_s = np.zeros(N_PERM)
    for k in range(N_PERM):
        perm = np.random.permutation(9) + 1
        perm_assign = {i: perm[i] for i in range(9)}
        perm_s[k] = luoshu_complementarity_score(perm_assign, var_corr)
    
    p = np.mean(perm_s >= obs_s)
    p = max(min(p, 1.0), 1.0/N_PERM)
    
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    pct = np.mean(perm_s <= obs_s) * 100
    
    # 轴对宫相关
    axis_info = []
    for pa, pb in LUOSHU_AXIS_PAIRS:
        r = var_corr[pa-1, pb-1] if not np.isnan(var_corr[pa-1, pb-1]) else np.nan
        axis_info.append(f"{pa}↔{pb}={r:+.2f}" if not np.isnan(r) else f"{pa}↔{pb}=N/A")
    
    print(f"  {var:>10}: 洛书得分={obs_s:.3f}, p={p:.4f}{sig}, 百分位={pct:.1f}%  [{', '.join(axis_info)}]")

# ============================================================
# 8. 附加检验：季节循环方差的空间洛书结构
# ============================================================
print("\n" + "=" * 70)
print("检验E：季节振幅的洛书结构")
print("=" * 70)
print("洛书预测：对宫数值差越大→季节振幅差异越大")
print("  1↔9(差8): 振幅差应最大(北温带vs亚热带)")
print("  2↔8(差6): 振幅差中等")
print("  3↔7(差4): 振幅差中等")
print("  4↔6(差2): 振幅差最小")
print()

for var in ['air', 'rhum', 'bowen', 'dtr']:
    if not climatology.get(var):
        continue
    print(f"  {var}:")
    amplitudes = {}
    for p in range(1, 10):
        if p in climatology[var]:
            # 原始振幅（需要从非归一化数据计算）
            sub = df[(df['palace'] == p)][['month', var]].dropna()
            monthly = sub.groupby('month')[var].mean()
            if len(monthly) >= 12:
                amp = monthly.max() - monthly.min()
                amplitudes[p] = amp
    
    # 轴对宫振幅差
    axis_amp_diffs = []
    num_diffs = []
    for pa, pb in LUOSHU_AXIS_PAIRS:
        if pa in amplitudes and pb in amplitudes:
            amp_diff = abs(amplitudes[pa] - amplitudes[pb])
            num_diff = abs(pa - pb)
            axis_amp_diffs.append(amp_diff)
            num_diffs.append(num_diff)
            print(f"    {pa}↔{pb}: 振幅差={amp_diff:.3f} (宫{pa}={amplitudes[pa]:.3f}, 宫{pb}={amplitudes[pb]:.3f})")
    
    if len(axis_amp_diffs) >= 3:
        sp_r, sp_p = stats.spearmanr(num_diffs, axis_amp_diffs)
        print(f"    数值差 vs 振幅差 Spearman: ρ={sp_r:+.3f}, p={sp_p:.4f}")

# ============================================================
# 综合结论
# ============================================================
print("\n" + "=" * 70)
print("综合结论")
print("=" * 70)

print("""
本实验的逻辑：

1. 洛书来自天文观测（太阳视运动+地球自转轴），其空间结构编码了
   天文-气候的因果关系

2. "洛书特质"= 九州气候的季节循环空间模式是否具有洛书特有的
   轴对称互补结构

3. 检验方法：
   - 不去季节！季节循环本身是信号
   - 9宫季节波形的相关矩阵 → 洛书排列拟合优度 vs 随机排列
   - 逐变量检验
   - 轴对宫互补度排序是否匹配洛书数值差排序

4. 核心判断标准：
   - 如果洛书排列的互补度得分优于大部分随机排列(p<0.05)
     → 九州气候确实具有洛书特质
   - 如果不优于 → 九州气候的空间差异是纬度/海陆的物理结果，
     不特异于洛书排列
""")

print("实验完成")
