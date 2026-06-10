#!/usr/bin/env python3
"""
实验D: 下临宫验证（天元玉册系统）
核心逻辑: 下临宫≠灾宫, 是非重言式预测
司天→所胜之宫(受刑者)→该宫五行受制→对应气候变量异常

预测方向:
  厥阴风木司天 → 2宫(西南/土) → 土湿受制 → rhum↓
  少阴/少阳司天 → 7宫(西/金) → 金燥受制 → rhum↑
  太阴湿土司天 → 1宫(北/水) → 水寒受制 → air↑
  阳明燥金司天 → 3宫(东/木) → 木风受制 → wspd↓
  太阳寒水司天 → 9宫(南/火) → 火热受制 → air↓

中见运抑制: 运克司天 → 效应减轻
"""

import numpy as np
import xarray as xr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 参数定义
# ============================================================
lat_lo, lat_hi = 29.62, 39.62
lon_lo, lon_hi = 107.45, 117.45
ext_lat_lo, ext_lat_hi = 17.5, 42.5
ext_lon_lo, ext_lon_hi = 97.5, 122.5

PALACE_WUXING = {
    1: '水', 2: '土', 3: '木', 4: '木', 5: '土',
    6: '金', 7: '金', 8: '土', 9: '火'
}

# 下临宫→预测
XIALIN_PRED = {
    2: {'var': 'rhum', 'direction': 'neg', 'desc': '土湿受制→rhum↓'},
    7: {'var': 'rhum', 'direction': 'pos', 'desc': '金燥受制→rhum↑(湿增)'},
    1: {'var': 'air', 'direction': 'pos', 'desc': '水寒受制→air↑(偏暖)'},
    3: {'var': 'wspd', 'direction': 'neg', 'desc': '木风受制→wspd↓'},
    9: {'var': 'air', 'direction': 'neg', 'desc': '火热受制→air↓(偏凉)'},
}

# 司天→下临宫映射
SITIAN_XIALIN = {
    '厥阴风木': 2,   # 木克土→2宫
    '少阴君火': 7,   # 火克金→7宫
    '少阳相火': 7,   # 火克金→7宫
    '太阴湿土': 1,   # 土克水→1宫
    '阳明燥金': 3,   # 金克木→3宫
    '太阳寒水': 9,   # 水克火→9宫
}

# 司天五行
SITIAN_WUXING = {
    '厥阴风木': '木', '少阴君火': '火', '少阳相火': '火',
    '太阴湿土': '土', '阳明燥金': '金', '太阳寒水': '水',
}

# 五行相克
KE = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}

# qi→月份
QI_MONTHS = {
    1: [1, 2], 2: [3, 4], 3: [5, 6],
    4: [7, 8], 5: [9, 10], 6: [11, 12],
}

# 司天对应qi(三之气=5-6月, 但司天影响上半年)
# 更精确: 司天主上半年, 在泉主下半年
# 但下临宫的描述是全年性的

TIANGAN = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
DIZHI = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']

DIZHI_SITIAN = {
    '子': '少阴君火', '午': '少阴君火',
    '丑': '太阴湿土', '未': '太阴湿土',
    '寅': '少阳相火', '申': '少阳相火',
    '卯': '阳明燥金', '酉': '阳明燥金',
    '辰': '太阳寒水', '戌': '太阳寒水',
    '巳': '厥阴风木', '亥': '厥阴风木',
}

def get_year_info(year):
    """获取年份完整运气信息"""
    tigan_idx = (year - 4) % 10
    dizhi_idx = (year - 4) % 12
    tigan = TIANGAN[tigan_idx]
    dizhi = DIZHI[dizhi_idx]
    
    TIANGAN_YUN = {0:'土',1:'金',2:'水',3:'木',4:'火',5:'土',6:'金',7:'水',8:'木',9:'火'}
    yun_wuxing = TIANGAN_YUN[tigan_idx]
    is_too_much = tigan_idx % 2 == 0
    
    sitian = DIZHI_SITIAN[dizhi]
    sitian_wuxing = SITIAN_WUXING[sitian]
    xialin_palace = SITIAN_XIALIN[sitian]
    
    # 中见运是否克司天
    zhongjian_ke_sitian = (KE.get(yun_wuxing) == sitian_wuxing)
    
    return {
        'year': year, 'tigan': tigan, 'dizhi': dizhi,
        'yun_wuxing': yun_wuxing, 'is_too_much': is_too_much,
        'sitian': sitian, 'sitian_wuxing': sitian_wuxing,
        'xialin_palace': xialin_palace,
        'zhongjian_ke_sitian': zhongjian_ke_sitian,
    }

def assign_palace(lat, lon):
    if lat < lat_lo:
        row = 0
    elif lat > lat_hi:
        row = 2
    else:
        row = 1
    if lon < lon_lo:
        col = 0
    elif lon > lon_hi:
        col = 2
    else:
        col = 1
    palace_map = {
        (0,0):6,(0,1):1,(0,2):8,
        (1,0):7,(1,1):5,(1,2):3,
        (2,0):2,(2,1):9,(2,2):4,
    }
    return palace_map.get((row,col), None)

# ============================================================
# 2. 加载数据
# ============================================================
print("=" * 70)
print("实验D: 下临宫验证（天元玉册·非重言式预测）")
print("=" * 70)

DATA_DIR = './data/ncep_raw'
variables = {}
for varname in ['air', 'rhum', 'wspd']:
    print(f"加载 {varname}...")
    ds = xr.open_dataset(f'{DATA_DIR}/{varname}.mon.mean.nc')
    variables[varname] = ds[varname]

lat = variables['air'].lat.values
lon = variables['air'].lon.values

import pandas as pd
dates = pd.to_datetime(variables['air'].time.values)
years_arr = dates.year.values
months_arr = dates.month.values

valid_years = range(1948, 2026)

# 九宫格点
palace_grids = {p: [] for p in range(1, 10)}
for i, la in enumerate(lat):
    for j, lo in enumerate(lon):
        if ext_lat_lo <= la <= ext_lat_hi and ext_lon_lo <= lo <= ext_lon_hi:
            p = assign_palace(la, lo)
            if p is not None:
                palace_grids[p].append((i, j))

print("\n九宫格点分布:")
for p in range(1, 10):
    print(f"  宫{p}({PALACE_WUXING[p]}): {len(palace_grids[p])}个格点")

# ============================================================
# 3. 计算qi标准化逐年逐宫异常
# ============================================================
print("\n计算qi标准化去季节异常...")

# {varname: {palace: {year: anomaly}}}
annual_anomalies = {}
# {varname: {palace: {year: {qi: anomaly}}}}
qi_annual_anomalies = {}

for varname in ['air', 'rhum', 'wspd']:
    print(f"  处理 {varname}...")
    data = variables[varname].values
    
    annual_anomalies[varname] = {p: {} for p in range(1, 10)}
    qi_annual_anomalies[varname] = {p: {} for p in range(1, 10)}
    
    for p in range(1, 10):
        grids = palace_grids[p]
        if len(grids) == 0:
            continue
        
        palace_data = np.array([data[:, i, j] for i, j in grids]).mean(axis=0)
        
        # qi标准化
        qi_groups = {qi: [] for qi in range(1, 7)}
        qi_indices = {qi: [] for qi in range(1, 7)}
        for t in range(len(palace_data)):
            for qi, qm in QI_MONTHS.items():
                if months_arr[t] in qm:
                    qi_groups[qi].append(palace_data[t])
                    qi_indices[qi].append(t)
        
        standardized = np.full(len(palace_data), np.nan)
        for qi in range(1, 7):
            vals = np.array(qi_groups[qi])
            idxs = np.array(qi_indices[qi])
            if len(vals) > 0:
                mu = vals.mean()
                sigma = vals.std()
                if sigma > 0:
                    standardized[idxs] = (vals - mu) / sigma
        
        # 逐年年均异常
        for year in valid_years:
            mask = (years_arr == year) & (~np.isnan(standardized))
            if mask.sum() >= 6:
                annual_anomalies[varname][p][year] = standardized[mask].mean()
        
        # 逐年分qi异常
        for year in valid_years:
            qi_annual_anomalies[varname][p][year] = {}
            for qi in range(1, 7):
                qi_months = QI_MONTHS[qi]
                mask = (years_arr == year) & np.isin(months_arr, qi_months) & (~np.isnan(standardized))
                if mask.sum() >= len(qi_months):
                    qi_annual_anomalies[varname][p][year][qi] = standardized[mask].mean()

# ============================================================
# 4. 核心检验1: 下临宫年均异常方向性
# ============================================================
print("\n" + "=" * 70)
print("核心检验1: 下临宫年均异常方向性")
print("=" * 70)

# 对每个司天类型, 检验下临宫变量是否方向性偏移
sitian_types = ['厥阴风木', '少阴君火', '少阳相火', '太阴湿土', '阳明燥金', '太阳寒水']

results_xialin = []

for sitian in sitian_types:
    xialin_p = SITIAN_XIALIN[sitian]
    pred = XIALIN_PRED[xialin_p]
    varname = pred['var']
    direction = pred['direction']
    
    # 该司天所有年份
    sitian_years = []
    for year in valid_years:
        info = get_year_info(year)
        if info['sitian'] == sitian:
            sitian_years.append(year)
    
    # 下临宫异常
    anom_list = [annual_anomalies[varname][xialin_p].get(y, np.nan) for y in sitian_years]
    anom_list = [a for a in anom_list if not np.isnan(a)]
    
    if not anom_list:
        continue
    
    # 方向一致率
    if direction == 'pos':
        consistent_rate = sum(1 for a in anom_list if a > 0) / len(anom_list)
    else:
        consistent_rate = sum(1 for a in anom_list if a < 0) / len(anom_list)
    
    mean_anom = np.mean(anom_list)
    
    # t检验(均值≠0)
    t_stat, p_val = stats.ttest_1samp(anom_list, 0) if len(anom_list) > 1 else (np.nan, np.nan)
    
    # 方向匹配
    dir_match = '✓' if (direction == 'pos' and mean_anom > 0) or (direction == 'neg' and mean_anom < 0) else '✗'
    
    result = {
        'sitian': sitian, 'xialin_p': xialin_p, 'var': varname,
        'direction': direction, 'n': len(anom_list), 'mean': mean_anom,
        'consistent_rate': consistent_rate, 't': t_stat, 'p': p_val,
        'dir_match': dir_match,
    }
    results_xialin.append(result)
    
    print(f"\n{sitian} → 下临宫{xialin_p}({PALACE_WUXING[xialin_p]}) → {varname}")
    print(f"  预测: {pred['desc']}")
    print(f"  n={len(anom_list)}, 均值={mean_anom:+.4f} {dir_match}, 一致率={consistent_rate:.1%}")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}")

# 汇总方向一致
total_dir = sum(1 for r in results_xialin if r['dir_match'] == '✓')
total_n = len(results_xialin)
print(f"\n方向一致: {total_dir}/{total_n} = {total_dir/total_n:.1%}")

# ============================================================
# 5. 核心检验2: 下临宫 vs 非下临宫对照
# ============================================================
print("\n" + "=" * 70)
print("核心检验2: 下临宫异常 vs 非下临宫同变量异常")
print("=" * 70)

# 对每个年份: 下临宫变量异常 vs 同年其他宫同变量异常
n_years = 0
n_xialin_larger = 0  # 下临宫异常方向性占优

for year in valid_years:
    info = get_year_info(year)
    xialin_p = info['xialin_palace']
    pred = XIALIN_PRED[xialin_p]
    varname = pred['var']
    direction = pred['direction']
    
    xialin_anom = annual_anomalies[varname][xialin_p].get(year, np.nan)
    if np.isnan(xialin_anom):
        continue
    
    # 其他宫同变量异常
    other_anoms = []
    for p in range(1, 10):
        if p == xialin_p:
            continue
        val = annual_anomalies[varname][p].get(year, np.nan)
        if not np.isnan(val):
            other_anoms.append(val)
    
    if not other_anoms:
        continue
    
    n_years += 1
    
    # 检验: 下临宫异常是否方向性更大
    median_other = np.median(other_anoms)
    if direction == 'pos':
        if xialin_anom > median_other:
            n_xialin_larger += 1
    else:
        if xialin_anom < median_other:
            n_xialin_larger += 1

rate = n_xialin_larger / n_years
print(f"下临宫方向性占优: {n_xialin_larger}/{n_years} = {rate:.1%}")
binom_p = stats.binomtest(n_xialin_larger, n_years, 0.5).pvalue
print(f"二项检验 p={binom_p:.4f}")

# ============================================================
# 6. 核心检验3: 中见运抑制效应
# ============================================================
print("\n" + "=" * 70)
print("核心检验3: 中见运抑制效应")
print("=" * 70)

# 分组: 中见运克司天 vs 不克
inhibited_abs = []  # 克→效应减轻→|异常|应更小
not_inhibited_abs = []

inhibited_dir = []  # 克→方向一致率应更低
not_inhibited_dir = []

for year in valid_years:
    info = get_year_info(year)
    xialin_p = info['xialin_palace']
    pred = XIALIN_PRED[xialin_p]
    varname = pred['var']
    direction = pred['direction']
    
    anom = annual_anomalies[varname][xialin_p].get(year, np.nan)
    if np.isnan(anom):
        continue
    
    if info['zhongjian_ke_sitian']:
        inhibited_abs.append(abs(anom))
        if direction == 'pos':
            inhibited_dir.append(anom > 0)
        else:
            inhibited_dir.append(anom < 0)
    else:
        not_inhibited_abs.append(abs(anom))
        if direction == 'pos':
            not_inhibited_dir.append(anom > 0)
        else:
            not_inhibited_dir.append(anom < 0)

print(f"中见运克司天: n={len(inhibited_abs)}, |异常|均值={np.mean(inhibited_abs):.4f}, 方向一致率={np.mean(inhibited_dir):.1%}")
print(f"中见运不克司天: n={len(not_inhibited_abs)}, |异常|均值={np.mean(not_inhibited_abs):.4f}, 方向一致率={np.mean(not_inhibited_dir):.1%}")

if inhibited_abs and not_inhibited_abs:
    t_abs, p_abs = stats.ttest_ind(inhibited_abs, not_inhibited_abs)
    print(f"|异常|比较: t={t_abs:.3f}, p={p_abs:.4f}")

if inhibited_dir and not_inhibited_dir:
    # 方向一致率比较(Fisher exact)
    from scipy.stats import fisher_exact
    a = sum(inhibited_dir)
    b = len(inhibited_dir) - a
    c = sum(not_inhibited_dir)
    d = len(not_inhibited_dir) - c
    odds, p_fisher = fisher_exact([[a,b],[c,d]])
    print(f"方向一致率比较: Fisher p={p_fisher:.4f}")

# ============================================================
# 7. 核心检验4: 分司天类型的下临宫检验
# ============================================================
print("\n" + "=" * 70)
print("核心检验4: 逐司天类型×下临宫×目标qi")
print("=" * 70)

# 更精确: 司天主三之气(5-6月), 但影响全年
# 先看三之气(5-6月)的信号, 再看年均

for sitian in sitian_types:
    xialin_p = SITIAN_XIALIN[sitian]
    pred = XIALIN_PRED[xialin_p]
    varname = pred['var']
    direction = pred['direction']
    sitian_wuxing = SITIAN_WUXING[sitian]
    
    sitian_years = [y for y in valid_years if get_year_info(y)['sitian'] == sitian]
    
    # 年均异常
    annual_anom_list = [annual_anomalies[varname][xialin_p].get(y, np.nan) for y in sitian_years]
    annual_anom_list = [a for a in annual_anom_list if not np.isnan(a)]
    
    # 三之气异常(司天主令)
    qi3_anom_list = []
    for y in sitian_years:
        qi_dict = qi_annual_anomalies[varname][xialin_p].get(y, {})
        a = qi_dict.get(3, np.nan)
        if not np.isnan(a):
            qi3_anom_list.append(a)
    
    # 上半年异常(司天管上半年)
    upper_anom_list = []
    for y in sitian_years:
        qi_dict = qi_annual_anomalies[varname][xialin_p].get(y, {})
        vals = [qi_dict.get(qi, np.nan) for qi in [1, 2, 3]]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            upper_anom_list.append(np.mean(vals))
    
    for label, anom_list in [('年均', annual_anom_list), ('三之气', qi3_anom_list), ('上半年', upper_anom_list)]:
        if not anom_list:
            continue
        mean_a = np.mean(anom_list)
        if direction == 'pos':
            rate = sum(1 for a in anom_list if a > 0) / len(anom_list)
        else:
            rate = sum(1 for a in anom_list if a < 0) / len(anom_list)
        t_s, p_s = stats.ttest_1samp(anom_list, 0) if len(anom_list) > 1 else (np.nan, np.nan)
        dir_m = '✓' if (direction == 'pos' and mean_a > 0) or (direction == 'neg' and mean_a < 0) else '✗'
        print(f"  {sitian}→宫{xialin_p} {varname} [{label}]: 均值={mean_a:+.4f}{dir_m}, 一致率={rate:.1%}(n={len(anom_list)}), p={p_s:.4f}")

# ============================================================
# 8. 核心检验5: 7宫特别检验(占1/3, 样本最大)
# ============================================================
print("\n" + "=" * 70)
print("核心检验5: 7宫(西/金)专项 — 样本量最大(少阴+少阳=2/6司天)")
print("=" * 70)

# 7宫下临年 vs 非7宫下临年
xialin7_years = [y for y in valid_years if get_year_info(y)['xialin_palace'] == 7]
non_xialin7_years = [y for y in valid_years if get_year_info(y)['xialin_palace'] != 7]

print(f"7宫下临年: {len(xialin7_years)}年, 非7宫下临年: {len(non_xialin7_years)}年")

# 7宫rhun异常
xialin7_rhum = [annual_anomalies['rhum'][7].get(y, np.nan) for y in xialin7_years]
xialin7_rhum = [a for a in xialin7_rhum if not np.isnan(a)]
non_xialin7_rhum = [annual_anomalies['rhum'][7].get(y, np.nan) for y in non_xialin7_years]
non_xialin7_rhum = [a for a in non_xialin7_rhum if not np.isnan(a)]

print(f"\n7宫rhum异常:")
print(f"  下临7宫年: 均值={np.mean(xialin7_rhum):+.4f}, n={len(xialin7_rhum)}")
print(f"  非下临7宫年: 均值={np.mean(non_xialin7_rhum):+.4f}, n={len(non_xialin7_rhum)}")

if xialin7_rhum and non_xialin7_rhum:
    t, p = stats.ttest_ind(xialin7_rhum, non_xialin7_rhum)
    print(f"  t={t:.3f}, p={p:.4f}")
    
    # 方向性: 预测rhum↑
    pos_rate_7 = sum(1 for a in xialin7_rhum if a > 0) / len(xialin7_rhum)
    pos_rate_non = sum(1 for a in non_xialin7_rhum if a > 0) / len(non_xialin7_rhum)
    print(f"  rhum>0比例: 下临年={pos_rate_7:.1%}, 非下临年={pos_rate_non:.1%}")

# 分三之气看
xialin7_qi3 = []
for y in xialin7_years:
    qi_dict = qi_annual_anomalies['rhum'][7].get(y, {})
    a = qi_dict.get(3, np.nan)
    if not np.isnan(a):
        xialin7_qi3.append(a)

non_xialin7_qi3 = []
for y in non_xialin7_years:
    qi_dict = qi_annual_anomalies['rhum'][7].get(y, {})
    a = qi_dict.get(3, np.nan)
    if not np.isnan(a):
        non_xialin7_qi3.append(a)

if xialin7_qi3 and non_xialin7_qi3:
    t, p = stats.ttest_ind(xialin7_qi3, non_xialin7_qi3)
    print(f"\n7宫rhum三之气异常:")
    print(f"  下临7宫年: 均值={np.mean(xialin7_qi3):+.4f}, n={len(xialin7_qi3)}")
    print(f"  非下临7宫年: 均值={np.mean(non_xialin7_qi3):+.4f}, n={len(non_xialin7_qi3)}")
    print(f"  t={t:.3f}, p={p:.4f}")

# ============================================================
# 9. 核心检验6: 2宫(西南/土)专项 — 灾宫系统永久排除
# ============================================================
print("\n" + "=" * 70)
print("核心检验6: 2宫(西南/土)专项 — 灾宫系统永久排除此宫")
print("=" * 70)

xialin2_years = [y for y in valid_years if get_year_info(y)['xialin_palace'] == 2]
non_xialin2_years = [y for y in valid_years if get_year_info(y)['xialin_palace'] != 2]

print(f"2宫下临年(厥阴司天=巳亥年): {len(xialin2_years)}年")

# 2宫rhum异常(预测:rhum↓)
xialin2_rhum = [annual_anomalies['rhum'][2].get(y, np.nan) for y in xialin2_years]
xialin2_rhum = [a for a in xialin2_rhum if not np.isnan(a)]
non_xialin2_rhum = [annual_anomalies['rhum'][2].get(y, np.nan) for y in non_xialin2_years]
non_xialin2_rhum = [a for a in non_xialin2_rhum if not np.isnan(a)]

print(f"\n2宫rhum异常:")
print(f"  下临2宫年: 均值={np.mean(xialin2_rhum):+.4f}, n={len(xialin2_rhum)}")
print(f"  非下临2宫年: 均值={np.mean(non_xialin2_rhum):+.4f}, n={len(non_xialin2_rhum)}")

if xialin2_rhum and non_xialin2_rhum:
    t, p = stats.ttest_ind(xialin2_rhum, non_xialin2_rhum)
    print(f"  t={t:.3f}, p={p:.4f}")
    neg_rate_2 = sum(1 for a in xialin2_rhum if a < 0) / len(xialin2_rhum)
    print(f"  rhum<0比例: 下临年={neg_rate_2:.1%}")

# ============================================================
# 10. 综合方向一致率
# ============================================================
print("\n" + "=" * 70)
print("综合: 全部年份下临宫方向一致率")
print("=" * 70)

all_dir_tests = []
all_dir_tests_qi3 = []

for year in valid_years:
    info = get_year_info(year)
    xialin_p = info['xialin_palace']
    pred = XIALIN_PRED[xialin_p]
    varname = pred['var']
    direction = pred['direction']
    
    # 年均
    anom = annual_anomalies[varname][xialin_p].get(year, np.nan)
    if not np.isnan(anom):
        if direction == 'pos':
            all_dir_tests.append(anom > 0)
        else:
            all_dir_tests.append(anom < 0)
    
    # 三之气
    qi_dict = qi_annual_anomalies[varname][xialin_p].get(year, {})
    qi3_anom = qi_dict.get(3, np.nan)
    if not np.isnan(qi3_anom):
        if direction == 'pos':
            all_dir_tests_qi3.append(qi3_anom > 0)
        else:
            all_dir_tests_qi3.append(qi3_anom < 0)

rate_annual = sum(all_dir_tests) / len(all_dir_tests) if all_dir_tests else np.nan
rate_qi3 = sum(all_dir_tests_qi3) / len(all_dir_tests_qi3) if all_dir_tests_qi3 else np.nan

print(f"年均方向一致率: {sum(all_dir_tests)}/{len(all_dir_tests)} = {rate_annual:.1%}")
if all_dir_tests:
    bp = stats.binomtest(sum(all_dir_tests), len(all_dir_tests), 0.5).pvalue
    print(f"二项检验 p={bp:.4f}")

print(f"三之气方向一致率: {sum(all_dir_tests_qi3)}/{len(all_dir_tests_qi3)} = {rate_qi3:.1%}")
if all_dir_tests_qi3:
    bp = stats.binomtest(sum(all_dir_tests_qi3), len(all_dir_tests_qi3), 0.5).pvalue
    print(f"二项检验 p={bp:.4f}")

# ============================================================
# 11. 下临宫 vs 灾宫 比较(仅不及年)
# ============================================================
print("\n" + "=" * 70)
print("比较: 下临宫 vs 灾宫(仅不及年)")
print("=" * 70)

xialin_dir_tests_buji = []
zaigong_dir_tests_buji = []

WUXING_PALACE = {'木':3,'火':9,'土':5,'金':7,'水':1}
DISASTER_VAR = {
    1: {'var': 'air', 'direction': 'neg'},
    3: {'var': 'wspd', 'direction': 'pos'},
    5: {'var': 'rhum', 'direction': 'pos'},
    7: {'var': 'rhum', 'direction': 'neg'},
    9: {'var': 'air', 'direction': 'pos'},
}

for year in valid_years:
    info = get_year_info(year)
    if info['is_too_much']:
        continue  # 仅不及年
    
    # 下临宫
    xialin_p = info['xialin_palace']
    xialin_pred = XIALIN_PRED[xialin_p]
    anom = annual_anomalies[xialin_pred['var']][xialin_p].get(year, np.nan)
    if not np.isnan(anom):
        if xialin_pred['direction'] == 'pos':
            xialin_dir_tests_buji.append(anom > 0)
        else:
            xialin_dir_tests_buji.append(anom < 0)
    
    # 灾宫
    zaigong_p = WUXING_PALACE[info['yun_wuxing']]
    zaigong_pred = DISASTER_VAR[zaigong_p]
    anom = annual_anomalies[zaigong_pred['var']][zaigong_p].get(year, np.nan)
    if not np.isnan(anom):
        if zaigong_pred['direction'] == 'pos':
            zaigong_dir_tests_buji.append(anom > 0)
        else:
            zaigong_dir_tests_buji.append(anom < 0)

print(f"不及年下临宫方向一致率: {sum(xialin_dir_tests_buji)}/{len(xialin_dir_tests_buji)} = {np.mean(xialin_dir_tests_buji):.1%}")
print(f"不及年灾宫方向一致率: {sum(zaigong_dir_tests_buji)}/{len(zaigong_dir_tests_buji)} = {np.mean(zaigong_dir_tests_buji):.1%}")

# ============================================================
# 12. 检验5宫(中)永远非下临宫的推论
# ============================================================
print("\n" + "=" * 70)
print("推论检验: 5宫(中)永远非下临宫")
print("=" * 70)

# 如果下临宫系统有物理意义, 5宫(中)应该相对不受灾
# 检验: 5宫的异常绝对值是否系统性小于其他宫
palace5_abs = [abs(annual_anomalies['air'][5].get(y, np.nan)) for y in valid_years]
palace5_abs = [a for a in palace5_abs if not np.isnan(a)]

other_abs = []
for p in [1,2,3,7,9]:  # 下临宫覆盖的5个宫
    vals = [abs(annual_anomalies['air'][p].get(y, np.nan)) for y in valid_years]
    other_abs.extend([a for a in vals if not np.isnan(a)])

print(f"5宫(中) air |异常|均值: {np.mean(palace5_abs):.4f} (n={len(palace5_abs)})")
print(f"下临宫覆盖宫 air |异常|均值: {np.mean(other_abs):.4f} (n={len(other_abs)})")

print("\n" + "=" * 70)
print("实验D完成")
print("=" * 70)
