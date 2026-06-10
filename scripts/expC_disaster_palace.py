#!/usr/bin/env python3
"""
实验C: 洛书灾宫验证
核心逻辑: 洛书编码空间结构(平均态)→不需验证;
         灾宫运算预测特定年份特定宫位的异常→可证伪检验

预测: 中运太过→本宫变量正异常; 中运不及→本宫变量负异常
方法: qi标准化去季节→逐年逐宫计算年际异常→按灾宫分组检验
"""

import numpy as np
import xarray as xr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 洛阳中心九宫格点定义
# ============================================================
# 洛阳: 34.62°N, 112.45°E
# Δ=5°, 九宫范围 lat[29.62,39.62], lon[107.45,117.45]
# NCEP 2.5°网格扩展范围: lat[17.5,42.5], lon[97.5,122.5]

lat_lo, lat_hi = 29.62, 39.62
lon_lo, lon_hi = 107.45, 117.45
ext_lat_lo, ext_lat_hi = 17.5, 42.5
ext_lon_lo, ext_lon_hi = 97.5, 122.5

# 九宫方位与五行映射
# 洛书: 4(东南) 9(南) 2(西南) / 3(东) 5(中) 7(西) / 8(东北) 1(北) 6(西北)
PALACE_WUXING = {
    1: '水', 2: '土', 3: '木', 4: '木', 5: '土',
    6: '金', 7: '金', 8: '土', 9: '火'
}

# 灾宫→预测变量及方向
# 太过→本宫变量正异常(偏盛); 不及→本宫变量负异常(偏衰)
DISASTER_VAR = {
    1: {'var': 'air', 'direction': 'neg', 'desc': '水→寒→air负异常(偏冷)'},
    3: {'var': 'wspd', 'direction': 'pos', 'desc': '木→风→wspd正异常(偏风大)'},
    5: {'var': 'rhum', 'direction': 'pos', 'desc': '土→湿→rhum正异常(偏湿)'},
    7: {'var': 'rhum', 'direction': 'neg', 'desc': '金→燥→rhum负异常(偏燥)'},
    9: {'var': 'air', 'direction': 'pos', 'desc': '火→热→air正异常(偏热)'},
}

# ============================================================
# 2. 天干→中运→灾宫 映射
# ============================================================
# 甲己化土, 乙庚化金, 丙辛化水, 丁壬化木, 戊癸化火
# 阳干=太过, 阴干=不及
TIANGAN = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
TIANGAN_INDEX = {t: i for i, t in enumerate(TIANGAN)}

def get_yun_from_year(year):
    """从年份获取中运信息"""
    tigan_idx = (year - 4) % 10  # 甲=0: 4AD为甲子年
    tigan = TIANGAN[tigan_idx]
    
    # 中运五行
    TIANGAN_YUN = {
        0: '土', 1: '金', 2: '水', 3: '木', 4: '火',
        5: '土', 6: '金', 7: '水', 8: '木', 9: '火'
    }
    
    # 太过/不及: 阳干(奇数index)=太过, 阴干(偶数index)=不及
    # 甲(index0)=阳=太过, 乙(index1)=阴=不及...
    # Wait, convention: 甲丙戊庚壬为阳干, 乙丁己辛癸为阴干
    # index 0,2,4,6,8 → 阳 → 太过
    # index 1,3,5,7,9 → 阴 → 不及
    is_too_much = tigan_idx % 2 == 0  # 阳干=太过
    
    wuxing = TIANGAN_YUN[tigan_idx]
    
    # 灾宫
    WUXING_PALACE = {'木': 3, '火': 9, '土': 5, '金': 7, '水': 1}
    disaster_palace = WUXING_PALACE[wuxing]
    
    return {
        'tigan': tigan,
        'wuxing': wuxing,
        'is_too_much': is_too_much,
        'disaster_palace': disaster_palace,
        'taishao': '太过' if is_too_much else '不及'
    }

# ============================================================
# 3. 九宫格点分配
# ============================================================
def assign_palace_ncep2p5(lat, lon):
    """NCEP 2.5°网格九宫分配"""
    # 宫格边界
    row_bounds = [ext_lat_lo, lat_lo, lat_hi, ext_lat_hi]
    col_bounds = [ext_lon_lo, lon_lo, lon_hi, ext_lon_hi]
    
    palace_map = {
        (0,0): 6, (0,1): 1, (0,2): 8,  # 下: 东北/北/西北
        (1,0): 7, (1,1): 5, (1,2): 3,  # 中: 东/中/西
        (2,0): 2, (2,1): 9, (2,2): 4,  # 上: 东南/南/西南
    }
    
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
    
    return palace_map.get((row, col), None)

# ============================================================
# 4. 加载数据
# ============================================================
print("=" * 60)
print("实验C: 洛书灾宫验证")
print("=" * 60)

DATA_DIR = './data/ncep_raw'

# 加载变量
variables = {}
for varname in ['air', 'rhum', 'wspd']:
    print(f"加载 {varname}...")
    ds = xr.open_dataset(f'{DATA_DIR}/{varname}.mon.mean.nc')
    variables[varname] = ds[varname]

# 获取经纬度
lat = variables['air'].lat.values
lon = variables['air'].lon.values

# 时间处理
time = variables['air'].time.values
# NCEP时间: hours since 1800-01-01
import pandas as pd
dates = pd.to_datetime(time)
years = dates.year.values
months = dates.month.values

# 限定年份范围(1948-2025, 完整年)
valid_years = range(1948, 2026)

# ============================================================
# 5. 九宫格点分类
# ============================================================
palace_grids = {p: [] for p in range(1, 10)}

for i, la in enumerate(lat):
    for j, lo in enumerate(lon):
        if ext_lat_lo <= la <= ext_lat_hi and ext_lon_lo <= lo <= ext_lon_hi:
            p = assign_palace_ncep2p5(la, lo)
            if p is not None:
                palace_grids[p].append((i, j))

print("\n九宫格点分布:")
for p in range(1, 10):
    print(f"  宫{p}({PALACE_WUXING[p]}): {len(palace_grids[p])}个格点")

# ============================================================
# 6. qi标准化去季节 + 逐年逐宫异常计算
# ============================================================
print("\n计算qi标准化去季节异常...")

# 六气时段→月份映射
QI_MONTHS = {
    1: [1, 2],      # 初之气(大寒→春分): 约1-2月
    2: [3, 4],      # 二之气(春分→小满): 约3-4月
    3: [5, 6],      # 三之气(小满→大暑): 约5-6月
    4: [7, 8],      # 四之气(大暑→秋分): 约7-8月
    5: [9, 10],     # 五之气(秋分→小雪): 约9-10月
    6: [11, 12],    # 终之气(小雪→大寒): 约11-12月
}

def qi_month(month):
    """月份→qi编号"""
    for qi, months in QI_MONTHS.items():
        if month in months:
            return qi
    return None

# 对每个变量、每个宫位: qi标准化→年均异常
annual_anomalies = {}  # {varname: {palace: {year: anomaly}}}

for varname in ['air', 'rhum', 'wspd']:
    print(f"  处理 {varname}...")
    data = variables[varname].values
    # data shape: (time, lat, lon)
    
    annual_anomalies[varname] = {p: {} for p in range(1, 10)}
    
    for p in range(1, 10):
        grids = palace_grids[p]
        if len(grids) == 0:
            continue
        
        # 宫内空间均值
        palace_data = np.array([data[:, i, j] for i, j in grids]).mean(axis=0)
        # shape: (time,)
        
        # qi标准化: 在每个qi组内去均值
        qi_groups = {qi: [] for qi in range(1, 7)}
        qi_indices = {qi: [] for qi in range(1, 7)}
        
        for t in range(len(palace_data)):
            q = qi_month(months[t])
            if q is not None:
                qi_groups[q].append(palace_data[t])
                qi_indices[q].append(t)
        
        # 计算qi标准化异常
        standardized = np.full(len(palace_data), np.nan)
        for qi in range(1, 7):
            vals = np.array(qi_groups[qi])
            idxs = np.array(qi_indices[qi])
            if len(vals) > 0:
                mu = vals.mean()
                sigma = vals.std()
                if sigma > 0:
                    standardized[idxs] = (vals - mu) / sigma
        
        # 逐年取年均异常
        for year in valid_years:
            mask = (years == year) & (~np.isnan(standardized))
            if mask.sum() >= 6:  # 至少6个月有效
                annual_anomalies[varname][p][year] = standardized[mask].mean()

# ============================================================
# 7. 灾宫验证: 核心检验
# ============================================================
print("\n" + "=" * 60)
print("核心检验: 灾宫预测异常")
print("=" * 60)

# 对每个五行灾宫, 分组: 太过年 vs 不及年 vs 非灾宫年
# 检验: 灾宫变量异常是否方向性偏移

results = []

for disaster_wuxing, disaster_p in [(w, p) for w, p in [('木',3),('火',9),('土',5),('金',7),('水',1)]]:
    var_info = DISASTER_VAR[disaster_p]
    varname = var_info['var']
    direction = var_info['direction']
    
    # 获取灾宫年均异常序列
    palace_anom = annual_anomalies[varname][disaster_p]
    
    # 太过/不及年分组
    too_much_years = []
    not_enough_years = []
    other_palace_anom = {p: [] for p in range(1, 10) if p != disaster_p}
    
    for year in valid_years:
        yun_info = get_yun_from_year(year)
        if yun_info['wuxing'] == disaster_wuxing:
            if yun_info['is_too_much']:
                too_much_years.append(year)
            else:
                not_enough_years.append(year)
    
    # 灾宫异常
    too_much_anom = [palace_anom[y] for y in too_much_years if y in palace_anom and not np.isnan(palace_anom[y])]
    not_enough_anom = [palace_anom[y] for y in not_enough_years if y in palace_anom and not np.isnan(palace_anom[y])]
    
    # 非灾宫(同变量)对照
    other_anom_list = []
    for p in range(1, 10):
        if p == disaster_p:
            continue
        if DISASTER_VAR.get(p, {}).get('var') == varname:
            # 同变量但不同灾宫的宫位作为对照
            other_p_anom = annual_anomalies[varname][p]
            for year in valid_years:
                yun_info = get_yun_from_year(year)
                if yun_info['wuxing'] != disaster_wuxing and year in other_p_anom:
                    val = other_p_anom[year]
                    if not np.isnan(val):
                        other_anom_list.append(val)
    
    # 所有年的该宫异常(作为全局对照)
    all_anom = [palace_anom[y] for y in valid_years if y in palace_anom and not np.isnan(palace_anom[y])]
    
    # 检验1: 太过/不及年灾宫异常是否方向性偏移
    if direction == 'pos':
        # 太过→正异常, 不及→负异常
        # 预测: 太过年均值 > 0, 不及年均值 < 0
        pred_too = '正(偏盛)'
        pred_not = '负(偏衰)'
    else:
        # 太过→负异常, 不及→正异常
        pred_too = '负(偏盛)'
        pred_not = '正(偏衰)'
    
    too_much_mean = np.mean(too_much_anom) if too_much_anom else np.nan
    not_enough_mean = np.mean(not_enough_anom) if not_enough_anom else np.nan
    all_mean = np.mean(all_anom) if all_anom else np.nan
    
    # t检验: 太过年 vs 全体
    t_stat_too, p_val_too = stats.ttest_1samp(too_much_anom, 0) if len(too_much_anom) > 1 else (np.nan, np.nan)
    t_stat_not, p_val_not = stats.ttest_1samp(not_enough_anom, 0) if len(not_enough_anom) > 1 else (np.nan, np.nan)
    
    # 方向一致性: 太过年异常方向是否符合预测
    if direction == 'pos':
        too_consistent = sum(1 for a in too_much_anom if a > 0) / len(too_much_anom) if too_much_anom else np.nan
        not_consistent = sum(1 for a in not_enough_anom if a < 0) / len(not_enough_anom) if not_enough_anom else np.nan
    else:
        too_consistent = sum(1 for a in too_much_anom if a < 0) / len(too_much_anom) if too_much_anom else np.nan
        not_consistent = sum(1 for a in not_enough_anom if a > 0) / len(not_enough_anom) if not_enough_anom else np.nan
    
    result = {
        'disaster_wuxing': disaster_wuxing,
        'disaster_palace': disaster_p,
        'var': varname,
        'direction': direction,
        'pred_too': pred_too,
        'pred_not': pred_not,
        'too_much_n': len(too_much_anom),
        'not_enough_n': len(not_enough_anom),
        'too_much_mean': too_much_mean,
        'not_enough_mean': not_enough_mean,
        'all_mean': all_mean,
        'too_consistent': too_consistent,
        'not_consistent': not_consistent,
        't_stat_too': t_stat_too,
        'p_val_too': p_val_too,
        't_stat_not': t_stat_not,
        'p_val_not': p_val_not,
    }
    results.append(result)
    
    print(f"\n--- 灾宫{disaster_p}({disaster_wuxing}) → {varname} ---")
    print(f"  预测: 太过→{pred_too}, 不及→{pred_not}")
    print(f"  太过年(n={len(too_much_anom)}): 均值={too_much_mean:.4f}, 方向一致率={too_consistent:.1%}")
    print(f"  不及年(n={len(not_enough_anom)}): 均值={not_enough_mean:.4f}, 方向一致率={not_consistent:.1%}")
    print(f"  全年均值: {all_mean:.4f}")
    print(f"  t检验(太过≠0): t={t_stat_too:.3f}, p={p_val_too:.4f}")
    print(f"  t检验(不及≠0): t={t_stat_not:.3f}, p={p_val_not:.4f}")

# ============================================================
# 8. 综合检验: 灾宫 vs 非灾宫
# ============================================================
print("\n" + "=" * 60)
print("综合检验: 灾宫异常 vs 非灾宫异常")
print("=" * 60)

# 对每个年份, 灾宫变量异常 是否 > 同年其他宫的同变量异常
n_years = 0
n_disaster_larger = 0  # 灾宫异常绝对值 > 非灾宫中位数绝对值

for year in valid_years:
    yun_info = get_yun_from_year(year)
    disaster_p = yun_info['disaster_palace']
    var_info = DISASTER_VAR[disaster_p]
    varname = var_info['var']
    direction = var_info['direction']
    
    disaster_anom = annual_anomalies[varname][disaster_p].get(year, np.nan)
    if np.isnan(disaster_anom):
        continue
    
    # 非灾宫同变量异常
    other_anoms = []
    for p in range(1, 10):
        if p == disaster_p:
            continue
        val = annual_anomalies[varname][p].get(year, np.nan)
        if not np.isnan(val):
            other_anoms.append(val)
    
    if len(other_anoms) == 0:
        continue
    
    n_years += 1
    
    # 检验: 灾宫异常是否方向性更大
    if direction == 'pos':
        if yun_info['is_too_much']:
            # 太过→正异常→灾宫异常应 > 非灾宫
            if disaster_anom > np.median(other_anoms):
                n_disaster_larger += 1
        else:
            # 不及→负异常→灾宫异常应 < 非灾宫
            if disaster_anom < np.median(other_anoms):
                n_disaster_larger += 1
    else:
        if yun_info['is_too_much']:
            # 太过→负异常→灾宫异常应 < 非灾宫
            if disaster_anom < np.median(other_anoms):
                n_disaster_larger += 1
        else:
            # 不及→正异常→灾宫异常应 > 非灾宫
            if disaster_anom > np.median(other_anoms):
                n_disaster_larger += 1

rate = n_disaster_larger / n_years if n_years > 0 else np.nan
print(f"灾宫异常方向性占优: {n_disaster_larger}/{n_years} = {rate:.1%}")
print(f"随机期望: 50%")

# 二项检验
binom_p = stats.binom_test(n_disaster_larger, n_years, 0.5) if hasattr(stats, 'binom_test') else \
          stats.binomtest(n_disaster_larger, n_years, 0.5).pvalue
print(f"二项检验 p={binom_p:.4f}")

# ============================================================
# 9. 逐五行灾宫方向一致性汇总
# ============================================================
print("\n" + "=" * 60)
print("汇总: 逐五行灾宫方向一致性")
print("=" * 60)

for r in results:
    p = r['disaster_palace']
    w = r['disaster_wuxing']
    # 综合方向一致率: 太过+不及都方向一致
    too_rate = r['too_consistent']
    not_rate = r['not_consistent']
    combined = (too_rate + not_rate) / 2 if not (np.isnan(too_rate) or np.isnan(not_rate)) else np.nan
    sig_too = '✓' if r['p_val_too'] < 0.05 else ''
    sig_not = '✓' if r['p_val_not'] < 0.05 else ''
    print(f"宫{p}({w}): 太过一致率={too_rate:.1%}{sig_too}, 不及一致率={not_rate:.1%}{sig_not}, 综合={combined:.1%}")

# ============================================================
# 10. 下临宫检验(客气加临顺逆→宫位异常强度)
# ============================================================
print("\n" + "=" * 60)
print("下临宫检验: 顺逆→宫位异常强度")
print("=" * 60)

# 地支→司天映射
DIZHI = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
DIZHI_SITIAN = {
    '子': '少阴君火', '午': '少阴君火',
    '丑': '太阴湿土', '未': '太阴湿土',
    '寅': '少阳相火', '申': '少阳相火',
    '卯': '阳明燥金', '酉': '阳明燥金',
    '辰': '太阳寒水', '戌': '太阳寒水',
    '巳': '厥阴风木', '亥': '厥阴风木',
}

# 三之气(司天)对应宫位
SITIAN_PALACE = {
    '厥阴风木': 3,   # 木→宫3
    '少阴君火': 9,   # 火→宫9
    '少阳相火': 7,   # 相火→宫7(洛书7=金,但相火=辅位)
    '太阴湿土': 5,   # 土→宫5
    '阳明燥金': 7,   # 金→宫7
    '太阳寒水': 1,   # 水→宫1
}

# 司天在宫位的顺逆判定(简化版: 看客气五行与主宫五行关系)
# 三之气主气=少阳相火(火), 客气=司天
# 顺逆判定
def shunni(keqi_wuxing, zhuqi_wuxing):
    """客气五行 vs 主气五行 → 顺逆"""
    SHENG = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
    KE = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}
    
    if keqi_wuxing == zhuqi_wuxing:
        return 0  # 同气
    elif SHENG[keqi_wuxing] == zhuqi_wuxing:
        return 1  # 客生主→顺
    elif SHENG[zhuqi_wuxing] == keqi_wuxing:
        return -0.5  # 主生客→泄
    elif KE[keqi_wuxing] == zhuqi_wuxing:
        return -1  # 客克主→逆
    elif KE[zhuqi_wuxing] == keqi_wuxing:
        return 0.5  # 主克客→微逆
    return 0

# 对每年: 司天→三之气客气→三之气主气(少阳相火)的顺逆
# 然后检验: 逆年(顺逆<0)宫9/宫1等关键宫位异常是否更大

# 简化检验: 司天对冲年(大逆) vs 司天同气年(大顺)
# 大逆: 少阴君火司天→太阳寒水在泉 → 火水对冲
#       → 宫9(火)和宫1(水)冲突最剧烈
# 大顺: 太阴湿土司天→太阴湿土四之气同气

# 按司天类型分组, 看各宫异常强度
sitian_groups = {}
for year in valid_years:
    dizhi_idx = (year - 4) % 12
    dizhi = DIZHI[dizhi_idx]
    sitian = DIZHI_SITIAN[dizhi]
    if sitian not in sitian_groups:
        sitian_groups[sitian] = []
    sitian_groups[sitian].append(year)

print("司天分组年份统计:")
for sitian, yrs in sitian_groups.items():
    print(f"  {sitian}: {len(yrs)}年")

# 对每个司天组, 看对应宫位的异常
# 司天在三之气(5-6月)主令→应看5-6月异常
# 但我们目前只有年均异常, 先用年均

print("\n各司天组→对应宫位异常均值:")
for sitian, yrs in sitian_groups.items():
    palace = SITIAN_PALACE[sitian]
    wuxing = PALACE_WUXING[palace]
    var_info = DISASTER_VAR.get(palace, None)
    
    if var_info is None:
        print(f"  {sitian}→宫{palace}({wuxing}): 无预测变量")
        continue
    
    varname = var_info['var']
    anom_vals = [annual_anomalies[varname][palace].get(y, np.nan) for y in yrs]
    anom_vals = [v for v in anom_vals if not np.isnan(v)]
    
    if anom_vals:
        print(f"  {sitian}→宫{palace}({wuxing}) {varname}: 均值={np.mean(anom_vals):.4f}, n={len(anom_vals)}")
    else:
        print(f"  {sitian}→宫{palace}({wuxing}) {varname}: 无有效数据")

# ============================================================
# 11. 更强检验: 灾宫+司天叠加效应
# ============================================================
print("\n" + "=" * 60)
print("叠加检验: 灾宫×司天×太少 三重交互")
print("=" * 60)

# 最强预测: 天符年(中运与司天同气)→力量叠加→灾宫异常应最大
# 天符: 中运五行 = 司天五行

tianfu_years = []
for year in valid_years:
    yun_info = get_yun_from_year(year)
    dizhi_idx = (year - 4) % 12
    dizhi = DIZHI[dizhi_idx]
    sitian = DIZHI_SITIAN[dizhi]
    
    # 司天五行
    SITIAN_WUXING = {
        '厥阴风木': '木', '少阴君火': '火', '少阳相火': '火',
        '太阴湿土': '土', '阳明燥金': '金', '太阳寒水': '水'
    }
    sitian_wuxing = SITIAN_WUXING[sitian]
    
    if yun_info['wuxing'] == sitian_wuxing:
        tianfu_years.append({
            'year': year,
            'yun': yun_info['wuxing'],
            'taishao': yun_info['taishao'],
            'disaster_palace': yun_info['disaster_palace']
        })

print(f"天符年: {len(tianfu_years)}年")
for tf in tianfu_years:
    p = tf['disaster_palace']
    var_info = DISASTER_VAR[p]
    varname = var_info['var']
    anom = annual_anomalies[varname][p].get(tf['year'], np.nan)
    direction = var_info['direction']
    if not np.isnan(anom):
        expected_sign = 'pos' if tf['taishao'] == '太过' else 'neg'
        if direction == 'neg':
            expected_sign = 'neg' if tf['taishao'] == '太过' else 'pos'
        actual_sign = 'pos' if anom > 0 else 'neg'
        match = '✓' if expected_sign == actual_sign else '✗'
        print(f"  {tf['year']}: {tf['yun']}{tf['taishao']}→宫{p} {varname}={anom:.4f} {match}")

# 天符年vs非天符年: 灾宫异常绝对值比较
tianfu_disaster_anom = []
non_tianfu_disaster_anom = []

for year in valid_years:
    yun_info = get_yun_from_year(year)
    p = yun_info['disaster_palace']
    var_info = DISASTER_VAR[p]
    varname = var_info['var']
    anom = annual_anomalies[varname][p].get(year, np.nan)
    if np.isnan(anom):
        continue
    
    is_tianfu = any(tf['year'] == year for tf in tianfu_years)
    if is_tianfu:
        tianfu_disaster_anom.append(abs(anom))
    else:
        non_tianfu_disaster_anom.append(abs(anom))

if tianfu_disaster_anom and non_tianfu_disaster_anom:
    t_stat, p_val = stats.ttest_ind(tianfu_disaster_anom, non_tianfu_disaster_anom)
    print(f"\n天符年灾宫|异常|均值: {np.mean(tianfu_disaster_anom):.4f} (n={len(tianfu_disaster_anom)})")
    print(f"非天符年灾宫|异常|均值: {np.mean(non_tianfu_disaster_anom):.4f} (n={len(non_tianfu_disaster_anom)})")
    print(f"t检验: t={t_stat:.3f}, p={p_val:.4f}")

# ============================================================
# 12. 最关键: 逐五行灾宫分太多/不及的方向性t检验
# ============================================================
print("\n" + "=" * 60)
print("关键检验: 灾宫太多/不及年的方向性偏移")
print("=" * 60)

# 汇总表
print(f"{'灾宫':>4} {'五行':>4} {'变量':>6} {'太多n':>6} {'太多均值':>8} {'太多p':>8} {'不及n':>6} {'不及均值':>8} {'不及p':>8} {'方向一致':>8}")
print("-" * 80)

total_tests = 0
total_consistent = 0

for r in results:
    p = r['disaster_palace']
    direction = r['direction']
    
    # 判断方向是否一致
    too_consistent_dir = (r['too_much_mean'] > 0) if direction == 'pos' else (r['too_much_mean'] < 0)
    not_consistent_dir = (r['not_enough_mean'] < 0) if direction == 'pos' else (r['not_enough_mean'] > 0)
    
    both_consistent = too_consistent_dir and not_consistent_dir
    
    total_tests += 2  # 太过+不及
    if too_consistent_dir:
        total_consistent += 1
    if not_consistent_dir:
        total_consistent += 1
    
    mark = '✓✓' if both_consistent else ('✓' if (too_consistent_dir or not_consistent_dir) else '✗✗')
    
    print(f"宫{p:>2} {r['disaster_wuxing']:>4} {r['var']:>6} "
          f"{r['too_much_n']:>6} {r['too_much_mean']:>8.4f} {r['p_val_too']:>8.4f} "
          f"{r['not_enough_n']:>6} {r['not_enough_mean']:>8.4f} {r['p_val_not']:>8.4f} "
          f"{mark:>8}")

print(f"\n方向一致率: {total_consistent}/{total_tests} = {total_consistent/total_tests:.1%}")
print(f"随机期望: 50%")

# 二项检验
binom_p2 = stats.binomtest(total_consistent, total_tests, 0.5).pvalue
print(f"二项检验 p={binom_p2:.4f}")

# ============================================================
# 13. 补充: 分季节(司天/在泉)的灾宫检验
# ============================================================
print("\n" + "=" * 60)
print("补充: 分司天/在泉半年的灾宫检验")
print("=" * 60)

# 司天管上半年(初之气到三之气: 1-6月)
# 在泉管下半年(四之气到终之气: 7-12月)

# 重新计算: 上半年/下半年 分别的qi标准化异常
half_annual_anomalies = {}  # {varname: {palace: {year: {'upper': anom, 'lower': anom}}}}

for varname in ['air', 'rhum', 'wspd']:
    print(f"  处理 {varname}...")
    data = variables[varname].values
    
    half_annual_anomalies[varname] = {p: {} for p in range(1, 10)}
    
    for p in range(1, 10):
        grids = palace_grids[p]
        if len(grids) == 0:
            continue
        
        palace_data = np.array([data[:, i, j] for i, j in grids]).mean(axis=0)
        
        # qi标准化
        qi_groups = {qi: [] for qi in range(1, 7)}
        qi_indices = {qi: [] for qi in range(1, 7)}
        
        for t in range(len(palace_data)):
            q = qi_month(months[t])
            if q is not None:
                qi_groups[q].append(palace_data[t])
                qi_indices[q].append(t)
        
        standardized = np.full(len(palace_data), np.nan)
        for qi in range(1, 7):
            vals = np.array(qi_groups[qi])
            idxs = np.array(qi_indices[qi])
            if len(vals) > 0:
                mu = vals.mean()
                sigma = vals.std()
                if sigma > 0:
                    standardized[idxs] = (vals - mu) / sigma
        
        # 逐年分上/下半年
        for year in valid_years:
            mask_upper = (years == year) & (months <= 6) & (~np.isnan(standardized))
            mask_lower = (years == year) & (months > 6) & (~np.isnan(standardized))
            
            upper = standardized[mask_upper].mean() if mask_upper.sum() >= 3 else np.nan
            lower = standardized[mask_lower].mean() if mask_lower.sum() >= 3 else np.nan
            
            half_annual_anomalies[varname][p][year] = {'upper': upper, 'lower': lower}

# 司天管上半年 → 灾宫上半年异常(司天方向)
# 在泉管下半年 → 灾宫下半年异常(在泉方向)
# 预测: 司天与中运同气(天符)→上半年灾宫异常更强

print("\n天符年 vs 非天符年: 灾宫分半年异常强度")
for half in ['upper', 'lower']:
    half_name = '上半年(司天)' if half == 'upper' else '下半年(在泉)'
    
    tianfu_abs = []
    non_tianfu_abs = []
    
    for year in valid_years:
        yun_info = get_yun_from_year(year)
        p = yun_info['disaster_palace']
        var_info = DISASTER_VAR[p]
        varname = var_info['var']
        
        anom = half_annual_anomalies[varname][p].get(year, {}).get(half, np.nan)
        if np.isnan(anom):
            continue
        
        is_tianfu = any(tf['year'] == year for tf in tianfu_years)
        if is_tianfu:
            tianfu_abs.append(abs(anom))
        else:
            non_tianfu_abs.append(abs(anom))
    
    if tianfu_abs and non_tianfu_abs:
        t, pval = stats.ttest_ind(tianfu_abs, non_tianfu_abs)
        print(f"  {half_name}: 天符|异常|={np.mean(tianfu_abs):.4f}(n={len(tianfu_abs)}), "
              f"非天符|异常|={np.mean(non_tianfu_abs):.4f}(n={len(non_tianfu_abs)}), "
              f"t={t:.3f}, p={pval:.4f}")

print("\n" + "=" * 60)
print("实验C完成")
print("=" * 60)
