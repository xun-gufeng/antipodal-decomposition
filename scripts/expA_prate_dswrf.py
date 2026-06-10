"""
实验A: 洛书对宫约束 — 物理独立变量替代测试
水(降水prate) ↔ 火(短波辐射dswrf)

对比:
- 原变量: 寒(-TMIN) ↔ 暑(TMAX) — 同源共线，预期失败
- 新变量: 水(-prate) ↔ 火(dswrf) — 物理独立，预期对宫反号
- 参考:   木(wspd) ↔ 金(100-rhum) — 已知3/4反号

同时保留木↔金作为内部对照。
"""
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy import stats
from itertools import combinations
from numpy.linalg import inv
import datetime, time, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

t0 = time.time()
print("="*72)
print("实验A: 洛书对宫 — 物理独立变量替代 (prate↔dswrf)")
print("="*72)

# ===== 常量 =====
MONTH_TO_QI = {1:0,2:0,3:1,4:1,5:2,6:2,7:3,8:3,9:4,10:4,11:5,12:5}
LUOSHU_WUXING = {1:'水',2:'土',3:'木',4:'木',5:'土',6:'金',7:'金',8:'土',9:'火'}
OPPOSITE_PAIRS = [(1,9),(2,8),(3,7),(4,6)]
OPPOSITE_NAMES = {(1,9):'水↔火',(2,8):'土↔土',(3,7):'木↔金',(4,6):'木↔金'}

lat_lo, lat_hi = 29.62, 39.62
lon_lo, lon_hi = 107.45, 117.45

# ===== 宫位分配 (通用) =====
def assign_palace_2d(lat2d, lon2d):
    pm = {('上','左'):6,('上','中'):1,('上','右'):8,
          ('中','左'):7,('中','中'):5,('中','右'):3,
          ('下','左'):2,('下','中'):9,('下','右'):4}
    out = np.empty(lat2d.shape, dtype=int)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            la, lo = lat2d[i,j], lon2d[i,j]
            r = '上' if la > lat_hi else ('下' if la < lat_lo else '中')
            c = '左' if lo < lon_lo else ('右' if lo > lon_hi else '中')
            out[i,j] = pm.get((r,c), -1)
    return out

# ===== 高效聚合 =====
def ncep_to_df_gauss(data3d, pal2d, yrs, mos, yr_range, vname):
    """NCEP高斯网格月数据 → DataFrame(palace, year, qi, var)"""
    yr0, yr1 = yr_range
    tmask = (yrs >= yr0) & (yrs <= yr1)
    d = data3d[tmask].astype(np.float64)
    d[d < -9000] = np.nan
    
    n_time = d.shape[0]
    pmean = np.full((n_time, 9), np.nan)
    for p in range(1, 10):
        mask = (pal2d == p)
        if mask.any():
            pmean[:, p-1] = np.nanmean(d[:, mask], axis=1)
    
    qi_sub = np.array([MONTH_TO_QI.get(int(m), 0) for m in mos[tmask]])
    yr_sub = yrs[tmask]
    
    rows = []
    for p in range(1, 10):
        col = pmean[:, p-1]
        if np.all(np.isnan(col)): continue
        df_p = pd.DataFrame({'palace':p, 'year':yr_sub, 'qi':qi_sub, vname:col})
        df_p = df_p.groupby(['palace','year','qi'])[vname].mean().reset_index()
        rows.append(df_p)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ===== 加载NCEP 2.5°网格 (rhum, wspd, uwnd, vwnd) =====
ncep_dir = './data/ncep_raw'
YR = (1961, 2020)

print("\n[1] 加载NCEP 2.5°网格数据 (rhum, wspd, vwnd)...")
ds = nc.Dataset(f'{ncep_dir}/rhum.mon.mean.nc')
tv = np.array(ds.variables['time'][:])
origin = datetime.datetime(1800,1,1)
dates = [origin + datetime.timedelta(hours=float(x)) for x in tv]
ncep_yrs_25 = np.array([d.year for d in dates])
ncep_mos_25 = np.array([d.month for d in dates])
ncep_lat_25 = ds.variables['lat'][:]; ncep_lon_25 = ds.variables['lon'][:]
rhum_raw = ds.variables['rhum'][:]; rhum_raw[rhum_raw < -9000] = np.nan
ds.close()

ds = nc.Dataset(f'{ncep_dir}/wspd.mon.mean.nc')
wspd_raw = ds.variables['wspd'][:]; wspd_raw[wspd_raw < -9000] = np.nan
ds.close()

ds = nc.Dataset(f'{ncep_dir}/vwnd.mon.mean.nc')
vwnd_raw = ds.variables['vwnd'][:]; vwnd_raw[vwnd_raw < -9000] = np.nan
ds.close()

print(f"  rhum: {rhum_raw.shape}, wspd: {wspd_raw.shape}, vwnd: {vwnd_raw.shape}")

# 空间截取 + 宫位
LR, LO = (17.5, 42.5), (97.5, 122.5)
lm = (ncep_lat_25>=LR[0])&(ncep_lat_25<=LR[1])
lom = (ncep_lon_25>=LO[0])&(ncep_lon_25<=LO[1])

lat_s = ncep_lat_25[lm]; lon_s = ncep_lon_25[lom]
lg, lng = np.meshgrid(lat_s, lon_s, indexing='ij')
pal2d_25 = assign_palace_2d(lg, lng)

rhum_s = rhum_raw[:, lm, :][:, :, lom]; del rhum_raw
wspd_s = wspd_raw[:, lm, :][:, :, lom]; del wspd_raw
vwnd_s = vwnd_raw[:, lm, :][:, :, lom]; del vwnd_raw

print(f"  2.5°宫位:")
for p in range(1,10):
    print(f"    宫{p}({LUOSHU_WUXING[p]}): {np.sum(pal2d_25==p)}格点")

# 聚合
print("\n  聚合 2.5°...")
df_rh = ncep_to_df_gauss(rhum_s, pal2d_25, ncep_yrs_25, ncep_mos_25, YR, 'rhum'); del rhum_s
df_ws = ncep_to_df_gauss(wspd_s, pal2d_25, ncep_yrs_25, ncep_mos_25, YR, 'wspd'); del wspd_s
df_vw = ncep_to_df_gauss(vwnd_s, pal2d_25, ncep_yrs_25, ncep_mos_25, YR, 'vwnd'); del vwnd_s

# ===== 加载NCEP高斯网格 (prate, dswrf) =====
print("\n[2] 加载NCEP T62高斯网格 (prate, dswrf)...")
ds = nc.Dataset(f'{ncep_dir}/prate.sfc.mon.mean.nc')
gauss_lat = ds.variables['lat'][:]; gauss_lon = ds.variables['lon'][:]
tv_g = np.array(ds.variables['time'][:])
dates_g = [origin + datetime.timedelta(hours=float(x)) for x in tv_g]
gauss_yrs = np.array([d.year for d in dates_g])
gauss_mos = np.array([d.month for d in dates_g])
prate_raw = ds.variables['prate'][:]; prate_raw[prate_raw < -9000] = np.nan
ds.close()

ds = nc.Dataset(f'{ncep_dir}/dswrf.sfc.mon.mean.nc')
dswrf_raw = ds.variables['dswrf'][:]; dswrf_raw[dswrf_raw < -9000] = np.nan
ds.close()

print(f"  prate: {prate_raw.shape}, dswrf: {dswrf_raw.shape}")
print(f"  gauss lat: {gauss_lat.min():.2f} to {gauss_lat.max():.2f}, n={len(gauss_lat)}")
print(f"  gauss lon: {gauss_lon.min():.2f} to {gauss_lon.max():.2f}, n={len(gauss_lon)}")

# 空间截取
lm_g = (gauss_lat>=LR[0])&(gauss_lat<=LR[1])
lom_g = (gauss_lon>=LO[0])&(gauss_lon<=LO[1])
lat_gs = gauss_lat[lm_g]; lon_gs = gauss_lon[lom_g]
lg_g, lng_g = np.meshgrid(lat_gs, lon_gs, indexing='ij')
pal2d_g = assign_palace_2d(lg_g, lng_g)

prate_s = prate_raw[:, lm_g, :][:, :, lom_g]; del prate_raw
dswrf_s = dswrf_raw[:, lm_g, :][:, :, lom_g]; del dswrf_raw

print(f"\n  T62宫位:")
for p in range(1,10):
    print(f"    宫{p}({LUOSHU_WUXING[p]}): {np.sum(pal2d_g==p)}格点")

# 聚合
print("\n  聚合 T62...")
df_pr = ncep_to_df_gauss(prate_s, pal2d_g, gauss_yrs, gauss_mos, YR, 'prate'); del prate_s
df_ds = ncep_to_df_gauss(dswrf_s, pal2d_g, gauss_yrs, gauss_mos, YR, 'dswrf'); del dswrf_s

# ===== 合并 =====
print("\n[3] 合并数据...")
df = df_rh.merge(df_ws, on=['palace','year','qi'], how='inner')
df = df.merge(df_vw, on=['palace','year','qi'], how='inner')
df = df.merge(df_pr, on=['palace','year','qi'], how='inner')
df = df.merge(df_ds, on=['palace','year','qi'], how='inner')

# 变量定义
df['燥'] = 100 - df['rhum']
df['风_wspd'] = df['wspd']
df['风_vwnd'] = df['vwnd']
df['水_prate'] = -df['prate']   # 负号: prate越大→水越多→水(负号)越大
df['火_dswrf'] = df['dswrf']     # dswrf越大→火(热)越大

palaces = sorted(df['palace'].unique())
av_opp = [(a,b) for a,b in OPPOSITE_PAIRS if a in palaces and b in palaces]

print(f"  合并: {len(df)}行, 宫位={palaces}")
for col in ['风_wspd','风_vwnd','燥','水_prate','火_dswrf']:
    nan_c = df[col].isna().sum()
    print(f"  {col}: NaN={nan_c}/{len(df)}")

# ===== 偏相关函数 =====
def pcorr_qi(data, v1, v2):
    d = data[[v1, v2, 'qi']].dropna().values
    if len(d) < 30: return np.nan
    C = np.cov(d, rowvar=False)
    try:
        P = inv(C)
        r = -P[0,1] / np.sqrt(abs(P[0,0]*P[1,1]))
        return np.clip(r, -1, 1)
    except: return np.nan

def scorr_qi(data, v1, v2):
    d = data[[v1, v2, 'qi']].dropna().values
    if len(d) < 30: return np.nan
    # 按qi组内标准化后做简单相关
    qi_vals = d[:,2].astype(int)
    d_z = d[:,:2].copy()
    for q in np.unique(qi_vals):
        m = qi_vals == q
        if m.sum() > 2:
            d_z[m,0] = (d[m,0] - d[m,0].mean()) / (d[m,0].std() + 1e-12)
            d_z[m,1] = (d[m,1] - d[m,1].mean()) / (d[m,1].std() + 1e-12)
    r = np.corrcoef(d_z[:,0], d_z[:,1])[0,1]
    return r

# ===== 实验1: 三套五行变量的对宫反号检验 =====
print(f"\n{'='*72}")
print("实验1: 三套变量体系的对宫反号检验")
print("="*72)

# 定义三套变量
VAR_SETS = {
    '原变量(TMIN/TMAX)': {
        '水': None,   # CUG数据未在此加载，标记为None
        '火': None,
        '木': '风_wspd',
        '金': '燥',
    },
    '新变量(prate/dswrf)': {
        '水': '水_prate',
        '火': '火_dswrf',
        '木': '风_wspd',
        '金': '燥',
    },
    '扩展(vwnd+prate/dswrf)': {
        '水': '水_prate',
        '火': '火_dswrf',
        '木': '风_vwnd',
        '金': '燥',
    },
}

# 只用后两套(有实际变量)
active_sets = {k:v for k,v in VAR_SETS.items() if all(vv is not None for vv in v.values())}

for label, vmap in active_sets.items():
    vlist = [vmap['水'], vmap['火'], vmap['木'], vmap['金']]
    PAIRS_6 = list(combinations(range(4), 2))
    PAIR_NAMES = {(0,1):'水↔火',(0,2):'水↔木',(0,3):'水↔金',
                  (1,2):'火↔木',(1,3):'火↔金',(2,3):'木↔金'}
    PAIR_TYPE = {(0,1):'克(对宫)',(0,2):'生',(0,3):'生',
                 (1,2):'生',(1,3):'克',(2,3):'克(对宫)'}
    
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  变量: 水={vlist[0]}, 火={vlist[1]}, 木={vlist[2]}, 金={vlist[3]}")
    print(f"{'─'*60}")
    
    # 计算偏相关
    R = {}
    for p in palaces:
        dp = df[df['palace']==p]
        for (i,j) in PAIRS_6:
            R[(p,i,j)] = pcorr_qi(dp, vlist[i], vlist[j])
    
    # 对宫反号检验
    rows = []
    for (p1,p2) in av_opp:
        for pair in PAIRS_6:
            r1 = R.get((p1, pair[0], pair[1]), np.nan)
            r2 = R.get((p2, pair[0], pair[1]), np.nan)
            if np.isnan(r1) or np.isnan(r2): continue
            rows.append({
                'opp':(p1,p2), 'opp_name':OPPOSITE_NAMES.get((p1,p2),'?'),
                'pair':pair, 'pname':PAIR_NAMES[pair],
                'ptype':PAIR_TYPE[pair],
                'rA':r1, 'rB':r2,
                'flip': np.sign(r1) != np.sign(r2)
            })
    res = pd.DataFrame(rows)
    
    # 汇总
    nf = res['flip'].sum(); nt = len(res); frac = nf/nt if nt else 0
    try: bp = stats.binomtest(nf, nt, 0.5).pvalue
    except: bp = stats.binom_test(nf, nt, 0.5)
    
    print(f"\n  总计: {nf}/{nt} 反号 ({frac:.1%})  Binomial p={bp:.4f}")
    
    print(f"\n  按五行对:")
    for pair in PAIRS_6:
        s = res[res['pair']==pair]
        if len(s)==0: continue
        n = s['flip'].sum(); t = len(s)
        print(f"    {PAIR_NAMES[pair]:10s}[{PAIR_TYPE[pair]:7s}]: {n}/{t} ({n/t:.0%})")
    
    print(f"\n  按对宫对:")
    for opp in av_opp:
        s = res[res['opp']==opp]
        if len(s)==0: continue
        n = s['flip'].sum(); t = len(s)
        print(f"    宫{opp[0]}↔宫{opp[1]}({OPPOSITE_NAMES[opp]:4s}): {n}/{t} ({n/t:.0%})")
    
    print(f"\n  详细结果(偏相关控制qi):")
    print(f"    {'对宫':>8s} {'五行对':>8s} {'类型':>8s} {'r(宫A)':>8s} {'r(宫B)':>8s} {'反号':>4s}")
    for _, r in res.iterrows():
        print(f"    宫{r['opp'][0]}↔宫{r['opp'][1]} {r['pname']:8s} {r['ptype']:8s} "
              f"{r['rA']:+8.4f} {r['rB']:+8.4f} {'✓' if r['flip'] else '✗':>4s}")

# ===== 实验2: 水↔火 专项深入 =====
print(f"\n{'='*72}")
print("实验2: 水↔火 专项 — prate↔dswrf 对宫反号")
print("="*72)

# 两种风的对比
for wind_var, wind_label in [('风_wspd','wspd'), ('风_vwnd','vwnd')]:
    vlist = ['水_prate', '火_dswrf', wind_var, '燥']
    
    print(f"\n{'─'*60}")
    print(f"  木={wind_label}")
    print(f"{'─'*60}")
    
    # 只看 水↔火 这一对
    for (p1,p2) in av_opp:
        dp1 = df[df['palace']==p1]
        dp2 = df[df['palace']==p2]
        r1 = pcorr_qi(dp1, '水_prate', '火_dswrf')
        r2 = pcorr_qi(dp2, '水_prate', '火_dswrf')
        r1s = scorr_qi(dp1, '水_prate', '火_dswrf')
        r2s = scorr_qi(dp2, '水_prate', '火_dswrf')
        flip_p = np.sign(r1) != np.sign(r2)
        flip_s = np.sign(r1s) != np.sign(r2s)
        
        print(f"  宫{p1}↔宫{p2}({OPPOSITE_NAMES[(p1,p2)]}):")
        print(f"    偏相关: {r1:+.4f} / {r2:+.4f} {'✓ 反号' if flip_p else '✗'}")
        print(f"    简单相关: {r1s:+.4f} / {r2s:+.4f} {'✓ 反号' if flip_s else '✗'}")
    
    # 汇总
    flips_p = 0; flips_s = 0; total = 0
    for (p1,p2) in av_opp:
        dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
        r1 = pcorr_qi(dp1, '水_prate', '火_dswrf')
        r2 = pcorr_qi(dp2, '水_prate', '火_dswrf')
        r1s = scorr_qi(dp1, '水_prate', '火_dswrf')
        r2s = scorr_qi(dp2, '水_prate', '火_dswrf')
        if not (np.isnan(r1) or np.isnan(r2)):
            if np.sign(r1) != np.sign(r2): flips_p += 1
            total += 1
        if not (np.isnan(r1s) or np.isnan(r2s)):
            if np.sign(r1s) != np.sign(r2s): flips_s += 1
    print(f"\n  水↔火 反号汇总: 偏相关 {flips_p}/{total}, 简单相关 {flips_s}/{total}")

# ===== 实验3: 木↔金 对照 (与之前结果比较) =====
print(f"\n{'='*72}")
print("实验3: 木↔金 对照 — wspd/vwnd ↔ 100-rhum")
print("="*72)

for wind_var, wind_label in [('风_wspd','wspd'), ('风_vwnd','vwnd')]:
    print(f"\n  {wind_label} ↔ 燥:")
    flips = 0; total = 0
    for (p1,p2) in av_opp:
        dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
        r1 = pcorr_qi(dp1, wind_var, '燥')
        r2 = pcorr_qi(dp2, wind_var, '燥')
        flip = np.sign(r1) != np.sign(r2) if not (np.isnan(r1) or np.isnan(r2)) else False
        if not (np.isnan(r1) or np.isnan(r2)):
            total += 1
            if flip: flips += 1
        print(f"    宫{p1}↔宫{p2}: {r1:+.4f} / {r2:+.4f} {'✓' if flip else '✗'}")
    print(f"  木↔金 反号: {flips}/{total}")

# ===== 实验4: Permutation test =====
print(f"\n{'='*72}")
print("实验4: Permutation test (1000次)")
print("="*72)

N_PERM = 1000
outer_p = [p for p in palaces if p != 5]

# 对两套主要变量体系做permutation
for label, vlist_labels in [
    ('prate/dswrf + wspd', ['水_prate','火_dswrf','风_wspd','燥']),
    ('prate/dswrf + vwnd', ['水_prate','火_dswrf','风_vwnd','燥']),
]:
    PAIRS_6 = list(combinations(range(4), 2))
    PAIR_NAMES_P = {(0,1):'水↔火',(0,2):'水↔木',(0,3):'水↔金',
                    (1,2):'火↔木',(1,3):'火↔金',(2,3):'木↔金'}
    
    # 计算观测R
    R_obs = {}
    for p in palaces:
        dp = df[df['palace']==p]
        for (i,j) in PAIRS_6:
            R_obs[(p,i,j)] = pcorr_qi(dp, vlist_labels[i], vlist_labels[j])
    
    # 观测反号
    obs_rows = []
    for (p1,p2) in av_opp:
        for pair in PAIRS_6:
            r1 = R_obs.get((p1,pair[0],pair[1]),np.nan)
            r2 = R_obs.get((p2,pair[0],pair[1]),np.nan)
            if np.isnan(r1) or np.isnan(r2): continue
            obs_rows.append({
                'opp':(p1,p2), 'pair':pair, 'flip': np.sign(r1)!=np.sign(r2)
            })
    obs_df = pd.DataFrame(obs_rows)
    obs_frac = obs_df['flip'].sum()/len(obs_df) if len(obs_df) > 0 else 0
    
    # 单独水↔火
    obs_sh = obs_df[obs_df['pair']==(0,1)]
    obs_sh_frac = obs_sh['flip'].sum()/len(obs_sh) if len(obs_sh) > 0 else 0
    # 单独木↔金
    obs_mj = obs_df[obs_df['pair']==(2,3)]
    obs_mj_frac = obs_mj['flip'].sum()/len(obs_mj) if len(obs_mj) > 0 else 0
    
    # Permutation
    perm_all = []
    perm_sh = []
    perm_mj = []
    
    for _ in range(N_PERM):
        sh = np.random.permutation(outer_p)
        rp = [(sh[2*k], sh[2*k+1]) for k in range(4)]
        pr_rows = []
        for (p1,p2) in rp:
            for pair in PAIRS_6:
                r1 = R_obs.get((p1,pair[0],pair[1]),np.nan)
                r2 = R_obs.get((p2,pair[0],pair[1]),np.nan)
                if np.isnan(r1) or np.isnan(r2): continue
                pr_rows.append({'pair':pair,'flip':np.sign(r1)!=np.sign(r2)})
        pr_df = pd.DataFrame(pr_rows)
        if len(pr_df) > 0:
            perm_all.append(pr_df['flip'].sum()/len(pr_df))
            sh_df = pr_df[pr_df['pair']==(0,1)]
            if len(sh_df) > 0: perm_sh.append(sh_df['flip'].sum()/len(sh_df))
            mj_df = pr_df[pr_df['pair']==(2,3)]
            if len(mj_df) > 0: perm_mj.append(mj_df['flip'].sum()/len(mj_df))
    
    perm_all = np.array(perm_all)
    p_all = np.mean(perm_all >= obs_frac)
    
    print(f"\n  {label}:")
    print(f"    全局: obs={obs_frac:.3f}  perm mean={perm_all.mean():.3f}±{perm_all.std():.3f}  p={p_all:.4f}")
    
    perm_sh = np.array(perm_sh)
    perm_mj = np.array(perm_mj)
    if len(perm_sh) > 0:
        p_sh = np.mean(perm_sh >= obs_sh_frac)
        print(f"    水↔火: obs={obs_sh_frac:.3f}  p={p_sh:.4f}")
    if len(perm_mj) > 0:
        p_mj = np.mean(perm_mj >= obs_mj_frac)
        print(f"    木↔金: obs={obs_mj_frac:.3f}  p={p_mj:.4f}")

# ===== 补充: 变量间简单相关 (同宫) =====
print(f"\n{'='*72}")
print("补充: 关键变量对在各宫的简单相关")
print("="*72)

# 不分宫, 全域
print("\n  全域(不分宫):")
dp = df.copy()
for (v1,v2), name in [
    (('水_prate','火_dswrf'),'水↔火'),
    (('风_wspd','燥'),'木↔金(wspd)'),
    (('风_vwnd','燥'),'木↔金(vwnd)'),
    (('水_prate','风_wspd'),'水↔木'),
    (('火_dswrf','燥'),'火↔金'),
]:
    d = dp[[v1,v2,'qi']].dropna()
    r = np.corrcoef(d[v1],d[v2])[0,1]
    rp = pcorr_qi(dp, v1, v2)
    print(f"    {name:16s}: 简单r={r:+.4f}  偏相关r={rp:+.4f}")

# ===== 补充: 各宫 水↔火 原始相关 =====
print(f"\n  各宫 水↔火 (prate↔dswrf) 偏相关(控制qi):")
for p in palaces:
    dp = df[df['palace']==p]
    r = pcorr_qi(dp, '水_prate', '火_dswrf')
    rs = scorr_qi(dp, '水_prate', '火_dswrf')
    print(f"    宫{p}({LUOSHU_WUXING[p]}): 偏相关={r:+.4f}  简单={rs:+.4f}")

# ===== 补充: 六气分层 =====
print(f"\n{'='*72}")
print("补充: 六气分层 水↔火 (prate↔dswrf)")
print("="*72)

QI_NAMES = ['初之气','二之气','三之气','四之气','五之气','终之气']
for qi in range(6):
    print(f"\n  {QI_NAMES[qi]}(qi={qi}):")
    flips_p = 0; total = 0
    for (p1,p2) in av_opp:
        dp1 = df[(df['palace']==p1)&(df['qi']==qi)]
        dp2 = df[(df['palace']==p2)&(df['qi']==qi)]
        if len(dp1) < 10 or len(dp2) < 10: continue
        r1 = np.corrcoef(dp1['水_prate'], dp1['火_dswrf'])[0,1]
        r2 = np.corrcoef(dp2['水_prate'], dp2['火_dswrf'])[0,1]
        flip = np.sign(r1) != np.sign(r2)
        if not (np.isnan(r1) or np.isnan(r2)):
            total += 1
            if flip: flips_p += 1
        print(f"      宫{p1}↔宫{p2}: {r1:+.4f} / {r2:+.4f} {'✓' if flip else '✗'}")
    if total > 0:
        print(f"    → 反号: {flips_p}/{total}")

elapsed = time.time() - t0
print(f"\n{'='*72}")
print(f"完成 | 耗时 {elapsed:.0f}s")
print("="*72)
