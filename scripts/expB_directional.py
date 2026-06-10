"""
实验B: 洛书对宫约束 — 水火方向性通量测试

核心假说:
  水火在洛书中的对冲不是"降水少↔辐射多"的标量反号,
  而是"润下(潜热主导)↔炎上(感热主导)"的方向性对立.

  物理实现:
  - 北方干燥区: 水汽稀缺, 感热占主导 → T↑则rhum↓(负相关) → 灎上方向
  - 南方湿润区: 水汽充沛, 潜热占主导 → T↑则rhum↑(正相关) → 润下方向
  - 宫1(水,北)↔宫9(火,南)的T-rhum耦合应反号

  与实验A对比:
  - 实验A(prate/dswrf): 物理反相关太强太均匀, 0/4反号
  - 实验B(T/rhum): 耦合方向由能量分配决定, 空间异质, 预期可反号

补充测试:
  - Bowen ratio代理: σ(T)/σ(rhum) 宫位分布
  - 热通量(SHTFL/LHTFL)如果已下载
  - 木↔金对照(wspd/燥)复现
  - Permutation统计检验
"""
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy import stats
from itertools import combinations
from numpy.linalg import inv
import datetime, time, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

t0 = time.time()
print("="*72)
print("实验B: 洛书对宫 — 水火方向性通量测试")
print("="*72)

# ===== 常量 =====
MONTH_TO_QI = {1:0,2:0,3:1,4:1,5:2,6:2,7:3,8:3,9:4,10:4,11:5,12:5}
LUOSHU_WUXING = {1:'水',2:'土',3:'木',4:'木',5:'土',6:'金',7:'金',8:'土',9:'火'}
OPPOSITE_PAIRS = [(1,9),(2,8),(3,7),(4,6)]
OPPOSITE_NAMES = {(1,9):'水↔火',(2,8):'土↔土',(3,7):'木↔金',(4,6):'木↔金'}

lat_lo, lat_hi = 29.62, 39.62
lon_lo, lon_hi = 107.45, 117.45
LR, LO = (17.5, 42.5), (97.5, 122.5)
YR = (1961, 2020)

# ===== 宫位分配 =====
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

# ===== 聚合函数 (2.5°网格) =====
def ncep_to_df_25(data3d, pal2d, yrs, mos, yr_range, vname):
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

# ===== 聚合函数 (T62高斯网格) =====
def ncep_to_df_gauss(data3d, pal2d, yrs, mos, yr_range, vname):
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

# ===== 偏相关函数 =====
def pcorr_qi(data, v1, v2):
    """偏相关控制qi (3变量偏相关)"""
    d = data[[v1, v2, 'qi']].dropna().values
    if len(d) < 30: return np.nan
    C = np.cov(d, rowvar=False)
    try:
        P = inv(C)
        r = -P[0,1] / np.sqrt(abs(P[0,0]*P[1,1]))
        return np.clip(r, -1, 1)
    except: return np.nan

def scorr_qi(data, v1, v2):
    """按qi组内标准化后简单相关"""
    d = data[[v1, v2, 'qi']].dropna().values
    if len(d) < 30: return np.nan
    qi_vals = d[:,2].astype(int)
    d_z = d[:,:2].copy()
    for q in np.unique(qi_vals):
        m = qi_vals == q
        if m.sum() > 2:
            d_z[m,0] = (d[m,0] - d[m,0].mean()) / (d[m,0].std() + 1e-12)
            d_z[m,1] = (d[m,1] - d[m,1].mean()) / (d[m,1].std() + 1e-12)
    r = np.corrcoef(d_z[:,0], d_z[:,1])[0,1]
    return r

# ===== 加载NCEP 2.5°网格 =====
ncep_dir = './data/ncep_raw'
origin = datetime.datetime(1800, 1, 1)

print("\n[1] 加载NCEP 2.5°网格数据...")
# 时间轴(从rhum读取)
ds = nc.Dataset(f'{ncep_dir}/rhum.mon.mean.nc')
tv = np.array(ds.variables['time'][:])
dates = [origin + datetime.timedelta(hours=float(x)) for x in tv]
ncep_yrs = np.array([d.year for d in dates])
ncep_mos = np.array([d.month for d in dates])
ncep_lat = ds.variables['lat'][:]; ncep_lon = ds.variables['lon'][:]
rhum_raw = ds.variables['rhum'][:].copy(); rhum_raw[rhum_raw < -9000] = np.nan
ds.close()

ds = nc.Dataset(f'{ncep_dir}/air.mon.mean.nc')
air_raw = ds.variables['air'][:].copy(); air_raw[air_raw < -9000] = np.nan
ds.close()

ds = nc.Dataset(f'{ncep_dir}/wspd.mon.mean.nc')
wspd_raw = ds.variables['wspd'][:].copy(); wspd_raw[wspd_raw < -9000] = np.nan
ds.close()

ds = nc.Dataset(f'{ncep_dir}/vwnd.mon.mean.nc')
vwnd_raw = ds.variables['vwnd'][:].copy(); vwnd_raw[vwnd_raw < -9000] = np.nan
ds.close()

print(f"  air: {air_raw.shape}, rhum: {rhum_raw.shape}, wspd: {wspd_raw.shape}")

# 空间截取
lm = (ncep_lat >= LR[0]) & (ncep_lat <= LR[1])
lom = (ncep_lon >= LO[0]) & (ncep_lon <= LO[1])
lat_s = ncep_lat[lm]; lon_s = ncep_lon[lom]
lg, lng = np.meshgrid(lat_s, lon_s, indexing='ij')
pal2d = assign_palace_2d(lg, lng)

air_s = air_raw[:, lm, :][:, :, lom]; del air_raw
rhum_s = rhum_raw[:, lm, :][:, :, lom]; del rhum_raw
wspd_s = wspd_raw[:, lm, :][:, :, lom]; del wspd_raw
vwnd_s = vwnd_raw[:, lm, :][:, :, lom]; del vwnd_raw

print(f"  2.5°宫位:")
for p in range(1, 10):
    print(f"    宫{p}({LUOSHU_WUXING[p]}): {np.sum(pal2d==p)}格点")

# 聚合
print("  聚合 2.5°...")
df_air = ncep_to_df_25(air_s, pal2d, ncep_yrs, ncep_mos, YR, 'air'); del air_s
df_rh = ncep_to_df_25(rhum_s, pal2d, ncep_yrs, ncep_mos, YR, 'rhum'); del rhum_s
df_ws = ncep_to_df_25(wspd_s, pal2d, ncep_yrs, ncep_mos, YR, 'wspd'); del wspd_s
df_vw = ncep_to_df_25(vwnd_s, pal2d, ncep_yrs, ncep_mos, YR, 'vwnd'); del vwnd_s

# ===== 加载T62高斯网格(prate, dswrf) =====
print("\n[2] 加载NCEP T62高斯网格 (prate, dswrf)...")
ds = nc.Dataset(f'{ncep_dir}/prate.sfc.mon.mean.nc')
gauss_lat = ds.variables['lat'][:]; gauss_lon = ds.variables['lon'][:]
tv_g = np.array(ds.variables['time'][:])
dates_g = [origin + datetime.timedelta(hours=float(x)) for x in tv_g]
gauss_yrs = np.array([d.year for d in dates_g])
gauss_mos = np.array([d.month for d in dates_g])
prate_raw = ds.variables['prate'][:].copy(); prate_raw[prate_raw < -9000] = np.nan
ds.close()

ds = nc.Dataset(f'{ncep_dir}/dswrf.sfc.mon.mean.nc')
dswrf_raw = ds.variables['dswrf'][:].copy(); dswrf_raw[dswrf_raw < -9000] = np.nan
ds.close()

lm_g = (gauss_lat >= LR[0]) & (gauss_lat <= LR[1])
lom_g = (gauss_lon >= LO[0]) & (gauss_lon <= LO[1])
lat_gs = gauss_lat[lm_g]; lon_gs = gauss_lon[lom_g]
lg_g, lng_g = np.meshgrid(lat_gs, lon_gs, indexing='ij')
pal2d_g = assign_palace_2d(lg_g, lng_g)

prate_s = prate_raw[:, lm_g, :][:, :, lom_g]; del prate_raw
dswrf_s = dswrf_raw[:, lm_g, :][:, :, lom_g]; del dswrf_raw

df_pr = ncep_to_df_gauss(prate_s, pal2d_g, gauss_yrs, gauss_mos, YR, 'prate'); del prate_s
df_ds = ncep_to_df_gauss(dswrf_s, pal2d_g, gauss_yrs, gauss_mos, YR, 'dswrf'); del dswrf_s

# ===== 检查热通量数据是否可用 =====
has_flux = False
shtfl_path = f'{ncep_dir}/shtfl.sfc.mon.mean.nc'
lhtfl_path = f'{ncep_dir}/lhtfl.sfc.mon.mean.nc'
if os.path.exists(shtfl_path) and os.path.exists(lhtfl_path):
    sz_sh = os.path.getsize(shtfl_path)
    sz_lh = os.path.getsize(lhtfl_path)
    # 尝试打开验证文件完整性
    try:
        _ds = nc.Dataset(shtfl_path); _ds.close()
        _ds = nc.Dataset(lhtfl_path); _ds.close()
        has_flux = True
        print(f"\n[3] 热通量数据可用! shtfl={sz_sh/1e6:.1f}MB, lhtfl={sz_lh/1e6:.1f}MB")
    except:
        print(f"\n[3] 热通量文件不完整(shtfl={sz_sh/1e6:.1f}MB, lhtfl={sz_lh/1e6:.1f}MB), 跳过")
else:
    print(f"\n[3] 热通量数据未下载, 跳过")

if has_flux:
    ds = nc.Dataset(shtfl_path)
    shtfl_raw = ds.variables['shtfl'][:].copy(); shtfl_raw[shtfl_raw < -9000] = np.nan
    ds.close()
    ds = nc.Dataset(lhtfl_path)
    lhtfl_raw = ds.variables['lhtfl'][:].copy(); lhtfl_raw[lhtfl_raw < -9000] = np.nan
    ds.close()
    
    # 用高斯网格的宫位
    shtfl_s = shtfl_raw[:, lm_g, :][:, :, lom_g]; del shtfl_raw
    lhtfl_s = lhtfl_raw[:, lm_g, :][:, :, lom_g]; del lhtfl_raw
    df_sh = ncep_to_df_gauss(shtfl_s, pal2d_g, gauss_yrs, gauss_mos, YR, 'shtfl'); del shtfl_s
    df_lh = ncep_to_df_gauss(lhtfl_s, pal2d_g, gauss_yrs, gauss_mos, YR, 'lhtfl'); del lhtfl_s

# ===== 合并 =====
print("\n[4] 合并数据...")
df = df_air.merge(df_rh, on=['palace','year','qi'], how='inner')
df = df.merge(df_ws, on=['palace','year','qi'], how='inner')
df = df.merge(df_vw, on=['palace','year','qi'], how='inner')
df = df.merge(df_pr, on=['palace','year','qi'], how='inner')
df = df.merge(df_ds, on=['palace','year','qi'], how='inner')
if has_flux:
    df = df.merge(df_sh, on=['palace','year','qi'], how='inner')
    df = df.merge(df_lh, on=['palace','year','qi'], how='inner')

# 变量定义
df['燥'] = 100 - df['rhum']
df['风_wspd'] = df['wspd']
df['风_vwnd'] = df['vwnd']
df['火_T'] = df['air']       # 温度 → 灎上代理
df['水_rhum'] = df['rhum']   # 湿度 → 润下代理
df['水_prate'] = -df['prate'] # 负号: prate越大→水越多
df['火_dswrf'] = df['dswrf']

if has_flux:
    df['火_shtfl'] = df['shtfl']  # 感热 → 炎上
    df['水_lhtfl'] = df['lhtfl']  # 潜热 → 润下
    # Bowen ratio
    df['bowen'] = df['shtfl'] / (df['lhtfl'] + 1e-10)
    # 水火指数 = (感热-潜热)/(感热+潜热) ∈ (-1,1)
    total_hf = df['shtfl'] + df['lhtfl']
    df['水火指数'] = (df['shtfl'] - df['lhtfl']) / (total_hf.abs() + 1e-10)

palaces = sorted(df['palace'].unique())
av_opp = [(a,b) for a,b in OPPOSITE_PAIRS if a in palaces and b in palaces]
print(f"  合并: {len(df)}行, 宫位={palaces}")

# ===== 变异性代理: Bowen ratio proxy (无热通量数据时) =====
# 用 σ(air)/σ(rhum) 在每个宫位的比值作为能量分配代理
print("\n[5] 计算变异性代理Bowen ratio...")
bowen_proxy = {}
for p in palaces:
    dp = df[df['palace'] == p]
    # 按qi组内标准化后计算标准差
    stds = dp.groupby('qi')[['air','rhum']].std()
    mean_std_T = stds['air'].mean()
    mean_std_rh = stds['rhum'].mean()
    bowen_proxy[p] = mean_std_T / (mean_std_rh + 1e-10)
    
print("  变异性代理 σ(T)/σ(rhum) by 宫位:")
for p in palaces:
    print(f"    宫{p}({LUOSHU_WUXING[p]}): {bowen_proxy[p]:.4f}")

# 对宫对比
print("\n  对宫 σ(T)/σ(rhum) 对比:")
for (p1,p2) in av_opp:
    print(f"    宫{p1}↔宫{p2}({OPPOSITE_NAMES[(p1,p2)]}): "
          f"{bowen_proxy[p1]:.4f} vs {bowen_proxy[p2]:.4f}  "
          f"差={bowen_proxy[p1]-bowen_proxy[p2]:+.4f}")

# ===================================================================
# 核心实验1: T-rhum 耦合方向对宫检验
# ===================================================================
print(f"\n{'='*72}")
print("核心实验1: T-rhum耦合方向 — 水火方向性通量测试")
print("="*72)
print("假说: 北方干燥区T↑→rhum↓(负, 炎上), 南方湿润区T↑→rhum↑(正, 润下)")
print("      → 宫1(水/北)↔宫9(火/南) T-rhum偏相关应反号")

# 逐宫T-rhum偏相关
T_rhum_by_palace = {}
print(f"\n  各宫 T(air)↔rhum 偏相关(控制qi):")
for p in palaces:
    dp = df[df['palace'] == p]
    rp = pcorr_qi(dp, 'air', 'rhum')
    rs = scorr_qi(dp, 'air', 'rhum')
    T_rhum_by_palace[p] = {'pcorr': rp, 'scorr': rs}
    print(f"    宫{p}({LUOSHU_WUXING[p]}): 偏相关={rp:+.4f}  qi标准化相关={rs:+.4f}")

# 对宫反号检验
print(f"\n  对宫 T↔rhum 反号检验:")
n_flip = 0; n_total = 0
for (p1,p2) in av_opp:
    r1 = T_rhum_by_palace[p1]['pcorr']
    r2 = T_rhum_by_palace[p2]['pcorr']
    flip = (not np.isnan(r1)) and (not np.isnan(r2)) and (np.sign(r1) != np.sign(r2))
    n_total += 1
    if flip: n_flip += 1
    print(f"    宫{p1}↔宫{p2}({OPPOSITE_NAMES[(p1,p2)]}): "
          f"{r1:+.4f} / {r2:+.4f} {'✓ 反号' if flip else '✗'}")
print(f"\n  T↔rhum 对宫反号: {n_flip}/{n_total}")

# ===================================================================
# 核心实验2: 完整五行体系 — T替代dswrf作为火变量
# ===================================================================
print(f"\n{'='*72}")
print("核心实验2: 完整五行体系 — T(火)/rhum(水)/wspd(木)/燥(金)")
print("="*72)

# 四变量体系
VARS = {
    '水': 'rhum',    # 湿度 = 润下
    '火': 'air',     # 温度 = 炎上
    '木': 'wspd',    # 风
    '金': '燥',      # 100-rhum
}
vlist = [VARS['水'], VARS['火'], VARS['木'], VARS['金']]

PAIRS_6 = list(combinations(range(4), 2))
PAIR_NAMES = {(0,1):'水↔火(湿↔温)',(0,2):'水↔木(湿↔风)',(0,3):'水↔金(湿↔燥)',
              (1,2):'火↔木(温↔风)',(1,3):'火↔金(温↔燥)',(2,3):'木↔金(风↔燥)'}
PAIR_TYPE = {(0,1):'克(对宫)',(0,2):'生',(0,3):'克(反五行)',
             (1,2):'生',(1,3):'克',(2,3):'克(对宫)'}

# 计算偏相关矩阵
R = {}
for p in palaces:
    dp = df[df['palace'] == p]
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

nf = res['flip'].sum(); nt = len(res)
try: bp = stats.binomtest(nf, nt, 0.5).pvalue
except: bp = stats.binom_test(nf, nt, 0.5)

print(f"\n  总计: {nf}/{nt} 反号 ({nf/nt:.1%})  Binomial p={bp:.4f}")

print(f"\n  按五行对:")
for pair in PAIRS_6:
    s = res[res['pair']==pair]
    if len(s)==0: continue
    n = s['flip'].sum(); t = len(s)
    print(f"    {PAIR_NAMES[pair]:16s}[{PAIR_TYPE[pair]:10s}]: {n}/{t} ({n/t:.0%})")

print(f"\n  按对宫对:")
for opp in av_opp:
    s = res[res['opp']==opp]
    if len(s)==0: continue
    n = s['flip'].sum(); t = len(s)
    print(f"    宫{opp[0]}↔宫{opp[1]}({OPPOSITE_NAMES[opp]:4s}): {n}/{t} ({n/t:.0%})")

print(f"\n  详细(偏相关控制qi):")
print(f"    {'对宫':>8s} {'五行对':>16s} {'类型':>10s} {'r(宫A)':>8s} {'r(宫B)':>8s} {'反号':>4s}")
for _, r in res.iterrows():
    print(f"    宫{r['opp'][0]}↔宫{r['opp'][1]} {r['pname']:16s} {r['ptype']:10s} "
          f"{r['rA']:+8.4f} {r['rB']:+8.4f} {'✓' if r['flip'] else '✗':>4s}")

# ===================================================================
# 实验3: 水↔火 专项对比 — 三套火变量
# ===================================================================
print(f"\n{'='*72}")
print("实验3: 水↔火 — 三套火变量对比")
print("="*72)

fire_vars = [
    ('air(温度=炎上)', 'air', 'rhum'),
    ('dswrf(辐射)', 'dswrf', 'prate'),  # 注意: 用prate(非负号)
    ('shtfl(感热=炎上)', 'shtfl', 'lhtfl') if has_flux else None,
]

for item in fire_vars:
    if item is None: continue
    label, fv, wv = item
    print(f"\n{'─'*60}")
    print(f"  火={label}  水={wv}")
    print(f"{'─'*60}")
    
    flips = 0; total = 0
    for (p1,p2) in av_opp:
        dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
        if fv not in df.columns or wv not in df.columns:
            continue
        r1 = pcorr_qi(dp1, fv, wv)
        r2 = pcorr_qi(dp2, fv, wv)
        flip = (not np.isnan(r1)) and (not np.isnan(r2)) and (np.sign(r1) != np.sign(r2))
        total += 1
        if flip: flips += 1
        print(f"    宫{p1}↔宫{p2}({OPPOSITE_NAMES[(p1,p2)]}): "
              f"{r1:+.4f} / {r2:+.4f} {'✓' if flip else '✗'}")
    print(f"  → 反号: {flips}/{total}")

# ===================================================================
# 实验4: Permutation test
# ===================================================================
print(f"\n{'='*72}")
print("实验4: Permutation test (1000次)")
print("="*72)

N_PERM = 1000
outer_p = [p for p in palaces if p != 5]

# 对 T-rhum 和 wspd-燥 两个核心对做permutation
test_pairs = [
    ('T↔rhum (水火方向)', 'air', 'rhum'),
    ('wspd↔燥 (木金)', 'wspd', '燥'),
]

for label, v1, v2 in test_pairs:
    # 观测R
    R_obs = {}
    for p in palaces:
        dp = df[df['palace']==p]
        R_obs[p] = pcorr_qi(dp, v1, v2)
    
    # 观测反号数
    obs_flips = 0; obs_total = 0
    for (p1,p2) in av_opp:
        r1 = R_obs.get(p1, np.nan)
        r2 = R_obs.get(p2, np.nan)
        if np.isnan(r1) or np.isnan(r2): continue
        obs_total += 1
        if np.sign(r1) != np.sign(r2): obs_flips += 1
    obs_frac = obs_flips / obs_total if obs_total > 0 else 0
    
    # Permutation
    perm_fracs = []
    for _ in range(N_PERM):
        sh = np.random.permutation(outer_p)
        rp = [(sh[2*k], sh[2*k+1]) for k in range(4)]
        pf = 0; pt = 0
        for (p1,p2) in rp:
            r1 = R_obs.get(p1, np.nan)
            r2 = R_obs.get(p2, np.nan)
            if np.isnan(r1) or np.isnan(r2): continue
            pt += 1
            if np.sign(r1) != np.sign(r2): pf += 1
        if pt > 0: perm_fracs.append(pf / pt)
    
    perm_fracs = np.array(perm_fracs)
    p_val = np.mean(perm_fracs >= obs_frac)
    
    print(f"\n  {label}:")
    print(f"    观测: {obs_flips}/{obs_total} 反号 ({obs_frac:.1%})")
    print(f"    Permutation: mean={perm_fracs.mean():.3f}±{perm_fracs.std():.3f}")
    print(f"    p-value: {p_val:.4f}")

# ===================================================================
# 实验5: 水↔火 — 全域vs分宫 耦合结构对比
# ===================================================================
print(f"\n{'='*72}")
print("实验5: 耦合结构空间异质性")
print("="*72)

# 全域T-rhum相关(不分宫)
dp = df.copy()
r_global = pcorr_qi(dp, 'air', 'rhum')
rs_global = scorr_qi(dp, 'air', 'rhum')
print(f"\n  全域 T↔rhum: 偏相关={r_global:+.4f}  qi标准化相关={rs_global:+.4f}")

# 逐宫相关 → 空间异质性指标
print(f"\n  逐宫 T↔rhum 偏相关(控制qi):")
palace_rs = []
for p in palaces:
    dp = df[df['palace'] == p]
    r = pcorr_qi(dp, 'air', 'rhum')
    palace_rs.append(r)
    print(f"    宫{p}({LUOSHU_WUXING[p]}): {r:+.4f}")

palace_rs = np.array(palace_rs)
print(f"\n  空间异质性: range={palace_rs.max()-palace_rs.min():.4f}, "
      f"std={palace_rs.std():.4f}")
print(f"  符号分布: 正号={np.sum(palace_rs>0)}, 负号={np.sum(palace_rs<0)}")

# 对比: wspd↔燥 的空间异质性
print(f"\n  对照: 逐宫 wspd↔燥 偏相关(控制qi):")
palace_rs2 = []
for p in palaces:
    dp = df[df['palace'] == p]
    r = pcorr_qi(dp, 'wspd', '燥')
    palace_rs2.append(r)
    print(f"    宫{p}({LUOSHU_WUXING[p]}): {r:+.4f}")

palace_rs2 = np.array(palace_rs2)
print(f"  空间异质性: range={palace_rs2.max()-palace_rs2.min():.4f}, "
      f"std={palace_rs2.std():.4f}")
print(f"  符号分布: 正号={np.sum(palace_rs2>0)}, 负号={np.sum(palace_rs2<0)}")

# ===================================================================
# 实验6: Bowen ratio 如果有热通量数据
# ===================================================================
if has_flux:
    print(f"\n{'='*72}")
    print("实验6: Bowen ratio 直接测试")
    print("="*72)
    
    # 逐宫Bowen ratio
    print(f"\n  逐宫 Bowen ratio (=SHTFL/LHTFL):")
    bowen_palace = {}
    for p in palaces:
        dp = df[df['palace'] == p]
        br = dp['bowen'].mean()
        si = dp['水火指数'].mean()
        bowen_palace[p] = br
        print(f"    宫{p}({LUOSHU_WUXING[p]}): β={br:.4f}  水火指数={si:+.4f}")
    
    # 对宫
    print(f"\n  对宫 Bowen ratio:")
    for (p1,p2) in av_opp:
        print(f"    宫{p1}↔宫{p2}({OPPOSITE_NAMES[(p1,p2)]}): "
              f"β={bowen_palace[p1]:.4f} vs {bowen_palace[p2]:.4f}")
    
    # SHTFL↔LHTFL 偏相关
    print(f"\n  SHTFL↔LHTFL 偏相关(控制qi):")
    for p in palaces:
        dp = df[df['palace'] == p]
        r = pcorr_qi(dp, 'shtfl', 'lhtfl')
        print(f"    宫{p}({LUOSHU_WUXING[p]}): {r:+.4f}")
    
    # 对宫反号
    print(f"\n  SHTFL↔LHTFL 对宫反号:")
    flips = 0; total = 0
    for (p1,p2) in av_opp:
        dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
        r1 = pcorr_qi(dp1, 'shtfl', 'lhtfl')
        r2 = pcorr_qi(dp2, 'shtfl', 'lhtfl')
        flip = (not np.isnan(r1)) and (not np.isnan(r2)) and (np.sign(r1) != np.sign(r2))
        total += 1
        if flip: flips += 1
        print(f"    宫{p1}↔宫{p2}: {r1:+.4f} / {r2:+.4f} {'✓' if flip else '✗'}")
    print(f"  → 反号: {flips}/{total}")

# ===================================================================
# 实验7: 土↔土 对宫检验 (宫2↔宫8 对照)
# ===================================================================
print(f"\n{'='*72}")
print("实验7: 土↔土 对照 — 宫2↔宫8同五行不应反号")
print("="*72)

# 同五行对宫, 任何变量对都不应系统性反号
print(f"\n  宫2(土)↔宫8(土) 各变量对偏相关:")
same_opp = [(2,8)]
test_vars = [('air','rhum','T↔rhum'),('wspd','燥','风↔燥'),('prate','dswrf','水↔火')]
for (p1,p2) in same_opp:
    for v1,v2,name in test_vars:
        if v1 not in df.columns or v2 not in df.columns: continue
        dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
        r1 = pcorr_qi(dp1, v1, v2)
        r2 = pcorr_qi(dp2, v1, v2)
        flip = (not np.isnan(r1)) and (not np.isnan(r2)) and (np.sign(r1) != np.sign(r2))
        print(f"    {name}: {r1:+.4f} / {r2:+.4f} {'✓ 反号' if flip else '✗ 同号'}")

# ===================================================================
# 汇总
# ===================================================================
print(f"\n{'='*72}")
print("汇总: 三套水火变量对宫反号对比")
print("="*72)

summary = []
# T↔rhum
flips = 0; total = 0
for (p1,p2) in av_opp:
    r1 = T_rhum_by_palace[p1]['pcorr']
    r2 = T_rhum_by_palace[p2]['pcorr']
    if not (np.isnan(r1) or np.isnan(r2)):
        total += 1
        if np.sign(r1) != np.sign(r2): flips += 1
summary.append(('T↔rhum (方向性)', flips, total))

# prate↔dswrf (从实验A已知)
flips2 = 0; total2 = 0
for (p1,p2) in av_opp:
    dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
    r1 = pcorr_qi(dp1, 'prate', 'dswrf')
    r2 = pcorr_qi(dp2, 'prate', 'dswrf')
    if not (np.isnan(r1) or np.isnan(r2)):
        total2 += 1
        if np.sign(r1) != np.sign(r2): flips2 += 1
summary.append(('prate↔dswrf (标量)', flips2, total2))

# wspd↔燥
flips3 = 0; total3 = 0
for (p1,p2) in av_opp:
    dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
    r1 = pcorr_qi(dp1, 'wspd', '燥')
    r2 = pcorr_qi(dp2, 'wspd', '燥')
    if not (np.isnan(r1) or np.isnan(r2)):
        total3 += 1
        if np.sign(r1) != np.sign(r2): flips3 += 1
summary.append(('wspd↔燥 (木金对照)', flips3, total3))

if has_flux:
    flips4 = 0; total4 = 0
    for (p1,p2) in av_opp:
        dp1 = df[df['palace']==p1]; dp2 = df[df['palace']==p2]
        r1 = pcorr_qi(dp1, 'shtfl', 'lhtfl')
        r2 = pcorr_qi(dp2, 'shtfl', 'lhtfl')
        if not (np.isnan(r1) or np.isnan(r2)):
            total4 += 1
            if np.sign(r1) != np.sign(r2): flips4 += 1
    summary.append(('shtfl↔lhtfl (Bowen)', flips4, total4))

print(f"\n  {'变量对':>24s} {'反号数':>6s} {'比例':>6s}")
for name, f, t in summary:
    print(f"  {name:>24s} {f}/{t:>2d}    {f/t:.0%}" if t > 0 else f"  {name:>24s} N/A")

elapsed = time.time() - t0
print(f"\n{'='*72}")
print(f"完成 | 耗时 {elapsed:.0f}s")
print("="*72)
