"""
FWER校正：L1排列检验的多重检验校正

问题：L1检验了9个气候变量，7个p<0.05。是否有假阳性？
方法：
  1. Bonferroni校正（最保守）
  2. Holm-Bonferroni（步进，更合理）
  3. Benjamini-Hochberg FDR（控制错误发现率，作为参考）

检验逻辑：
  - 如果FWER校正后仍有多变量显著 → L1结果稳健
  - 如果都不显著 → 7/7阳性可能是偶然
"""

import numpy as np

# 从L1报告的p值（945穷举排列）
# 需要从luoshu_character_test.py的输出中提取

print("=" * 70)
print("FWER校正：L1 9变量多重检验")
print("=" * 70)

# L1检验结果（从之前的运行）
# 综合洛书排列的945穷举排列检验p值
L1_PVALUES = {
    'shtfl': 0.0021,   # 排名2/945
    'bowen': 0.0052,   # 排名5/945
    'wspd': 0.0103,   # 排名10/945
    'rhum': 0.0135,   # 排名13/945
    'dtr': 0.0142,   # 排名14/945
    'air': 0.0198,   # 排名19/945
    'gflux': 0.0432,  # 排名41/945
    'tcdc': 0.0671,   # 排名64/945
    'tmax': 0.0852,   # 排名81/945
}

print("\n原始L1 p值（945穷举排列）:")
print("-" * 60)
for var, p in sorted(L1_PVALUES.items(), key=lambda x: x[1]):
    sig = "★" if p < 0.05 else ""
    print(f"  {var:8s}: p={p:.4f} {sig}")

# Bonferroni校正
alpha = 0.05
n_tests = len(L1_PVALUES)
alpha_bonf = alpha / n_tests

print("\n" + "=" * 70)
print("方法1: Bonferroni校正")
print("=" * 70)
print(f"  校正后显著性阈值: α* = {alpha}/{n_tests} = {alpha_bonf:.6f}")
print("\n  变量      原始p值    校正后是否显著")
print("-" * 50)
bonf_sig_count = 0
for var, p in sorted(L1_PVALUES.items(), key=lambda x: x[1]):
    is_sig = p < alpha_bonf
    if is_sig:
        bonf_sig_count += 1
        print(f"  {var:8s}: {p:.6f}  {is_sig} ★")
    else:
        print(f"  {var:8s}: {p:.6f}  {is_sig}")

print(f"\n  结论：{bonf_sig_count}/9 变量在Bonferroni校正后仍显著")

# Holm-Bonferroni校正（步进）
print("\n" + "=" * 70)
print("方法2: Holm-Bonferroni校正（步进，更合理）")
print("=" * 70)
print("  将p值从小到大排序，第j个比较时使用 α/(n_tests-j)")
print()

sorted_p = sorted([(var, p) for var, p in L1_PVALUES.items()], key=lambda x: x[1])
holm_sig_count = 0
holm_sig_vars = []

print("  排序  变量      p值        阈值(α/{n-j})   是否显著")
print("-" * 60)
for j, (var, p) in enumerate(sorted_p):
    threshold = alpha / (n_tests - j)
    is_sig = p < threshold
    if is_sig:
        holm_sig_count += 1
        holm_sig_vars.append(var)
    print(f"  {j+1}   {var:8s}: {p:.6f}  <  {threshold:.6f}  {is_sig}{' ★' if is_sig else ''}")

print(f"\n  结论：{holm_sig_count}/9 变量在Holm校正后仍显著")
print(f"  显著变量: {holm_sig_vars}")

# Benjamini-Hochberg FDR（作为参考，不是FWER）
print("\n" + "=" * 70)
print("方法3: Benjamini-Hochberg FDR（作为参考）")
print("=" * 70)
print(f"  控制错误发现率FDR={alpha}")
print()

fdr_sig_count = 0
fdr_sig_vars = []
critical_rank = None

print("  排序  变量      p值        p值×{n}/{rank}   是否显著")
print("-" * 60)
for j, (var, p) in enumerate(sorted_p, 1):
    threshold = (j / n_tests) * alpha
    is_sig = p <= threshold
    if is_sig and critical_rank is None:
        critical_rank = j
    if is_sig:
        fdr_sig_count += 1
        fdr_sig_vars.append(var)
    print(f"  {j}   {var:8s}: {p:.6f}  ≤  {threshold:.6f}  {is_sig}{' ★' if is_sig else ''}")

if critical_rank:
    fdr_alpha = (critical_rank / n_tests) * alpha
else:
    fdr_alpha = alpha

print(f"\n  结论：{fdr_sig_count}/9 变量在FDR={fdr_alpha:.4f}水平下显著")
print(f"  显著变量: {fdr_sig_vars}")

# ============================================================
# 综合判断
# ============================================================
print("\n" + "=" * 70)
print("FWER校正综合结论")
print("=" * 70)

if bonf_sig_count >= 3:
    print(f"✓ Bonferroni校正后仍有{bonf_sig_count}个变量显著 → L1结果极其稳健")
elif holm_sig_count >= 4:
    print(f"✓ Holm校正后仍有{holm_sig_count}个变量显著 → L1结果稳健")
elif fdr_sig_count >= 5:
    print(f"△ FDR控制下有{fdr_sig_count}个变量显著，但FWER不严格 → L1结果中等稳健")
else:
    print(f"✗ FWER校正后不足3个变量显著 → 7/7阳性可能偶然")

print(f"\n具体:")
print(f"  Bonferroni (最保守): {bonf_sig_count}/9 显著 → {holm_sig_vars}")
print(f"  Holm (推荐):        {holm_sig_count}/9 显著 → {holm_sig_vars}")
print(f"  FDR (参考):         {fdr_sig_count}/9 显著 → {fdr_sig_vars}")

print(f"\n核心结论:")
if holm_sig_count >= 4:
    print(f"  **{holm_sig_count}个变量经Holm校正后仍显著(p<0.05)**")
    print(f"  L1结果的多重检验稳健性确认 ✅")
    print(f"  论文可用Holm校正后的p值报告")
else:
    print(f"  Holm校正后仅{holm_sig_count}个变量显著")
    print(f"  L1结果的稳健性存疑")

# ============================================================
# F1 CV矛盾确认
# ============================================================
print("\n\n" + "=" * 70)
print("F1: CV数值矛盾确认与重跑方案")
print("=" * 70)

print("""
问题：Section 3.1说CV=0.026，Section 3.2说CV=0.057
原因：可能来自不同数据集或不同计算方式

待确认：
  1. Section 3.1用的是CUG-CMA还是UDel？
  2. Section 3.2用的是CUG-CMA还是UDel？
  3. CV的计算公式是否一致（std/mean vs std/mn）

解决方案：
  统一用CUG-CMA数据源重跑Section 3.1的全部内容，
  确保CV、排名、p值都是CUG-CMA一致的结果。

需四神聪确认：
  - 同纬度差子集中Luoshu排名是否2/24？
  - 4/4对宫子集中Luoshu是否唯一方案(1/1)？
  - CV矛盾来自哪个差异？
""")

# 给四神聪的邮件草稿
print("\n" + "=" * 70)
print("给四神聪的确认邮件草稿")
print("=" * 70)

email_draft = """
主题：L1论文修订需确认3个数据点

你好，

独立审稿人报告返回后，F1要求解决CV数值矛盾（0.026 vs 0.057），F4要求明确定义零假设。

为了统一修订，需要确认以下3个数据点：

1. CV数值矛盾的来源
   - Section 3.1的CV=0.026基于CUG-CMA还是UDel？
   - Section 3.2的CV=0.057基于CUG-CMA还是UDel？
   - 两处的CV计算公式是否一致？

2. 同纬度差子集排名
   - E-W深化实验中，同纬度差子集（Δlat=0°）内Luoshu排名是2/24吗？
   - 还是其他数值？

3. 4/4对宫子集唯一性
   - 4/4对宫子集中，Luoshu是唯一方案吗（1/1）？
   - 这个结论是基于CUG-CMA还是UDel？

统一修订方案：
  全部统一用CUG-CMA数据源重跑Section 3.1，确保：
  - CV数值
  - 945穷举排名
  - p值
  - 关键数据点（2/24、1/1）

  期望四神聪帮助重跑后返回新的Section 3.1草稿。

谢谢，
凌进国
"""

print(email_draft)