# -*- coding: utf-8 -*-
# ヴェロビ買い目（競輪専用・簡潔UI・5〜18対応）
# 要望仕様：
# 1) 最初に「何車」を選ぶ（既定=7）
# 2) ラインを必要なら入力
# 3) 脚質を入力（逃/両/追）
# 4) 指数（=評価順/確率）を入力
# 5) 結果：単勝・複勝・二車複・ワイド・三連複・三連単
# - オッズ非使用。確率ベース（p1/p2/p3）＋ライン/脚質係数
# - アンカー（参考予想補助）: force/boost/none

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="ヴェロビ買い目・簡潔UI（5〜18）", layout="wide")
st.title("ヴェロビ買い目：単・複・二車複・ワイド・三連複・三連単")
st.caption("参考予想補助｜これは娯楽としてのものであり、結果については一切の責任を負いません。")

# ----------------- プロファイル（7車→Nへ補間） -----------------
BASE_P12  = {1:0.382, 2:0.307, 3:0.278, 4:0.321, 5:0.292, 6:0.279, 7:0.171}  # 連対率
BASE_PIN3 = {1:0.519, 2:0.476, 3:0.472, 4:0.448, 5:0.382, 6:0.358, 7:0.322}  # 3着内
DEFAULT_P1_RATIO = 0.481
BASE_P1 = {r: max(min(BASE_P12[r]*DEFAULT_P1_RATIO, 0.95), 1e-4) for r in range(1,8)}

K12 = {
    "初日":   {1:0.9240837696, 2:1.4169381107, 3:1.0575539568, 4:0.8785046729, 5:0.6849315068, 6:1.0967741935, 7:0.7543859649},
    "2日目":  {1:1.1361256545, 2:0.6416938111, 3:0.8992805755, 4:1.1059190031, 5:1.2157534247, 6:0.9892473118, 7:1.2339181287},
    "最終日": {1:0.9240837696, 2:0.8306188925, 3:1.0575539568, 4:1.0373831776, 5:1.2089041096, 6:0.8422939068, 7:1.0526315789},
}
K1 = {"初日":{1:0.766,2:1.282,3:1.106,4:1.000,5:0.746,6:1.110,7:0.728}, "2日目":{}, "最終日":{}}
for r in range(1,8):
    K1["2日目"][r]  = K12["2日目"][r]
    K1["最終日"][r] = K12["最終日"][r]
K3 = {
    "初日":   {1:0.9749518304, 2:1.2857142857, 3:1.1207627119, 4:0.9196428571, 5:0.8010471204, 6:1.0502793296, 7:0.8043478261},
    "2日目":  {1:1.1156069364, 2:0.7457983193, 3:0.8368644068, 4:0.9687500000, 5:1.1020942408, 6:1.0279329609, 7:1.1428571429},
    "最終日": {1:0.8689788054, 2:0.9054621849, 3:1.0381355932, 4:1.1808035714, 5:1.1806282723, 6:0.8770949721, 7:1.1180124224},
}

LINE_COEF = {"初日":{"adj":1.35,"same":1.10,"diff":1.00}, "2日目":{"adj":1.20,"same":1.05,"diff":1.00}, "最終日":{"adj":1.25,"same":1.05,"diff":1.00}}

def style_factor_same_line(pos_a:int, pos_b:int)->float:
    a,b = sorted([pos_a,pos_b])
    if (a,b)==(1,2): return 1.15
    if (a,b)==(2,3): return 1.08
    if (a,b)==(1,3): return 1.03
    return 1.00

STYLE_COEF_DIFF = {("逃","逃"):0.90, ("両","逃"):0.95, ("逃","両"):0.95,
                   ("追","追"):1.00, ("両","追"):1.00, ("追","両"):1.00,
                   ("逃","追"):1.00, ("追","逃"):1.00, ("両","両"):1.00}

# ---------- 補助 ----------

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(0.9999, x))


def _interp_profile(base7: Dict[int,float], N: int) -> Dict[int,float]:
    xs = [(i-1)/6 for i in range(1,8)]
    ys = [base7[i] for i in range(1,8)]
    def interp(x: float) -> float:
        if x <= xs[0]: return ys[0]
        if x >= xs[-1]: return ys[-1]
        for a in range(6):
            if xs[a] <= x <= xs[a+1]:
                t = (x - xs[a]) / (xs[a+1]-xs[a])
                return ys[a]*(1-t) + ys[a+1]*t
        return ys[-1]
    return {r: float(interp(0.0 if N==1 else (r-1)/(N-1))) for r in range(1, N+1)}


def parse_line_pattern(pattern: str, N: int):
    pattern = re.sub(r"[^\d\s\-\|]", "", pattern).strip()
    chunks = [c for c in re.split(r"[\s\|]+", pattern) if c]
    id_map, pos_map, used = {}, {}, set()
    gid = 0
    for g in chunks:
        toks = [int(x) for x in re.findall(r"\d", g)]
        if not toks: continue
        lid = chr(ord('A')+gid)
        for idx, n in enumerate(toks,1):
            if 1<=n<=N and n not in used:
                id_map[n]=lid; pos_map[n]=idx; used.add(n)
        gid += 1
    for n in range(1,N+1):
        if n not in used:
            lid = chr(ord('A')+gid)
            id_map[n]=lid; pos_map[n]=1; gid += 1
    return id_map, pos_map


def parse_car_list(s: str, N: int) -> List[int]:
    if not s: return []
    s = s.replace("、"," ").replace(","," ").replace("　"," ")\
         .replace("・"," ").replace("/"," ").replace("-"," ")
    out = []
    for tok in s.split():
        tok = re.sub(r"\D", "", tok)
        if tok and 1 <= int(tok) <= N:
            out.append(int(tok))
    return out


def parse_verovi_order(s: str, N: int) -> List[int]:
    if not s: return []
    t = re.sub(r"\s+", "", s)
    if N <= 9 and re.fullmatch(r"[1-9]{%d}" % N, t):
        order = [int(ch) for ch in t]
    else:
        toks = [int(x) for x in re.findall(r"\d+", s)]
        if len(toks) != N: return []
        order = toks
    return order if sorted(order)==list(range(1,N+1)) else []

# ---------- PL（順序あり確率） ----------

def odds_ratio(x: float) -> float:
    x = clamp01(x)
    return x / max(1e-9, 1.0-x)


def pl2(strength: Dict[int,float], i:int, j:int) -> float:
    S = sum(strength.values())
    if S <= 0: return 0.0
    out = 0.0
    for a,b in ((i,j),(j,i)):
        sa, sb = strength[a], strength[b]
        out += (sa/S) * (sb/(S-sa)) if S-sa>0 else 0.0
    return out


def pl3(strength: Dict[int,float], i:int, j:int, k:int) -> float:
    S = sum(strength.values())
    if S <= 0: return 0.0
    out = 0.0
    for a,b,c in ((i,j,k),(i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i)):
        sa, sb, sc = strength[a], strength[b], strength[c]
        if S-sa>0 and S-sa-sb>0:
            out += (sa/S) * (sb/(S-sa)) * (sc/(S-sa-sb))
    return out

# ---------- 入力ステップ ----------

c1,c2,c3 = st.columns([1,1,1])
with c1:
    N = st.number_input("何車（5〜18）", min_value=5, max_value=18, value=7, step=1)
with c2:
    day = st.selectbox("開催", ["初日","2日目","最終日"], index=0)
with c3:
    anchor = st.selectbox("参考予想補助(◎) 番号", options=["なし"] + list(range(1,int(N)+1)), index=0)
    anchor = None if anchor=="なし" else int(anchor)

st.subheader("ライン（任意） 例: 123 45 6 7 / 1-2-3 | 4-5 | 6 | 7")
pattern = st.text_input("ラインパターン", value="")
id_map, pos_map = parse_line_pattern(pattern, int(N))

st.subheader("脚質（任意・車番を空白区切り）")
colA,colB,colC = st.columns(3)
with colA:
    raw_nige = st.text_input("逃", value="")
with colB:
    raw_ryo  = st.text_input("両", value="")
with colC:
    raw_tsu  = st.text_input("追", value="")

default_style = st.radio("未指定の既定脚質", ["追","両","逃"], index=0, horizontal=True)
list_nige = parse_car_list(raw_nige, int(N))
list_ryo  = parse_car_list(raw_ryo,  int(N))
list_tsu  = parse_car_list(raw_tsu,  int(N))

# スタイル辞書
style_map = {i: ("逃" if i in list_nige else ("両" if i in list_ryo else ("追" if i in list_tsu else default_style))) for i in range(1,int(N)+1)}
# 重複チェック
_dup = (set(list_nige)&set(list_ryo)) | (set(list_nige)&set(list_tsu)) | (set(list_ryo)&set(list_tsu))
if _dup:
    st.error(f"脚質の重複指定: {sorted(_dup)}")

st.subheader("指数入力（どちらか）")
mode = st.radio("方法", ["ヴェロビ評価順（1行）","手動で確率を入力"], horizontal=True)

p1: Dict[int,float] = {}
p12: Dict[int,float] = {}
p3: Dict[int,float] = {}

if mode.startswith("ヴェロビ"):
    order_str = st.text_input("ヴェロビ評価順 例: 3142576 / 3 1 4 2 5 7 6", value="")
    order = parse_verovi_order(order_str, int(N))
    if order:
        rank_map = {car: i+1 for i,car in enumerate(order)}
        # 7→N補間
        bp1  = _interp_profile(BASE_P1,   int(N))
        bp12 = _interp_profile(BASE_P12,  int(N))
        bp3p = _interp_profile(BASE_PIN3, int(N))
        k1   = _interp_profile(K1[day],   int(N))
        k12  = _interp_profile(K12[day],  int(N))
        k3   = _interp_profile(K3[day],   int(N))
        for no,rk in rank_map.items():
            p1[no]  = clamp01(bp1[rk]  * k1[rk])
            p12[no] = clamp01(bp12[rk] * k12[rk])
            p3[no]  = clamp01(bp3p[rk] * k3[rk])
        with st.expander("自動算出された確率（確認）", expanded=False):
            st.dataframe(pd.DataFrame({
                "番号": sorted(rank_map.keys()),
                "p1": [p1[i] for i in sorted(rank_map.keys())],
                "p2": [p12[i] for i in sorted(rank_map.keys())],
                "p3": [p3[i] for i in sorted(rank_map.keys())],
            }), use_container_width=True)
    else:
        st.info("N桁の並び（重複なし）で入力してください。")
else:
    st.caption("0〜1の小数で入力。未入力は0扱い。p2=連対率、p3=3着内率")
    df = pd.DataFrame({"番号": list(range(1,int(N)+1)), "p1":[0.0]*int(N), "p2":[0.0]*int(N), "p3":[0.0]*int(N)})
    edited = st.data_editor(df, num_rows="fixed", use_container_width=True, key="manual_probs",
                            column_config={"番号": st.column_config.NumberColumn(disabled=True),
                                           "p1": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
                                           "p2": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999),
                                           "p3": st.column_config.NumberColumn(format="%.4f", min_value=0.0, max_value=0.9999)})
    for r in edited.itertuples():
        i = int(r.番号)
        p1[i]  = clamp01(r.p1)
        p12[i] = clamp01(r.p2)
        p3[i]  = clamp01(r.p3 if r.p3>0 else max(r.p2, r.p1))

# ---------- 係数（ライン×脚質） ----------

def pair_coef(i:int,j:int)->float:
    li, lj = id_map.get(i), id_map.get(j)
    pi, pj = pos_map.get(i,1), pos_map.get(j,1)
    if li == lj:
        base = LINE_COEF[day]["adj"] if abs(pi-pj)==1 else LINE_COEF[day]["same"]
        style = style_factor_same_line(pi,pj)
    else:
        base = LINE_COEF[day]["diff"]
        style = STYLE_COEF_DIFF.get((style_map.get(i,"追"), style_map.get(j,"追")),
                                    STYLE_COEF_DIFF.get((style_map.get(j,"追"), style_map.get(i,"追")), 1.0))
    return base * style

# ---------- 計算 ----------

if st.button("計算", type="primary"):
    nums = list(range(1,int(N)+1))
    if not any(p12.get(i,0)>0 for i in nums):
        st.error("確率が未入力です（ヴェロビ並び or 確率表）")
    else:
        # 強さ（PL用）
        strength = {i: max(1e-9, odds_ratio(p1.get(i,0.0))) for i in nums}
        # 単勝・複勝
        tan = sorted(((i, p1.get(i,0.0)) for i in nums), key=lambda x: (-x[1], x[0]))[:1]
        fuku = sorted(((i, p12.get(i,0.0)) for i in nums), key=lambda x: (-x[1], x[0]))[:2]

        st.subheader("単勝（上位1）")
        for i,s in tan:
            st.write(f"**{i}** | p1={s:.4f}")

        st.subheader("複勝=連対率ベース（上位2）")
        for i,s in fuku:
            st.write(f"**{i}** | p2={s:.4f}")

        # 二車複
        pairs = []
        for i,j in combinations(nums,2):
            core = (p12.get(i,0.0) * p12.get(j,0.0))
            coef = pair_coef(i,j)
            s = core * coef
            # PL2で補正（ややp1寄り）：固定w=0.6
            q = pl2(strength, i, j)
            s = (q**0.6) * (core**0.4) * coef
            pairs.append(((min(i,j),max(i,j)), s))
        top2 = sorted(pairs, key=lambda x: (-x[1], x[0]))[:4]

        st.subheader("二車複（上位4）")
        for (a,b),s in top2:
            st.write(f"**{a}-{b}** | Score={s:.6f}")

        # ワイド
        def Qwide(i:int,j:int)->float:
            s = 0.0
            for k in nums:
                if k==i or k==j: continue
                s += pl3(strength, i,j,k)
            return s
        wides = []
        for i,j in combinations(nums,2):
            core = p3.get(i,0.0) * p3.get(j,0.0)
            q = Qwide(i,j)
            coef = pair_coef(i,j)
            s = (q**0.4) * (core**0.6) * coef
            wides.append(((min(i,j),max(i,j)), s))
        topW = sorted(wides, key=lambda x: (-x[1], x[0]))[:3]

        st.subheader("ワイド（上位3）")
        for (a,b),s in topW:
            st.write(f"**{a}-{b}** | Score={s:.6f}")

        # 三連複
        trios = []
        for i,j,k in combinations(nums,3):
            core = p3.get(i,0.0) * p3.get(j,0.0) * p3.get(k,0.0)
            q = pl3(strength, i,j,k)
            coef = (pair_coef(i,j)*pair_coef(i,k)*pair_coef(j,k))**(1.0/3.0)
            s = (q**0.6) * (core**0.4) * coef
            trios.append(((i,j,k), s))
        topT = sorted(trios, key=lambda x: (-x[1], x[0]))[:3]

        st.subheader("三連複（上位3）")
        for (a,b,c),s in topT:
            st.write(f"**{a}-{b}-{c}** | Score={s:.6f}")

        # 三連単（強さ上位から最大8頭で計算）
        pool = sorted(nums, key=lambda i: -strength[i])[:min(8, len(nums))]
        tri_list = []
        S_all = sum(strength[i] for i in pool)
        for i,j,k in permutations(pool,3):
            d1 = S_all
            d2 = d1 - strength[i]
            d3 = d2 - strength[j]
            if d2<=0 or d3<=0: continue
            p_ijk = (strength[i]/d1) * (strength[j]/d2) * (strength[k]/d3)
            # 順序ペアの係数は簡潔化=1.0（将来拡張可）
            tri_list.append(((i,j,k), p_ijk))
        topTT = sorted(tri_list, key=lambda x: (-x[1], x[0]))[:12]

        st.subheader("三連単（上位12｜強さ上位から計算）")
        for (a,b,c),s in topTT:
            st.write(f"**{a}-{b}-{c}** | Score={s:.6f}")

        # 参考予想補助（anchor）の反映は force/boost を簡潔運用
        policy = st.radio("参考予想補助（◎）の扱い", ["none","boost","force"], index=0, horizontal=True)
        if anchor is not None and policy != "none":
            st.info("※ この簡潔版ではリスト順序の再計算は行わず、force/boostの実買い配分でご使用ください。必要なら完全版で再計算対応します。")

st.markdown("---")
st.caption("これは娯楽としてのものであり、結果については一切の責任を負いません。")

