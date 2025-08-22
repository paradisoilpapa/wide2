# app.py
# 7車立て 二車複4点自動選定（通算＋日別係数固定・オッズ不要）
# pip install streamlit

from __future__ import annotations
from dataclasses import dataclass
import itertools, re
import streamlit as st

# ================== 通算の基準値（％→小数） ==================
BASE_P12  = {1:0.382, 2:0.307, 3:0.278, 4:0.321, 5:0.292, 6:0.279, 7:0.171}  # 連対率
BASE_PIN3 = {1:0.519, 2:0.476, 3:0.472, 4:0.448, 5:0.382, 6:0.358, 7:0.322}  # 3着内率

# ================== 日別係数（あなたの実測から算出） ==================
# k12(day,rank) = (日別連対率) / (通算連対率)
K12 = {
    "初日":   {1:0.9240837696, 2:1.4169381107, 3:1.0575539568, 4:0.8785046729, 5:0.6849315068, 6:1.0967741935, 7:0.7543859649},
    "2日目":  {1:1.1361256545, 2:0.6416938111, 3:0.8992805755, 4:1.1059190031, 5:1.2157534247, 6:0.9892473118, 7:1.2339181287},
    "最終日": {1:0.9240837696, 2:0.8306188925, 3:1.0575539568, 4:1.0373831776, 5:1.2089041096, 6:0.8422939068, 7:1.0526315789},
}
# k3(day,rank) = (日別3着内率) / (通算3着内率)
K3 = {
    "初日":   {1:0.9749518304, 2:1.2857142857, 3:1.1207627119, 4:0.9196428571, 5:0.8010471204, 6:1.0502793296, 7:0.8043478261},
    "2日目":  {1:1.1156069364, 2:0.7457983193, 3:0.8368644068, 4:0.9687500000, 5:1.1020942408, 6:1.0279329609, 7:1.1428571429},
    "最終日": {1:0.8689788054, 2:0.9054621849, 3:1.0381355932, 4:1.1808035714, 5:1.1806282723, 6:0.8770949721, 7:1.1180124224},
}

# ================== ライン・相性係数 ==================
LINE_COEF = {  # 同一ライン：隣接/非隣接、別ライン
    "初日":   {"adj":1.35, "same":1.10, "diff":1.00},
    "2日目":  {"adj":1.20, "same":1.05, "diff":1.00},
    "最終日": {"adj":1.25, "same":1.05, "diff":1.00},
}

def role_name(pos:int)->str:
    return {1:"先行", 2:"番手", 3:"三番手"}.get(pos, "その他")

def style_factor_same_line(pos_a:int, pos_b:int)->float:
    a,b = sorted([pos_a,pos_b])
    if (a,b)==(1,2): return 1.15  # 先行-番手
    if (a,b)==(2,3): return 1.08  # 番手-三番手
    if (a,b)==(1,3): return 1.03  # 先行-三番手
    return 1.00

STYLE_COEF_DIFF = {
    ("逃","逃"):0.90, ("両","逃"):0.95,
    ("追","追"):1.00, ("両","追"):1.00, ("追","両"):1.00,
    ("両","両"):1.00, ("逃","追"):1.00, ("追","逃"):1.00,
}

# ================== 型 ==================
@dataclass
class Runner:
    no:int
    rank:int          # ヴェロビ評価順位(1..7)
    line:str          # A,B,C...
    pos:int           # 1=先行/2=番手/3=三番手…
    style:str         # '逃','両','追'

# ================== 補助 ==================
def parse_line_pattern(pattern:str):
    """ '123 45 6 7' → 車番→ラインID, 車番→ライン内位置 """
    pattern = re.sub(r"[^\d\s]", "", pattern).strip()
    groups = [g for g in pattern.split() if g]
    id_map, pos_map = {}, {}
    for gi, g in enumerate(groups):
        lid = chr(ord('A') + gi)
        for idx, ch in enumerate(g):
            n = int(ch)
            id_map[n]  = lid
            pos_map[n] = idx+1
    used = set(id_map.keys())
    for n in range(1,8):
        if n not in used:
            lid = chr(ord('A') + len(groups))
            id_map[n]  = lid
            pos_map[n] = 1
            groups.append(str(n))
    return id_map, pos_map

def line_factor(a:Runner, b:Runner, day:str)->float:
    if a.line == b.line:
        if abs(a.pos - b.pos) == 1:
            return LINE_COEF[day]["adj"]
        return LINE_COEF[day]["same"]
    return LINE_COEF[day]["diff"]

def style_factor(a:Runner, b:Runner)->float:
    if a.line == b.line:
        return style_factor_same_line(a.pos, b.pos)
    key = (a.style, b.style)
    return STYLE_COEF_DIFF.get(key, STYLE_COEF_DIFF.get((key[1],key[0]), 1.00))

# ================== スコア計算 ==================
def score_pair(a:Runner, b:Runner, day:str, w:float)->float:
    # 連対×3内のブレンド（w=0.7 推奨）
    p12_i = max(min(BASE_P12[a.rank]  * K12[day][a.rank],  0.9999), 0.0001)
    p12_j = max(min(BASE_P12[b.rank]  * K12[day][b.rank],  0.9999), 0.0001)
    p3_i  = max(min(BASE_PIN3[a.rank] * K3[day][a.rank],   0.9999), 0.0001)
    p3_j  = max(min(BASE_PIN3[b.rank] * K3[day][b.rank],   0.9999), 0.0001)
    L = line_factor(a,b,day)
    R = style_factor(a,b)
    s12 = (p12_i * p12_j)
    s3  = (p3_i  * p3_j)
    return (s12**w) * (s3**(1.0-w)) * L * R

def pick_pairs(runners:list[Runner], day:str, w:float, k:int=4):
    cand = []
    for a,b in itertools.combinations(runners,2):
        s = score_pair(a,b,day,w)
        cand.append(((min(a.no,b.no), max(a.no,b.no)), s,
                     a.line==b.line and abs(a.pos-b.pos)==1, {a.no,b.no}))
    cand.sort(key=lambda x: x[1], reverse=True)

    # 制約: ①同一ライン隣接を最低1点 ②同一選手は最大2点 ③計k点
    selected, cnt = [], {}
    # 隣接を1点確保
    for pair,s,adj,inv in cand:
        if not adj: continue
        i,j = list(inv)
        if cnt.get(i,0)>=2 or cnt.get(j,0)>=2: continue
        selected.append((pair,s,adj)); cnt[i]=cnt.get(i,0)+1; cnt[j]=cnt.get(j,0)+1
        break
    # 残り充足
    for pair,s,adj,inv in cand:
        if len(selected)>=k: break
        i,j = list(inv)
        if cnt.get(i,0)>=2 or cnt.get(j,0)>=2: continue
        if (pair,s,adj) in selected: continue
        selected.append((pair,s,adj)); cnt[i]=cnt.get(i,0)+1; cnt[j]=cnt.get(j,0)+1
    return selected[:k]

# ================== UI ==================
st.set_page_config(page_title="ヴェロビ二車複4点（固定係数）", layout="wide")
st.title("ヴェロビ：二車複 4点（オッズ不要・固定係数・ライン考慮）")

day = st.selectbox("開催日", ["初日","2日目","最終日"], index=0)
w   = st.slider("連対重視ブレンド係数 w（0〜1）", 0.0, 1.0, 0.7, 0.05)

st.subheader("ライン入力（例：123 45 6 7）")
pattern = st.text_input("ラインパターン", value="123 45 6 7")
id_map, pos_map = parse_line_pattern(pattern)

st.subheader("各選手の入力（評価順位・脚質）")
cols = st.columns(7, gap="small")
runners = []
for i in range(7):
    with cols[i]:
        no = i+1
        st.markdown(f"**{no}番**")
        rank = st.number_input("評価順位", 1, 7, value=min(no,7), key=f"rank{no}")
        style = st.selectbox("脚質", ['逃','両','追'], index=0 if no in [1,6] else 2, key=f"style{no}")
        runners.append(Runner(no=no, rank=rank, line=id_map[no], pos=pos_map[no], style=style))

st.markdown("---")
if st.button("４点を選定"):
    picks = pick_pairs(runners, day, w, k=4)
    if not picks:
        st.error("候補が出ません。入力を確認してください。")
    else:
        st.subheader("選定結果（上位4点）")
        for i,(pair,score,adj) in enumerate(picks,1):
            a,b = pair
            st.write(f"**{i}. {a}-{b}** | スコア: {score:.6f} {'（同一ライン隣接）' if adj else ''}")

with st.expander("参照：固定係数（検算用）", expanded=False):
    st.write("K12（連対率係数）", K12)
    st.write("K3（3着内率係数）", K3)
