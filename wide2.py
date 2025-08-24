# app.py
# 7車立て 二車複4点自動選定（p1+p12・固定係数・脚質まとめ入力） v3
# pip install streamlit

from __future__ import annotations
from dataclasses import dataclass
import itertools, re
import streamlit as st

# ================== 通算の基準値（％→小数） ==================
BASE_P12  = {1:0.382, 2:0.307, 3:0.278, 4:0.321, 5:0.292, 6:0.279, 7:0.171}  # 連対率

# 通算 p1（1着率）は p12 に比例させる簡易近似（rank1の 0.184/0.382 ≈ 0.481 を採用）
DEFAULT_P1_RATIO = 0.481
BASE_P1 = {r: max(min(BASE_P12[r]*DEFAULT_P1_RATIO, 0.95), 0.0001) for r in range(1,8)}

# ================== 日別係数（あなたの実測から算出） ==================
# k12(day,rank) = (日別連対率) / (通算連対率)
K12 = {
    "初日":   {1:0.9240837696, 2:1.4169381107, 3:1.0575539568, 4:0.8785046729, 5:0.6849315068, 6:1.0967741935, 7:0.7543859649},
    "2日目":  {1:1.1361256545, 2:0.6416938111, 3:0.8992805755, 4:1.1059190031, 5:1.2157534247, 6:0.9892473118, 7:1.2339181287},
    "最終日": {1:0.9240837696, 2:0.8306188925, 3:1.0575539568, 4:1.0373831776, 5:1.2089041096, 6:0.8422939068, 7:1.0526315789},
}
# k1(day,rank) = (日別1着率) / (通算1着率)
# 初日は実測係数、2日目/最終日は p1日別が未整備なので K12 を暫定流用（差し替え可能）
K1 = {
    "初日":   {1:0.766, 2:1.282, 3:1.106, 4:1.000, 5:0.746, 6:1.110, 7:0.728},
    "2日目":  {},
    "最終日": {},
}
for r in range(1,8):
    K1["2日目"][r]  = K12["2日目"][r]
    K1["最終日"][r] = K12["最終日"][r]

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

STYLE_COEF_DIFF = {  # 別線のみ適用（同線は“位置”で評価）
    ("逃","逃"):0.90, ("両","逃"):0.95, ("逃","両"):0.95,
    ("追","追"):1.00, ("両","追"):1.00, ("追","両"):1.00,
    ("逃","追"):1.00, ("追","逃"):1.00, ("両","両"):1.00,
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
    return STYLE_COEF_DIFF.get(key, 1.00)

def parse_car_list(s: str, n_max: int = 7) -> list[int]:
    """ '1 6' / '1,6' / '１・６' 等を [1,6] に """
    if not s: return []
    t = s.replace("、", " ").replace(",", " ").replace("　", " ") \
         .replace("・", " ").replace("/", " ").replace("-", " ")
    out = []
    for tok in t.split():
        tok = "".join(ch for ch in tok if ch.isdigit())
        if not tok: continue
        try:
            v = int(tok)
            if 1 <= v <= n_max:
                out.append(v)
        except:
            pass
    return out

# ================== 当日p1/p12とPL近似 ==================
def day_p1(rank:int, day:str)->float:
    return max(min(BASE_P1[rank]  * K1[day][rank],  0.9999), 0.0001)

def day_p12(rank:int, day:str)->float:
    return max(min(BASE_P12[rank] * K12[day][rank], 0.9999), 0.0001)

def pl_joint_prob(p1_day_by_runner:dict[int,float], i:int, j:int)->float:
    """ Plackett–Luceの上位2位“同時”確率の対称近似。
        入力は“当日の p1”を全員分集め、内部で正規化（sum=1）。 """
    total = sum(p1_day_by_runner.values())
    if total <= 0: return 0.0
    # 正規化（数値安定化のためクリップ）
    pi = max(min(p1_day_by_runner[i]/total, 0.9999), 0.0001)
    pj = max(min(p1_day_by_runner[j]/total, 0.9999), 0.0001)
    # q_ij ≈ p(i1st,j2nd)+p(j1st,i2nd) = p_i p_j (1/(1-p_i)+1/(1-p_j))
    return (pi*pj) * ((1.0/(1.0-pi)) + (1.0/(1.0-pj)))

# ================== スコア計算（p1×PL と p12 のCobb–Douglasブレンド） ==================
def pick_pairs(runners:list[Runner], day:str, w:float, k:int=4):
    # 全員の当日p1（正規化用）
    p1_day_map = {r.no: day_p1(r.rank, day) for r in runners}

    cand = []
    for a,b in itertools.combinations(runners,2):
        # 核：PL近似のjoint & 連対積
        qpl = pl_joint_prob(p1_day_map, a.no, b.no)
        s12 = day_p12(a.rank, day) * day_p12(b.rank, day)
        core = (qpl**w) * (s12**(1.0-w))
        # ライン・役割/脚質
        L = line_factor(a,b,day)
        R = style_factor(a,b)
        s = core * L * R
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
    return selected[:k], cand

# ================== UI ==================
st.set_page_config(page_title="ヴェロビ二車複4点（p1+p12・固定係数）", layout="wide")
st.title("ヴェロビ：二車複 4点（p1+p12・固定係数・ライン考慮）")

day = st.selectbox("開催日", ["初日","2日目","最終日"], index=0)
w   = st.slider("ブレンド係数 w（PL:p12）", 0.0, 1.0, 0.7, 0.05)

st.subheader("ライン入力（例：123 45 6 7）")
pattern = st.text_input("ラインパターン", value="123 45 6 7")
id_map, pos_map = parse_line_pattern(pattern)

# ---- 脚質：車番まとめ入力（ミス削減） ----
st.subheader("脚質入力（車番をまとめて）")
c1, c2, c3 = st.columns(3)
raw_nige = c1.text_input("逃（例: 1 6）", value="")
raw_ryo  = c2.text_input("両（例: 2 3）", value="")
raw_tsu  = c3.text_input("追（例: 4 5 7）", value="")

list_nige = parse_car_list(raw_nige, 7)
list_ryo  = parse_car_list(raw_ryo, 7)
list_tsu  = parse_car_list(raw_tsu, 7)

dup = set(list_nige) & set(list_ryo) | set(list_nige) & set(list_tsu) | set(list_ryo) & set(list_tsu)
if dup:
    st.error(f"脚質の重複指定: {sorted(dup)} が複数の欄に入っています。どれか一方にしてください。")

default_unassigned = st.radio("未指定車番の既定脚質", options=["追", "両", "逃"], index=0, horizontal=True)

st.subheader("各選手の入力（評価順位のみ）")
cols = st.columns(7, gap="small")
runners = []
for i in range(7):
    with cols[i]:
        no = i+1
        st.markdown(f"**{no}番**")
        rank = st.number_input("評価順位", 1, 7, value=min(no,7), key=f"rank{no}")

        # まとめ入力から脚質を自動付与（見た目だけ表示）
        if no in list_nige:   style = "逃"
        elif no in list_ryo:  style = "両"
        elif no in list_tsu:  style = "追"
        else:                 style = default_unassigned
        st.caption(f"脚質: {style}")

        runners.append(Runner(no=no, rank=rank, line=id_map[no], pos=pos_map[no], style=style))

st.markdown("---")
if st.button("４点を選定"):
    if dup:
        st.error("脚質の重複指定を解消してください。")
    else:
        picks, cand = pick_pairs(runners, day, w, k=4)
        if not picks:
            st.error("候補が出ません。入力を確認してください。")
        else:
            st.subheader("選定結果（上位4点）")
            for i,(pair,score,adj) in enumerate(picks,1):
                a,b = pair
                st.write(f"**{i}. {a}-{b}** | スコア: {score:.6f} {'（同一ライン隣接）' if adj else ''}")

with st.expander("参照：固定係数（検算用）", expanded=False):
    st.write("K1（1着率係数）", K1)
    st.write("K12（連対率係数）", K12)

st.caption("""
Score = [(PL由来の上位2同時確率)^w × (連対積)^(1-w)] × ライン係数 × 役割/脚質係数。
・同一ラインは“位置”で評価（先行-番手1.15 等）、別線は“脚質”で微調整（逃-逃0.90 等）。
・通算p1は p12の比例近似（既定0.481倍）。p1日別が揃い次第、K1を差し替えるだけで反映されます。
・制約：同一ライン隣接を最低1点／同一選手は最大2点／合計4点。
""")
