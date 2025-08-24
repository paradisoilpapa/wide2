# app.py
# 7車立て 二車複4点 + 三連複 + ワイド（p1+p12/p3・固定係数・脚質まとめ入力・ヴェロビ並び一括） v5
# pip install streamlit

from __future__ import annotations
from dataclasses import dataclass
import itertools, re, math
import streamlit as st

# ================== 通算の基準値（％→小数） ==================
BASE_P12  = {1:0.382, 2:0.307, 3:0.278, 4:0.321, 5:0.292, 6:0.279, 7:0.171}  # 連対率
DEFAULT_P1_RATIO = 0.481  # rank1: 0.184/0.382 ≈ 0.481
BASE_P1 = {r: max(min(BASE_P12[r]*DEFAULT_P1_RATIO, 0.95), 0.0001) for r in range(1,8)}  # 1着率の比例近似

# 三連複/ワイド用に3着内率も（分析アプリと同じ値）
BASE_PIN3 = {1:0.519, 2:0.476, 3:0.472, 4:0.448, 5:0.382, 6:0.358, 7:0.322}

# ================== 日別係数 ==================
K12 = {
    "初日":   {1:0.9240837696, 2:1.4169381107, 3:1.0575539568, 4:0.8785046729, 5:0.6849315068, 6:1.0967741935, 7:0.7543859649},
    "2日目":  {1:1.1361256545, 2:0.6416938111, 3:0.8992805755, 4:1.1059190031, 5:1.2157534247, 6:0.9892473118, 7:1.2339181287},
    "最終日": {1:0.9240837696, 2:0.8306188925, 3:1.0575539568, 4:1.0373831776, 5:1.2089041096, 6:0.8422939068, 7:1.0526315789},
}
K1 = {
    "初日":   {1:0.766, 2:1.282, 3:1.106, 4:1.000, 5:0.746, 6:1.110, 7:0.728},  # 実測
    "2日目":  {},
    "最終日": {},
}
for r in range(1,8):  # p1日別が未整備のため暫定でK12流用（差し替え可）
    K1["2日目"][r]  = K12["2日目"][r]
    K1["最終日"][r] = K12["最終日"][r]

# 三連複/ワイド用の3内率係数（分析アプリ準拠）
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
    if (a,b)==(1,2): return 1.15
    if (a,b)==(2,3): return 1.08
    if (a,b)==(1,3): return 1.03
    return 1.00
STYLE_COEF_DIFF = {  # 別線のみ
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

def parse_verovi_order(s: str) -> list[int]:
    """ ヴェロビ評価順（車番の並び）を1行で受け取り、長さ7・重複なしなら[car,...]を返す """
    if not s: return []
    t = re.sub(r"[^\d]", "", s)  # 数字以外削除
    if len(t) != 7: return []
    order = [int(ch) for ch in t]
    if sorted(order) != list(range(1,8)): return []
    return order

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

# ================== 当日p1/p12/p3 と PL ==================
def day_p1(rank:int, day:str)->float:
    return max(min(BASE_P1[rank]  * K1[day][rank],  0.9999), 0.0001)
def day_p12(rank:int, day:str)->float:
    return max(min(BASE_P12[rank] * K12[day][rank], 0.9999), 0.0001)
def day_p3(rank:int, day:str)->float:
    return max(min(BASE_PIN3[rank]* K3[day][rank], 0.9999), 0.0001)

def pl_joint_prob_pair(strength:dict[int,float], i:int, j:int)->float:
    """PLで上位2が{i,j}になる確率（順序入替を和）。strengthは正の“強さ”（ここでは当日p1）。"""
    S = sum(strength.values())
    if S <= 0: return 0.0
    pi, pj = strength[i], strength[j]
    out = 0.0
    for a,b in [(i,j),(j,i)]:
        sa, sb = strength[a], strength[b]
        out += (sa/S) * (sb/(S - sa)) if S-sa>0 else 0.0
    return out

def pl_joint_prob_trio(strength:dict[int,float], i:int, j:int, k:int)->float:
    """PLで上位3が{i,j,k}になる確率（全順序6通りの和）。"""
    S = sum(strength.values())
    if S <= 0: return 0.0
    out = 0.0
    for a,b,c in [(i,j,k),(i,k,j),(j,i,k),(j,k,i),(k,i,j),(k,j,i)]:
        sa, sb, sc = strength[a], strength[b], strength[c]
        if S-sa>0 and S-sa-sb>0:
            out += (sa/S) * (sb/(S-sa)) * (sc/(S-sa-sb))
    return out

# ================== スコア計算（二車複・三連複・ワイド） ==================
def score_pairs_and_pick(runners:list[Runner], day:str, w_pair:float, k:int=4):
    """二車複：Score = [(PL2)^w × (p12_i p12_j)^(1-w)] × L × R"""
    strength = {r.no: day_p1(r.rank, day) for r in runners}
    cand = []
    for a,b in itertools.combinations(runners,2):
        qpl2 = pl_joint_prob_pair(strength, a.no, b.no)
        s12  = day_p12(a.rank, day) * day_p12(b.rank, day)
        core = (qpl2**w_pair) * (s12**(1.0-w_pair))
        L = line_factor(a,b,day)
        R = style_factor(a,b)
        s = core * L * R
        cand.append(((min(a.no,b.no), max(a.no,b.no)), s,
                     a.line==b.line and abs(a.pos-b.pos)==1, {a.no,b.no}))
    cand.sort(key=lambda x: x[1], reverse=True)

    # 制約: ①同一ライン隣接を最低1点 ②同一選手は最大2点 ③計k点
    selected, cnt = [], {}
    for pair,s,adj,inv in cand:  # 隣接を1点確保
        if not adj: continue
        i,j = list(inv)
        if cnt.get(i,0)>=2 or cnt.get(j,0)>=2: continue
        selected.append((pair,s,adj)); cnt[i]=cnt.get(i,0)+1; cnt[j]=cnt.get(j,0)+1
        break
    for pair,s,adj,inv in cand:  # 残り充足
        if len(selected)>=k: break
        i,j = list(inv)
        if cnt.get(i,0)>=2 or cnt.get(j,0)>=2: continue
        if (pair,s,adj) in selected: continue
        selected.append((pair,s,adj)); cnt[i]=cnt.get(i,0)+1; cnt[j]=cnt.get(j,0)+1
    return selected[:k], cand

def score_trios_and_pick(runners:list[Runner], day:str, w_trio:float, m:int=3, axis_pair:set[int]|None=None):
    """三連複：Score = [(PL3)^w × (p3_i p3_j p3_k)^(1-w)] × (L3 × R3)"""
    strength = {r.no: day_p1(r.rank, day) for r in runners}
    def p3(rank:int): return day_p3(rank, day)
    by_no = {r.no:r for r in runners}

    cand = []
    for a,b,c in itertools.combinations(sorted(by_no.keys()),3):
        if axis_pair and not axis_pair.issubset({a,b,c}):
            continue
        ra, rb, rc = by_no[a], by_no[b], by_no[c]
        qpl3 = pl_joint_prob_trio(strength, a,b,c)
        s3   = p3(ra.rank) * p3(rb.rank) * p3(rc.rank)
        core = (qpl3**w_trio) * (s3**(1.0-w_trio))
        # 3者のライン/脚質係数（幾何平均）
        L12 = line_factor(ra,rb,day); L13 = line_factor(ra,rc,day); L23 = line_factor(rb,rc,day)
        R12 = style_factor(ra,rb);    R13 = style_factor(ra,rc);    R23 = style_factor(rb,rc)
        L3  = (L12*L13*L23)**(1.0/3.0)
        R3  = (R12*R13*R23)**(1.0/3.0)
        s   = core * L3 * R3
        cand.append(((a,b,c), s, {"qpl3":qpl3, "p3prod":s3, "L3":L3, "R3":R3}))
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[:m], cand

def score_wide_and_pick(runners:list[Runner], day:str, w_wide:float, n:int=3):
    """ワイド：Score = [(Qwide)^w × (p3_i p3_j)^(1-w)] × L × R
       Qwide(i,j) = Σ_{k≠i,j} PL3(i,j,k)  （i,jが“上位3内”に入る確率）"""
    strength = {r.no: day_p1(r.rank, day) for r in runners}
    by_no = {r.no:r for r in runners}
    # 事前に全PL3を必要に応じ算出して合算
    def Qwide(i:int,j:int)->float:
        s = 0.0
        for k in by_no.keys():
            if k==i or k==j: continue
            s += pl_joint_prob_trio(strength, i,j,k)
        return s

    cand = []
    for a,b in itertools.combinations(sorted(by_no.keys()),2):
        ra, rb = by_no[a], by_no[b]
        q = Qwide(a,b)
        s3pair = day_p3(ra.rank, day) * day_p3(rb.rank, day)
        core = (q**w_wide) * (s3pair**(1.0-w_wide))
        L = line_factor(ra,rb,day)
        R = style_factor(ra,rb)
        s = core * L * R
        cand.append(((a,b), s, {"Qwide":q, "p3prod":s3pair, "L":L, "R":R}))
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[:n], cand

# ================== UI ==================
st.set_page_config(page_title="ヴェロビ 二車複＋三連複＋ワイド（p1+p12/p3）", layout="wide")
st.title("ヴェロビ：二車複 4点 ＋ 三連複 ＋ ワイド（p1+p12/p3・固定係数・ライン考慮）")

day = st.selectbox("開催日", ["初日","2日目","最終日"], index=0)

c_w1, c_w3, c_ww = st.columns(3)
with c_w1:
    w_pair = st.slider("二車複 w_pair（PL2 : p12）", 0.0, 1.0, 0.7, 0.05)
with c_w3:
    w_trio = st.slider("三連複 w_trio（PL3 : p3）", 0.0, 1.0, 0.6, 0.05)
with c_ww:
    w_wide = st.slider("ワイド w_wide（Qwide : p3）", 0.0, 1.0, 0.4, 0.05)

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
dup = (set(list_nige) & set(list_ryo)) | (set(list_nige) & set(list_tsu)) | (set(list_ryo) & set(list_tsu))
if dup:
    st.error(f"脚質の重複指定: {sorted(dup)} が複数の欄に入っています。どれか一方にしてください。")
default_unassigned = st.radio("未指定車番の既定脚質", options=["追", "両", "逃"], index=0, horizontal=True)

# ---- ヴェロビ評価順（車番並びを1行） or 手動 ----
st.subheader("評価順位の入力")
method = st.radio("入力方法を選択", ["ヴェロビ評価順（車番並びを1行）", "手動で各選手に入力"], horizontal=True, index=0)

rank_map = {}
if method.startswith("ヴェロビ"):
    vorder_str = st.text_input("ヴェロビ評価順（例: 3142576 や 3 1 4 2 5 7 6）", value="")
    # パース
    def parse_verovi_order_local(s: str) -> list[int]:
        if not s: return []
        t = re.sub(r"[^\d]", "", s)
        if len(t) != 7: return []
        order = [int(ch) for ch in t]
        if sorted(order) != list(range(1,8)): return []
        return order
    vorder = parse_verovi_order_local(vorder_str)
    if vorder:
        rank_map = {car: i+1 for i,car in enumerate(vorder)}
    else:
        st.info("7桁・重複なしで 1〜7 を並べてください。未入力や不正な場合は下の手動欄を使えます。")

# ---- 各選手レコード生成
cols = st.columns(7, gap="small")
runners = []
for i in range(7):
    with cols[i]:
        no = i+1
        st.markdown(f"**{no}番**")
        # 脚質の割当（まとめ入力から）
        if no in list_nige:   style = "逃"
        elif no in list_ryo:  style = "両"
        elif no in list_tsu:  style = "追"
        else:                 style = default_unassigned
        st.caption(f"脚質: {style}")
        # 順位
        if rank_map:
            rank = rank_map[no]; st.text(f"評価順位: {rank}")
        else:
            rank = st.number_input("評価順位", 1, 7, value=min(no,7), key=f"rank{no}")
        runners.append(Runner(no=no, rank=int(rank), line=id_map[no], pos=pos_map[no], style=style))

st.markdown("---")
if st.button("買い目を選定（二車複＋三連複＋ワイド）"):
    if dup:
        st.error("脚質の重複指定を解消してください。")
    else:
        # 二車複
        picks2, cand2 = score_pairs_and_pick(runners, day, w_pair, k=4)
        if not picks2:
            st.error("二車複の候補が出ません。入力を確認してください。")
        else:
            st.subheader("二車複：選定結果（上位4点）")
            for i,(pair,score,adj) in enumerate(picks2,1):
                a,b = pair
                st.write(f"**{i}. {a}-{b}** | Score: {score:.6f} {'（同一ライン隣接）' if adj else ''}")

            # 三連複（デフォは二車複1位ペアを“軸”）
            st.subheader("三連複：選定（デフォ=二車複1位ペアを軸）")
            axis = st.checkbox("三連複は“二車複1位ペア”を軸に限定する", value=True)
            m = st.slider("三連複の点数（上位）", 1, 6, 3, 1)
            axis_pair = set(picks2[0][0]) if axis else None
            top3, _ = score_trios_and_pick(runners, day, w_trio, m=m, axis_pair=axis_pair)
            if not top3:
                st.info("条件を満たす三連複がありません。軸を外すか点数を増やしてみてください。")
            else:
                for i,(tri,score,meta) in enumerate(top3,1):
                    a,b,c = tri
                    st.write(f"**{i}. {a}-{b}-{c}** | Score: {score:.6f} "
                             f"(PL3={meta['qpl3']:.5f}, p3prod={meta['p3prod']:.5f}, L3={meta['L3']:.2f}, R3={meta['R3']:.2f})")

            # ワイド
            st.subheader("ワイド：選定（p3寄りスコア）")
            n_wide = st.slider("ワイドの点数（上位）", 1, 5, 3, 1)
            topW, _allW = score_wide_and_pick(runners, day, w_wide, n=n_wide)
            if not topW:
                st.info("ワイド候補が出ません。入力を確認してください。")
            else:
                for i,(pair,score,meta) in enumerate(topW,1):
                    a,b = pair
                    st.write(f"**{i}. {a}-{b}** | Score: {score:.6f} "
                             f"(Qwide={meta['Qwide']:.5f}, p3prod={meta['p3prod']:.5f}, L={meta['L']:.2f}, R={meta['R']:.2f})")

            # オプション：オッズを入れてトリガミ回避チェック
            with st.expander("任意：オッズを入れて“Σ(1/O) ≤ 1”チェック（ワイド／三連複）", expanded=False):
                st.caption("等額買い時、Σ(1/O) ≤ 1 なら当たり時に最低でもトントン以上。低オッズが多いなら本数を絞る。")
                if topW:
                    st.markdown("**ワイド**")
                    inv_sum_w = 0.0
                    for pair,_,_ in topW:
                        key = f"oddsW_{pair}"
                        O = st.number_input(f"{pair} のオッズ（ワイド）", min_value=0.0, value=0.0, step=0.1, key=key)
                        inv_sum_w += (1.0/O) if O>0 else 0.0
                    st.write(f"Σ(1/O)_wide = **{inv_sum_w:.3f}** → {'✅ トリガミ回避可' if inv_sum_w<=1.0 else '⚠️ 要削減/配分'}")
                if top3:
                    st.markdown("**三連複**")
                    inv_sum_t = 0.0
                    for tri,_,_ in top3:
                        key = f"oddsT_{tri}"
                        O = st.number_input(f"{tri} のオッズ（三連複）", min_value=0.0, value=0.0, step=0.1, key=key)
                        inv_sum_t += (1.0/O) if O>0 else 0.0
                    st.write(f"Σ(1/O)_trio = **{inv_sum_t:.3f}** → {'✅ トリガミ回避可' if inv_sum_t<=1.0 else '⚠️ 要削減/配分'}")

with st.expander("参照：固定係数（検算用）", expanded=False):
    st.write("K1（1着率係数）", K1)
    st.write("K12（連対率係数）", K12)
    st.write("K3（3着内率係数）", K3)

st.caption("""
二車複 Score = [(PL2)^w_pair × (p12_i p12_j)^(1-w_pair)] × ライン係数 × 役割/脚質係数。
三連複 Score = [(PL3)^w_trio × (p3_i p3_j p3_k)^(1-w_trio)] × {ペア係数(ライン/脚質)の幾何平均}。
ワイド   Score = [(Qwide)^w_wide × (p3_i p3_j)^(1-w_wide)] × ライン係数 × 役割/脚質係数
（Qwide(i,j) = Σ_k PL3(i,j,k)：iとjが“上位3内”に含まれる確率）
・脚質は「逃/両/追」に車番を列挙（重複は赤エラー、未指定は既定で補完）。
・順位は「ヴェロビ評価順」を1行貼付で自動付与。未入力/不正は手動入力に切替。
・オッズ入力時は Σ(1/O)≤1 の判定でトリガミ回避、配分は別アプリ/手計算で調整してください。
""")
