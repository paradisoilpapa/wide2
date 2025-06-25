import streamlit as st
from itertools import combinations

# タイトルと説明
st.title("7車立て競輪レース　三連複・二車複オッズ計算アプリ")
st.write(
    "◎（本命）とヒモを選択すると、自動的に三連複および二車複の買い目が生成されます。各買い目のオッズを入力すると、合成オッズが計算され、購入基準の判定が表示されます。"
)

# 本命（◎）の選択
horses = list(range(1, 8))  # 1～7車
main = st.selectbox("◎ 本命", horses)

# ヒモ選択（最大4車まで）
options = [h for h in horses if h != main]
himo = st.multiselect("ヒモ (最大4車)", options)

# ヒモ選択数のバリデーション
if len(himo) > 4:
    st.error("ヒモは最大4車まで選択してください。")
    st.stop()  # これ以上の処理を中断

# 買い目の組み合わせ生成
himo_sorted = sorted(himo)
# 三連複: 本命＋ヒモから2車選ぶ組み合わせ
trifecta_combs = [tuple(sorted((main,)+comb)) for comb in combinations(himo_sorted, 2)] if len(himo_sorted) >= 2 else []
# 二車複: 本命＋ヒモ1車ずつの組み合わせ
pair_combs = [tuple(sorted((main, h))) for h in himo_sorted] if len(himo_sorted) >= 1 else []

# 三連複セクション
st.subheader("三連複")
if trifecta_combs:
    # テーブルヘッダー
    header_cols = st.columns([1, 1])
    header_cols[0].write("**買い目**")
    header_cols[1].write("**オッズ**")
    trif_odds = []  # オッズ入力値を格納するリスト
    for comb in trifecta_combs:
        comb_str = "".join(map(str, comb))  # 組み合わせを文字列に（例: (1,2,3) -> "123"）
        cols = st.columns([1, 1])
        cols[0].write(comb_str)
        # オッズ入力欄（ラベルは非表示）
        odd_value = cols[1].number_input(
            f"{comb_str} のオッズ", min_value=0.0, value=0.0, step=0.1, 
            label_visibility="hidden", key=f"odds_trif_{comb_str}"
        )
        trif_odds.append(odd_value)
    # 合成オッズ計算と表示
    # 合成オッズ = 1 ÷ (Σ(1/各オッズ))  【各オッズの逆数の合計の逆数】
    if all(o > 0 for o in trif_odds) and len(trif_odds) > 0:
        combined_odds = 1.0 / sum((1.0/o) for o in trif_odds)
        st.write(f"合成オッズ: {combined_odds:.2f}倍")
        # 基準値（3.0倍）との比較
        if combined_odds < 3.0:
            st.warning("購入見送り")
        else:
            st.success("購入基準クリア")
    else:
        st.write("合成オッズ: -")
        st.info("すべてのオッズを入力すると合成オッズが表示されます。")
else:
    # ヒモが2車未満の場合のメッセージ
    st.info("三連複の買い目を生成するには、ヒモを2車以上選択してください。")

# 二車複セクション
st.subheader("二車複")
if pair_combs:
    # テーブルヘッダー
    header_cols = st.columns([1, 1])
    header_cols[0].write("**買い目**")
    header_cols[1].write("**オッズ**")
    pair_odds = []
    for comb in pair_combs:
        comb_str = "".join(map(str, comb))  # 例: (1,2) -> "12"
        cols = st.columns([1, 1])
        cols[0].write(comb_str)
        odd_value = cols[1].number_input(
            f"{comb_str} のオッズ", min_value=0.0, value=0.0, step=0.1, 
            label_visibility="hidden", key=f"odds_pair_{comb_str}"
        )
        pair_odds.append(odd_value)
    # 合成オッズの計算と表示（二車複）
    if all(o > 0 for o in pair_odds) and len(pair_odds) > 0:
        combined_odds = 1.0 / sum((1.0/o) for o in pair_odds)
        st.write(f"合成オッズ: {combined_odds:.2f}倍")
        # 基準値（1.5倍）との比較
        if combined_odds < 1.5:
            st.warning("購入見送り")
        else:
            st.success("購入基準クリア")
    else:
        st.write("合成オッズ: -")
        st.info("すべてのオッズを入力すると合成オッズが表示されます。")
else:
    # ヒモ未選択の場合のメッセージ
    st.info("二車複の買い目を生成するには、ヒモを1車以上選択してください。")
