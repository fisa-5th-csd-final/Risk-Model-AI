import pandas as pd
import numpy as np
import os
import psutil
import gc
import threading
import time

# ===============================================================
# 💡 실시간 모니터링 스레드
# ===============================================================
stop_flag = False
def monitor(label="Progress"):
    start = time.time()
    while not stop_flag:
        mem = psutil.virtual_memory().used / 1024**3
        cpu = psutil.cpu_percent(interval=1)
        elapsed = time.time() - start
        print(f"[{label}] ⏱ {elapsed:6.1f}s | 💾 {mem:6.2f} GB | ⚙️ CPU {cpu:5.1f}%")
        time.sleep(4)

# ===============================================================
# ① Step 1: 파일 로드 + 고객 단위 요약
# ===============================================================
print("📂 Step 1: 데이터 로드 및 요약 중...")

dtype_card = {
    'customer_id': 'string',
    'BAS_YH': 'category',
    'SEX_CD': 'category',
    'MBR_RK': 'category'
}
dtype_account = {'customer_id': 'string'}
dtype_loan = {
    'customer_id': 'string',
    'loan_type': 'category',
    'interest_type': 'category',
    'repayment_method': 'category'
}

card = pd.read_csv('card.csv', dtype=dtype_card, low_memory=False)
account = pd.read_csv('account.csv', dtype=dtype_account, low_memory=False)
loan = pd.read_csv('loan.csv', dtype=dtype_loan, low_memory=False)

print(f"✅ 파일 로드 완료 | card: {len(card):,}행 | account: {len(account):,}행 | loan: {len(loan):,}행")

# -----------------------------
# 고객 단위로 요약 (행 수 최소화)
# -----------------------------
print("🔧 Step 1-1: account 요약 중...")
account_summary = account.groupby('customer_id', as_index=False).agg({
    'balance': 'sum'
})
print(f"✅ account 요약 완료: {len(account_summary):,}명")

print("🔧 Step 1-2: loan 요약 중...")
loan_summary = loan.groupby('customer_id', as_index=False).agg({
    'principal_amount': 'sum',
    'remaining_principal': 'sum',
    'interest_rate': 'mean',
    'loan_type': 'first',
    'interest_type': 'first',
    'repayment_method': 'first'
})
print(f"✅ loan 요약 완료: {len(loan_summary):,}명")

del account, loan
gc.collect()

# ===============================================================
# ② Step 2: card + account_summary + loan_summary 병합
# ===============================================================
print("\n📂 Step 2: 병합 시작 (card 기준 유지)")

stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("병합 진행",), daemon=True)
monitor_thread.start()

df = card.merge(account_summary, on='customer_id', how='left') \
         .merge(loan_summary, on='customer_id', how='left')

stop_flag = True
monitor_thread.join()

print(f"✅ Step 2 병합 완료: {df.shape}")
print(f"💾 현재 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")

# ===============================================================
# ③ 결측치 처리 + 타입 최적화
# ===============================================================
print("\n🧹 Step 3: 결측치 처리 중...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols = df.select_dtypes(include=['float', 'int']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

if len(num_cols) > 0:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
if len(cat_cols) > 0:
    df[cat_cols] = df[cat_cols].fillna('unknown')

df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')
gc.collect()
print(f"✅ 결측치 처리 완료 | 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")

# ===============================================================
# ④ 저장 (진행률 표시)
# ===============================================================
output_path = "train_dataset.csv"
print("\n💾 Step 4: train_dataset.csv 저장 중...")

chunk_size = 200_000
total_rows = len(df)
written = 0
first = True

stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("CSV 저장",), daemon=True)
monitor_thread.start()

for start in range(0, total_rows, chunk_size):
    df.iloc[start:start+chunk_size].to_csv(
        output_path,
        mode='w' if first else 'a',
        header=first,
        index=False,
        encoding='utf-8-sig'
    )
    first = False
    written += chunk_size
    progress = min(written / total_rows * 100, 100)
    print(f"  ▶ 저장 진행률: {progress:5.1f}% ({min(written,total_rows):,}/{total_rows:,})")

stop_flag = True
monitor_thread.join()

print(f"\n✅ train_dataset.csv 생성 완료! ({len(df):,}행, {len(df.columns)}컬럼)")
print(f"🧠 최종 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")
