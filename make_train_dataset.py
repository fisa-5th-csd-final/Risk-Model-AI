import dask.dataframe as dd
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

def monitor(label="Dask merge"):
    """5초마다 메모리·CPU·시간 경과 출력"""
    start = time.time()
    while not stop_flag:
        mem = psutil.virtual_memory().used / 1024**3
        cpu = psutil.cpu_percent(interval=1)
        elapsed = time.time() - start
        print(f"[{label}] ⏱ {elapsed:6.1f}s | 💾 {mem:6.2f} GB | ⚙️ CPU {cpu:5.1f}%")
        time.sleep(4)

# ===============================================================
# ① Step 1: card + account 병합
# ===============================================================
print("📂 Step 1: card + account 병합 중...")

dtype_card = {
    'customer_id': 'category',
    'BAS_YH': 'category',
    'SEX_CD': 'category',
    'MBR_RK': 'category'
}
dtype_account = {'customer_id': 'category'}

card = dd.read_csv('card.csv', dtype=dtype_card, blocksize="256MB")
account = dd.read_csv('account.csv', dtype=dtype_account, blocksize="128MB")

# 모니터링 스레드 시작
monitor_thread = threading.Thread(target=monitor, args=("Step 1 병합",), daemon=True)
monitor_thread.start()

df = card.merge(account, on='customer_id', how='left')
df = df.persist()  # 병합 결과를 메모리에 캐시 (compute 전에 최적화)
df.compute()       # 실제 연산 수행
stop_flag = True
monitor_thread.join()

print("✅ Step 1 병합 완료 (Dask DataFrame)")
print(f"💾 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")

# 임시 저장
tmp_path = "merged_card_account_tmp.parquet"
df.to_parquet(tmp_path, engine="pyarrow", write_index=False)
print(f"💾 임시 저장 완료 → {tmp_path}")

del card, account, df
gc.collect()
print("🧹 메모리 초기화 완료")

# ===============================================================
# ② Step 2: loan 병합
# ===============================================================
print("\n📂 Step 2: loan 병합 중...")

dtype_loan = {
    'customer_id': 'category',
    'loan_type': 'category',
    'interest_type': 'category',
    'repayment_method': 'category'
}

df = dd.read_parquet(tmp_path)
loan = dd.read_csv('loan.csv', dtype=dtype_loan, blocksize="256MB")

stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("Step 2 병합",), daemon=True)
monitor_thread.start()

df = df.merge(loan, on='customer_id', how='left')
df = df.persist()
df.compute()

stop_flag = True
monitor_thread.join()

print("✅ Step 2 병합 완료 (lazy → computed)")
print(f"💾 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")

# ===============================================================
# ③ 결측치 처리 + 타입 최적화
# ===============================================================
print("\n🧹 결측치 및 타입 정리 중...")

df = df.replace([np.inf, -np.inf], np.nan)
num_cols = [c for c, dt in df.dtypes.items() if np.issubdtype(dt, np.number)]
cat_cols = [c for c, dt in df.dtypes.items() if dt == "category" or dt == "object"]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna("unknown")

print("✅ 결측치 처리 완료 (lazy)")

# ===============================================================
# ④ 최종 저장
# ===============================================================
print("\n💾 최종 train_dataset.csv 저장 중...")
stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("CSV 저장",), daemon=True)
monitor_thread.start()

df.to_csv("train_dataset.csv", single_file=True, index=False, encoding='utf-8-sig')

stop_flag = True
monitor_thread.join()

print("✅ train_dataset.csv 생성 완료!")
print(f"📊 컬럼 수: {len(df.columns)}")

os.remove(tmp_path)
print("🧹 임시파일 삭제 완료!")
