import pandas as pd
import numpy as np
import psutil
import threading
import time
import gc
import os

# ===============================================================
# 실시간 모니터링 스레드
# ===============================================================
stop_flag = False

def monitor(label="병합 진행"):
    start = time.time()
    while not stop_flag:
        mem = psutil.virtual_memory().used / 1024**3
        cpu = psutil.cpu_percent(interval=1)
        elapsed = time.time() - start
        print(f"[{label}] {elapsed:6.1f}s | {mem:6.2f} GB | CPU {cpu:5.1f}%")
        time.sleep(4)

# ===============================================================
# 데이터 로드
# ===============================================================
print("데이터 로드 중...")

dtype_card = {
    'customer_id': 'string',
    'BAS_YH': 'category',
    'SEX_CD': 'category',
    'MBR_RK': 'category'
}
dtype_account = {
    'customer_id': 'string',
    'BAS_YH': 'category'
}
dtype_loan = {
    'customer_id': 'string',
    'BAS_YH': 'category',
    'loan_type': 'category',
    'interest_type': 'category',
    'repayment_method': 'category'
}

card = pd.read_csv('card.csv', dtype=dtype_card, low_memory=False)
account = pd.read_csv('account.csv', dtype=dtype_account, low_memory=False)
loan = pd.read_csv('loan.csv', dtype=dtype_loan, low_memory=False)

print(f"파일 로드 완료 — card: {len(card):,}, account: {len(account):,}, loan: {len(loan):,}")

# ===============================================================
# 데이터 병합
# ===============================================================
print("\n병합 시작...")
stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("데이터 병합",), daemon=True)
monitor_thread.start()

df = card.merge(account, on=['customer_id', 'BAS_YH'], how='left') \
         .merge(loan, on=['customer_id', 'BAS_YH'], how='left')

stop_flag = True
monitor_thread.join()

print(f"병합 완료: {df.shape[0]:,}행, {df.shape[1]}열")

# ===============================================================
# 결측치 처리 (category-safe)
# ===============================================================
print("\n결측치 처리 중...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

num_cols = df.select_dtypes(include=['float', 'int']).columns
cat_cols = df.select_dtypes(exclude=['float', 'int']).columns

# 수치형 → 중앙값
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# 범주형 → 'unknown'
for col in cat_cols:
    if pd.api.types.is_categorical_dtype(df[col]):
        if 'unknown' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(['unknown'])
        df[col] = df[col].fillna('unknown')
    else:
        df[col] = df[col].fillna('unknown')

# ===============================================================
# 수치형 단위 정리 및 타입 보정
# ===============================================================
print("\n단위 정리 및 타입 변환 중...")

# ① 금액 컬럼 절사 (천원 단위)
money_cols = ['salary', 'balance', 'principal_amount', 'remaining_principal']
for col in money_cols:
    if col in df.columns:
        df[col] = (df[col] // 10_000).astype(int)

# ② 나이·연체 여부 정수형 변환
int_cols = ['AGE', 'is_delinquent_x', 'is_delinquent_y']
for col in int_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

# ③ 금리만 소수점 2자리 유지
if 'interest_rate' in df.columns:
    df['interest_rate'] = df['interest_rate'].round(2)

print("✅ 단위 정리 및 타입 변환 완료")

gc.collect()
mem = df.memory_usage(deep=True).sum() / 1024**2
print(f"현재 메모리 사용량: {mem:.2f} MB")

# ===============================================================
# 저장 (chunk 단위)
# ===============================================================
print("\ntrain_dataset.csv 저장 중...")
output_path = 'train_dataset.csv'
if os.path.exists(output_path):
    os.remove(output_path)

stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("CSV 저장",), daemon=True)
monitor_thread.start()

chunk_size = 100_000
first = True
for i in range(0, len(df), chunk_size):
    df.iloc[i:i+chunk_size].to_csv(
        output_path,
        mode='w' if first else 'a',
        header=first,
        index=False,
        encoding='utf-8-sig'
    )
    first = False
    print(f"{i+chunk_size:,}행까지 저장 완료")

stop_flag = True
monitor_thread.join()

print(f"\n✅ train_dataset.csv 생성 완료 ({len(df):,}행, {len(df.columns)}열)")
print(df.head(3))
