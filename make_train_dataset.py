import pandas as pd
import numpy as np
import psutil
import threading
import time
import gc

# ===============================================================
# 실시간 모니터링 스레드
# ===============================================================
stop_flag = False

def monitor(label="train_dataset 생성"):
    """5초마다 CPU, 메모리, 경과시간 출력"""
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

print(f"파일 로드 완료 | card: {len(card):,}행 | account: {len(account):,}행 | loan: {len(loan):,}행")

# ===============================================================
# 병합 (분기 기준 1:1 매칭)
# ===============================================================
print("\n병합 시작 (분기 단위 1:1 매칭)")

stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("병합 진행",), daemon=True)
monitor_thread.start()

df = (
    card.merge(account, on=['customer_id', 'BAS_YH'], how='left')
        .merge(loan, on=['customer_id', 'BAS_YH'], how='left')
)

stop_flag = True
monitor_thread.join()

print(f"병합 완료: {len(df):,}행, {df['customer_id'].nunique():,}명")
print(f"현재 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")

# ===============================================================
# Step 3. 결측치 처리
# ===============================================================
print("\n결측치 처리 중...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

num_cols = df.select_dtypes(include=['float', 'int']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

if len(num_cols) > 0:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

for col in cat_cols:
    if df[col].dtype.name == "category":
        if 'unknown' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(['unknown'])
        df[col] = df[col].fillna('unknown')
    else:
        df[col] = df[col].fillna('unknown')

gc.collect()
print(f"결측치 처리 완료 | 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")

# ===============================================================
# 분기별 변화율 Feature 생성
# ===============================================================
print("\n분기별 변화율 Feature 생성 중...")

# 분기를 정렬 가능한 형태로 변환
def quarter_to_order(q):
    try:
        year, qtr = int(q[:4]), int(q[-1])
        return year * 4 + qtr
    except:
        return np.nan

df['quarter_order'] = df['BAS_YH'].astype(str).apply(quarter_to_order)

# 고객별 정렬
df.sort_values(by=['customer_id', 'quarter_order'], inplace=True)

# 변화율 계산 대상 컬럼
change_targets = ['salary', 'balance', 'principal_amount', 'remaining_principal']

for col in change_targets:
    if col in df.columns:
        df[f'{col}_diff'] = df.groupby('customer_id')[col].diff().fillna(0)
        df[f'{col}_pct_change'] = (
            df.groupby('customer_id')[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        )

print("변화율 Feature 생성 완료")

# ===============================================================
# 데이터 저장
# ===============================================================
print("\ntrain_dataset.csv 저장 중...")

output_path = "train_dataset.csv"
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
    print(f"저장 진행률: {progress:5.1f}% ({min(written, total_rows):,}/{total_rows:,})")

stop_flag = True
monitor_thread.join()

print(f"\ntrain_dataset.csv 생성 완료! ({len(df):,}행, {len(df.columns)}컬럼)")
print(f"최종 메모리 사용량: {psutil.virtual_memory().used / 1024**3:.2f} GB")
print(df.head(3))
