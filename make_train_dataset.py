import pandas as pd
import numpy as np
import gc
import os
import psutil

def print_mem(label=""):
    """현재 메모리 사용량 출력"""
    mem = psutil.virtual_memory().used / 1024**3
    print(f"💾 [{label}] 현재 메모리 사용량: {mem:.2f} GB")

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

card = pd.read_csv('card.csv', dtype=dtype_card, low_memory=False)
account = pd.read_csv('account.csv', dtype=dtype_account, low_memory=False)
print_mem("파일 로드 직후")

df = card.merge(account, on='customer_id', how='left')
print(f"✅ Step 1 병합 완료: {df.shape}")
print_mem("1단계 병합 후")

# 임시 저장
tmp_path = "merged_card_account_tmp.csv"
df.to_csv(tmp_path, index=False)
print(f"💾 임시 저장 완료 → {tmp_path}")

# 메모리 완전 해제
del card, account, df
gc.collect()
print_mem("메모리 초기화 후")

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

df = pd.read_csv(tmp_path, low_memory=False)
loan = pd.read_csv('loan.csv', dtype=dtype_loan, low_memory=False)
print_mem("2단계 파일 로드 직후")

# loan 컬럼을 분할 병합 (메모리 절약)
loan_cols = [col for col in loan.columns if col != 'customer_id']
split_size = 5
loan_colsets = [loan_cols[i:i + split_size] for i in range(0, len(loan_cols), split_size)]

for idx, cols in enumerate(loan_colsets, start=1):
    print(f"  ▶ loan 파트 {idx}/{len(loan_colsets)} 병합 중... ({len(cols)}개 컬럼)")
    df = df.merge(loan[['customer_id'] + cols], on='customer_id', how='left')
    gc.collect()
    print(f"     └ 병합 후 shape: {df.shape}")
    print_mem(f"loan 파트 {idx} 후")

del loan
gc.collect()
print_mem("loan 전체 병합 완료")

# ===============================================================
# ③ 결측치 처리 + 타입 최적화
# ===============================================================
print("\n🧹 결측치 및 타입 정리 중...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols = df.select_dtypes(include=['float', 'int']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

if len(num_cols) > 0:
    median_values = df[num_cols].median(numeric_only=True)
    df[num_cols] = df[num_cols].fillna(median_values)
if len(cat_cols) > 0:
    df[cat_cols] = df[cat_cols].fillna('unknown')

df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')
gc.collect()
print_mem("결측치 처리 완료")

# ===============================================================
# ④ 최종 저장
# ===============================================================
output_path = "train_dataset.csv"
print(f"\n💾 최종 train_dataset.csv 저장 중...")
chunk_size = 100_000
first = True
for i in range(0, len(df), chunk_size):
    df.iloc[i:i + chunk_size].to_csv(
        output_path,
        mode='w' if first else 'a',
        header=first,
        index=False,
        encoding='utf-8-sig'
    )
    first = False
    print(f"  ▶ {i + chunk_size:,}행까지 저장 완료")

print(f"✅ train_dataset.csv 생성 완료! (총 {len(df):,}행, {len(df.columns)}컬럼)")
print_mem("최종 완료")

# 임시파일 삭제
os.remove(tmp_path)
print("🧹 임시파일 삭제 완료!")
