import pandas as pd
import numpy as np
import gc

# ===============================================================
# ① 데이터 불러오기 (dtype 지정으로 메모리 절약)
# ===============================================================
dtype_card = {
    'customer_id': 'category',
    'BAS_YH': 'category',
    'SEX_CD': 'category',
    'MBR_RK': 'category'
}
dtype_account = {'customer_id': 'category'}
dtype_loan = {
    'customer_id': 'category',
    'loan_type': 'category',
    'interest_type': 'category',
    'repayment_method': 'category'
}

print("📂 파일 로드 중...")
card = pd.read_csv('card.csv', dtype=dtype_card, low_memory=False)
account = pd.read_csv('account.csv', dtype=dtype_account, low_memory=False)
loan = pd.read_csv('loan.csv', dtype=dtype_loan, low_memory=False)
print(f"✅ 파일 로드 완료: card({len(card):,}), account({len(account):,}), loan({len(loan):,})")

# ===============================================================
# ② 고객 기준 병합 (2단계: 컬럼 분할 병합)
# ===============================================================
print("\n🔗 1단계: card + account 병합 중...")
df = card.merge(account, on='customer_id', how='left')
del account
gc.collect()
print(f"✅ 중간 병합 완료: {df.shape}")

# -------------------------------
# loan 컬럼 나누어서 순차 병합
# -------------------------------
print("\n🔗 2단계: loan 컬럼 분할 병합 중...")

loan_cols = [col for col in loan.columns if col != 'customer_id']
split_size = 5  # 한 번에 병합할 컬럼 수 (메모리 줄이기)
loan_colsets = [loan_cols[i:i + split_size] for i in range(0, len(loan_cols), split_size)]

for idx, cols in enumerate(loan_colsets, start=1):
    print(f"  ▶ loan 파트 {idx}/{len(loan_colsets)} 병합 중... ({len(cols)}개 컬럼)")
    df = df.merge(loan[['customer_id'] + cols], on='customer_id', how='left')
    gc.collect()
    print(f"     └ 병합 후 shape: {df.shape}")

del loan
gc.collect()
print(f"✅ 전체 병합 완료: {df.shape}")

# ===============================================================
# ③ 결측치 처리 (벡터화 + 중앙값 기반)
# ===============================================================
print("\n🧹 결측치 처리 중...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

num_cols = df.select_dtypes(include=['float', 'int']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# 수치형은 중앙값, 범주형은 'unknown'
if len(num_cols) > 0:
    median_values = df[num_cols].median(numeric_only=True)
    df[num_cols] = df[num_cols].fillna(median_values)
if len(cat_cols) > 0:
    df[cat_cols] = df[cat_cols].fillna('unknown')

print("✅ 결측치 처리 완료")

# ===============================================================
# ④ 메모리 최적화 (downcast)
# ===============================================================
print("\n💾 메모리 최적화 중...")
df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')
gc.collect()
mem_usage = df.memory_usage(deep=True).sum() / 1024**2
print(f"✅ 메모리 사용량: {mem_usage:.2f} MB")

# ===============================================================
# ⑤ CSV 저장 (chunkwriter)
# ===============================================================
output_path = "train_dataset.csv"
print(f"\n💾 {output_path} 저장 중...")

first = True
chunk_size = 100_000

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

print("✅ train_dataset.csv 생성 완료!")
print(f"📊 최종 행 수: {len(df):,} | 컬럼 수: {len(df.columns)}")

# 샘플 출력
print("\n🎯 샘플 데이터:")
print(df.head(3))
