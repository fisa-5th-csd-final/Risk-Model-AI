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
def monitor(label="Merge"):
    start = time.time()
    while not stop_flag:
        mem = psutil.virtual_memory().used / 1024**3
        cpu = psutil.cpu_percent(interval=1)
        elapsed = time.time() - start
        print(f"[{label}] ⏱ {elapsed:6.1f}s | 💾 {mem:6.2f} GB | ⚙️ CPU {cpu:5.1f}%")
        time.sleep(4)

# ===============================================================
# ① Step 1: card + account 병합 (전체 메모리에 올림)
# ===============================================================
print("📂 Step 1: card + account 병합 중...")

dtype_card = {
    'customer_id': 'string',
    'BAS_YH': 'category',
    'SEX_CD': 'category',
    'MBR_RK': 'category'
}
dtype_account = {'customer_id': 'string'}

card = pd.read_csv('card.csv', dtype=dtype_card, low_memory=False)
account = pd.read_csv('account.csv', dtype=dtype_account, low_memory=False)

df = card.merge(account, on='customer_id', how='left')
print(f"✅ Step 1 병합 완료: {df.shape}")

# 임시 저장
tmp_path = "merged_card_account_tmp.csv"
df.to_csv(tmp_path, index=False)
del card, account, df
gc.collect()
print("🧹 메모리 초기화 완료")

# ===============================================================
# ② Step 2: loan을 chunk 단위로 병합
# ===============================================================
print("\n📂 Step 2: loan chunk 단위 병합 중...")

chunk_size = 100_000  # 🔧 메모리 여유에 따라 조정 (50k~200k 추천)
dtype_loan = {
    'customer_id': 'string',
    'loan_type': 'category',
    'interest_type': 'category',
    'repayment_method': 'category'
}

merged_base = pd.read_csv(tmp_path, low_memory=False)
output_path = "train_dataset.csv"
if os.path.exists(output_path):
    os.remove(output_path)

stop_flag = False
monitor_thread = threading.Thread(target=monitor, args=("loan 병합",), daemon=True)
monitor_thread.start()

chunk_idx = 1
for chunk in pd.read_csv('loan.csv', dtype=dtype_loan, chunksize=chunk_size):
    print(f"  ▶ loan chunk {chunk_idx} 병합 중... ({len(chunk):,}행)")
    merged = merged_base.merge(chunk, on='customer_id', how='left')
    
    # 결측치 간단히 처리
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.fillna("unknown", inplace=True)
    
    # 저장 (append 방식)
    merged.to_csv(output_path, mode='a', index=False, header=(chunk_idx==1), encoding='utf-8-sig')
    
    del merged, chunk
    gc.collect()
    print(f"     └ 저장 완료 / 현재 메모리: {psutil.virtual_memory().used / 1024**3:.2f} GB")
    chunk_idx += 1

stop_flag = True
monitor_thread.join()

print("✅ Step 2 전체 병합 완료!")
os.remove(tmp_path)
print("🧹 임시파일 삭제 완료!")

print(f"🎉 최종 train_dataset.csv 생성 완료 ({chunk_idx-1:,}개 청크)")
