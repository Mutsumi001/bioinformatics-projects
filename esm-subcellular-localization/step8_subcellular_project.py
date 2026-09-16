import requests
import numpy as np
import pandas as pd
import torch
from io import StringIO
from Bio import SeqIO
import esm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- 1. 数据集：100个真实人源蛋白（UniProt公开数据） ----------------------
print("="*70)
print("【项目】基于ESM2 embedding的蛋白亚细胞定位预测：核蛋白 vs 细胞膜蛋白")
print("="*70)

# 50个经典人源核蛋白（label=1）
nuclear_proteins = {
    "P04637": "TP53", "P0CG48": "UBC", "P62805": "H4", "P04908": "H2A",
    "P16403": "H1", "P38398": "BRCA1", "P04150": "GR", "Q13875": "BARD1",
    "P19484": "RARA", "P10275": "AR", "P03372": "ER", "Q96I51": "FOXM1",
    "P08047": "SP1", "Q01196": "E2F1", "P17026": "XRCC1", "P12036": "NF1",
    "Q13547": "HDAC1", "P02811": "FOS", "P01100": "JUN", "P15407": "ETS1",
    "P17028": "RUNX1", "Q01826": "MYC", "P04198": "MAX", "P15516": "REL",
    "P19838": "NFKB1", "P22090": "POU2F1", "Q01196": "E2F1", "P17027": "TP53BP1",
    "P20226": "TBP", "P24928": "POLR2A", "P18583": "SRF", "Q04206": "RELA",
    "P17542": "STAT1", "P42212": "EYA2", "P06749": "GSTP1", "P08670": "VIM",
    "P35527": "KRT9", "P02545": "LMNA", "Q14676": "NBN", "P49916": "MDM2",
    "P20701": "ITGAM", "P08651": "TOP1", "P11365": "TOP2A", "P31269": "RPA1",
    "Q04835": "SMC1A", "P49841": "GSK3B", "P0DP23": "CALM1", "P60709": "ACTB",
    "P02533": "KRT14", "P04264": "KRT1"
}

# 50个经典人源细胞膜蛋白（label=0）
membrane_proteins = {
    "P04626": "EGFR", "P00533": "EGFR2", "P35568": "IRS1", "P08588": "ADRB1",
    "P13945": "CTLA4", "Q99836": "CD28", "P20718": "IL2RA", "P11233": "RAS",
    "P04049": "RAF1", "P15056": "BRAF", "P01375": "TNF", "P01579": "IFNG",
    "P05231": "IL6", "P01579": "IFNG", "P13232": "IL4R", "P20814": "CXCR4",
    "P49286": "OPRM1", "P25025": "HTR2A", "P30968": "SLC6A4", "P21918": "DRD2",
    "P08123": "SLC2A1", "P11166": "GLUT1", "P08236": "ATP1A1", "Q92735": "ABCB1",
    "P23976": "AQP1", "P33681": "AQP2", "P25025": "HTR2A", "P41143": "D4R",
    "P30939": "SLC6A3", "Q12809": "KCNH2", "Q14764": "SCN5A", "P35498": "SCN1A",
    "P19256": "CACNA1C", "Q06432": "GABA", "P30542": "ADORA1", "P0DMS8": "NAV17",
    "P0DP23": "CALM1", "P04049": "RAF1", "P08588": "ADRB1", "P04626": "EGFR",
    "P14770": "IL1B", "P01583": "IL2", "P13232": "IL4R", "P20814": "CXCR4",
    "P16871": "COL1A1", "P02751": "FN1", "P08571": "CD14", "P05107": "ITGB2",
    "P05556": "ITGB1", "P11215": "ITGAM"
}

def download_uniprot(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and ">" in resp.text:
            record = next(SeqIO.parse(StringIO(resp.text), "fasta"))
            return str(record.seq)
    except:
        pass
    return None

# 下载序列，去重
print("\n1. 正在从UniProt下载100个蛋白序列...")
all_data = {}
for uid, name in nuclear_proteins.items():
    if uid not in all_data:
        seq = download_uniprot(uid)
        if seq:
            all_data[uid] = (name, seq, 1)  # 核蛋白=1

for uid, name in membrane_proteins.items():
    if uid not in all_data:
        seq = download_uniprot(uid)
        if seq:
            all_data[uid] = (name, seq, 0)  # 膜蛋白=0

df_raw = pd.DataFrame([(k, v[0], v[1], v[2]) for k,v in all_data.items()],
                      columns=["uniprot_id", "name", "sequence", "label"])
df_raw = df_raw.drop_duplicates(subset=["uniprot_id"]).reset_index(drop=True)
print(f"   成功下载有效序列：{len(df_raw)}个（核蛋白{(df_raw['label']==1).sum()}，膜蛋白{(df_raw['label']==0).sum()}）")

# ---------------------- 2. 用ESM2提取embedding ----------------------
print("\n2. 正在用ESM2模型提取蛋白embedding...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

esm_features = []
batch_size = 10
for i in range(0, len(df_raw), batch_size):
    batch_df = df_raw.iloc[i:i+batch_size]
    batch_data = [(row["name"], row["sequence"]) for _, row in batch_df.iterrows()]
    _, _, batch_tokens = batch_converter(batch_data)
    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[6])
    rep = out["representations"][6]
    for j in range(rep.shape[0]):
        mean_emb = rep[j, 1:-1, :].mean(dim=0).numpy()
        esm_features.append(mean_emb)

X = np.array(esm_features)
y = df_raw["label"].values
print(f"   ESM特征矩阵形状：{X.shape}（{X.shape[0]}个样本，{X.shape[1]}维特征）")

# ---------------------- 3. 训练测试 + 模型评估 ----------------------
print("\n3. 划分数据集并训练随机森林分类器...")
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, df_raw["name"].values, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"\n4. 模型在测试集上的表现：")
print(f"   准确率：{acc:.3f}")
print(f"   AUC：{auc:.3f}")
print(f"\n   分类报告：")
print(classification_report(y_test, y_pred, target_names=["膜蛋白", "核蛋白"]))

# ---------------------- 4. 可视化结果 ----------------------
print("\n5. 生成结果可视化图表...")
fig, axes = plt.subplots(1,2, figsize=(12,5))

# 混淆矩阵
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["膜蛋白", "核蛋白"],
            yticklabels=["膜蛋白", "核蛋白"], ax=axes[0])
axes[0].set_title(f"Confusion Matrix (Acc={acc:.2f})")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, label=f"ESM+RF (AUC={auc:.3f})", linewidth=2)
axes[1].plot([0,1],[0,1],'k--', label="Random Guess")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("subcellular_localization_results.png", dpi=150, bbox_inches="tight")
print("   结果图已保存：subcellular_localization_results.png")

# 保存数据集和结果
df_raw.to_csv("subcellular_dataset.csv", index=False, encoding="utf-8-sig")
result_df = pd.DataFrame({
    "protein_name": names_test,
    "true_label": ["膜蛋白" if l==0 else "核蛋白" for l in y_test],
    "pred_label": ["膜蛋白" if p==0 else "核蛋白" for p in y_pred],
    "pred_proba_nuclear": y_proba.round(3)
})
result_df.to_csv("subcellular_predictions.csv", index=False, encoding="utf-8-sig")

# ---------------------- 5. 项目总结 ----------------------
print("\n" + "="*70)
print("✅ 项目完成！总结：")
print("="*70)
print(f"  任务：区分核蛋白和细胞膜蛋白")
print(f"  数据：{len(df_raw)}个真实人源蛋白（UniProt公开数据）")
print(f"  特征：ESM2-8M模型提取的320维embedding")
print(f"  模型：随机森林分类器")
print(f"  测试集准确率：{acc:.3f}，AUC：{auc:.3f}")
print("\n项目文件清单：")
print("  - subcellular_dataset.csv：原始数据集")
print("  - subcellular_predictions.csv：测试集预测结果")
print("  - subcellular_localization_results.png：混淆矩阵+ROC曲线")
print("  - step8_subcellular_project.py：完整可复现代码")
