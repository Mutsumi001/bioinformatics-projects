import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
import esm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ---------------------- 1. 准备TP53已知突变数据集 ----------------------
print("="*70)
print("【核心项目】基于ESM2的错义突变致病性预测 - TP53小样本演示")
print("="*70)

# TP53已知突变：致病突变和良性多态
# 格式：(位置, 突变型AA, 致病性 1=致病, 0=良性)
tp53_muts = [
    (175, "H", 1),  # 经典致癌突变
    (248, "Q", 1),
    (273, "H", 1),
    (282, "W", 1),
    (151, "S", 1),
    (249, "S", 1),
    (213, "Q", 1),
    (173, "M", 1),
    (220, "C", 1),
    (245, "S", 1),
    (72, "R", 0),   # 良性多态
    (21, "L", 0),
    (47, "S", 0),
    (185, "A", 0),
    (36, "H", 0),
    (203, "R", 0),
    (267, "L", 0),
    (110, "P", 0),
    (337, "C", 0),
    (341, "A", 0)
]

mut_df = pd.DataFrame(tp53_muts, columns=["pos", "alt_aa", "label"])
print(f"\n1. 数据集准备完成：共{len(mut_df)}个TP53突变（致病{(mut_df['label']==1).sum()}个，良性{(mut_df['label']==0).sum()}个）")

# 读取TP53野生型序列
record = next(SeqIO.parse("TP53_human.fasta", "fasta"))
ref_seq = str(record.seq)

# 生成突变型序列的函数（自动读野生型氨基酸，不用手动传）
def generate_mutated_seq(seq, pos, alt_aa):
    # 生物位置转Python索引
    idx = pos - 1
    # 自动读取参考序列里的野生型氨基酸
    ref_aa = seq[idx]
    # 替换氨基酸
    mutated_seq = seq[:idx] + alt_aa + seq[idx+1:]
    return mutated_seq, ref_aa

# ---------------------- 2. 加载ESM模型，提取位点embedding ----------------------
print("\n2. 加载ESM2模型，提取野生型/突变型的位点特征...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

def get_site_embedding(seq, pos):
    """提取序列中pos位置(生物编号)的位点embedding"""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        out = model(tokens, repr_layers=[6])
    rep = out["representations"][6]
    # pos是生物编号，加了<cls>所以索引+1
    site_emb = rep[0, pos, :].numpy()
    return site_emb

# 提取所有突变的特征：野生型位点embedding vs 突变型位点embedding的差值
print("正在批量提取突变特征...")
mut_features = []
for _, row in mut_df.iterrows():
    pos = row["pos"]
    alt_aa = row["alt_aa"]
    
    # 生成突变型序列（自动读野生型）
    mut_seq, ref_aa = generate_mutated_seq(ref_seq, pos, alt_aa)
    
    # 提取野生型和突变型在该位点的embedding
    ref_emb = get_site_embedding(ref_seq, pos)
    alt_emb = get_site_embedding(mut_seq, pos)
    
    # 核心特征：两个embedding的差值（突变前后大模型对这个位点理解的变化量）
    diff_emb = np.abs(ref_emb - alt_emb)
    
    # 拼接差值、野生型embedding、突变型embedding作为特征
    feature = np.concatenate([diff_emb, ref_emb, alt_emb])
    mut_features.append(feature)

X = np.array(mut_features)
y = mut_df["label"].values
print(f"   特征矩阵形状：{X.shape}（{X.shape[0]}个突变，{X.shape[1]}维特征）")

# ---------------------- 3. 训练模型并评估 ----------------------
print("\n3. 划分训练测试集，训练随机森林分类器...")
X_train, X_test, y_train, y_test, muts_train, muts_test = train_test_split(
    X, y, mut_df["pos"].values, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n4. 模型在测试集上的表现：")
print(f"   准确率：{acc:.3f}")
print(f"   AUC：{auc:.3f}")
print(f"\n   测试集预测明细：")
for i, pos in enumerate(muts_test):
    true_str = "致病" if y_test[i]==1 else "良性"
    pred_str = "致病" if y_pred[i]==1 else "良性"
    print(f"    第{pos:3d}位突变：真实{true_str}, 预测{pred_str}（致病概率{y_proba[i]:.2f}）")

# ---------------------- 5. 项目总结 ----------------------
print("\n" + "="*70)
print("✅ 最小闭环跑通！")
print("="*70)
print("  核心逻辑：")
print("  1. 对每个错义突变，生成突变型序列")
print("  2. 用ESM分别提取野生型、突变型在该位点的embedding")
print("  3. 计算两个向量的差值作为特征")
print("  4. 训练分类器预测突变是否致病")
print("\n下一步升级方向：")
print("  - 扩展到ClinVar全数据集（上千个跨基因突变）")
print("  - 按基因划分训练/测试集（避免数据泄露）")
print("  - 和SIFT/PolyPhen等传统工具做性能对比")
print("  - 升级到ESM2-650M大模型，提升准确率")
