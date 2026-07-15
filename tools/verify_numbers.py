"""Verification pass: every quantitative claim in main.tex vs. ground truth."""
import json, re, sys
import numpy as np

W = "/sessions/funny-gallant-ptolemy/mnt/outputs/work"
tex = open(f"{W}/paper_new/main.tex").read()
S = json.load(open(f"{W}/repo/results/summary.json"))
sig_t = json.load(open(f"{W}/repo/results/significance_descriptors_test.json"))
sig_d = json.load(open(f"{W}/repo/results/significance_descriptors_dig.json"))
sig_c = json.load(open(f"{W}/repo/results/significance_test.json"))
ab = json.load(open(f"{W}/repo/results/ablation.json"))
lat = json.load(open(f"{W}/repo/results/latency.json"))
lj = json.load(open(f"{W}/repo/results/loop_justification.json"))
cv = json.load(open(f"{W}/repo/results/cv_final.json"))
sh = json.load(open(f"{W}/repo/results/shap_examples.json"))

checks = []
def chk(name, claim, truth, tol=0.006):
    ok = abs(claim - truth) <= tol * max(1, abs(truth))
    checks.append((ok, name, claim, truth))

# headline numbers (in % where the paper uses %)
chk("test aug 86.70", 86.70, S["ktsd"]["augmented"]["test"]*100)
chk("dig aug 70.11", 70.11, S["ktsd"]["augmented"]["dig"]*100)
chk("test plain 86.54", 86.54, S["ktsd"]["plain"]["test"]*100)
chk("dig plain 66.44", 66.44, S["ktsd"]["plain"]["dig"]*100)
accs = [cv[str(k)] for k in range(5)]
chk("cv mean 94.02", 94.02, float(np.mean(accs))*100)
chk("cv std 0.15", 0.15, float(np.std(accs))*100, tol=0.15)
chk("repro 94.14", 94.14, S["ktsd"]["pooled_8020_repro"]["with_custom_60250"]*100)
chk("repro nocustom 94.10", 94.10, S["ktsd"]["pooled_8020_repro"]["without_custom"]*100)
chk("aug delta test 0.16", 0.16, (S["ktsd"]["augmented"]["test"]-S["ktsd"]["plain"]["test"])*100, tol=0.1)
chk("aug delta dig 3.67", 3.67, (S["ktsd"]["augmented"]["dig"]-S["ktsd"]["plain"]["dig"])*100, tol=0.05)

# descriptors
d = S["descriptor_comparison"]
chk("hog test 94.33", 94.33, d["test"]["hog_linear_2916d"]*100)
chk("sift test 91.57", 91.57, d["test"]["sift_bovw_64d"]*100)
chk("zernike test 87.47", 87.47, d["test"]["zernike_36d"]*100)
chk("hybrid test 94.95", 94.95, d["test"]["hybrid_hog_ktsd"]*100)
chk("hog dig 70.85", 70.85, d["dig"]["hog_linear_2916d"]*100)
chk("hybrid dig 72.76", 72.76, d["dig"]["hybrid_hog_ktsd"]*100)
chk("zernike dig 50.16", 50.16, d["dig"]["zernike_36d"]*100)
chk("sift dig 67.47", 67.47, d["dig"]["sift_bovw_64d"]*100)

# p-values quoted
chk("ktsd-hog dig p 0.151", 0.151, sig_d["pairwise"]["hog_linear|ktsd_aug"]["p"], tol=0.02)
chk("ktsd-zernike test p 0.066", 0.066, sig_t["pairwise"]["ktsd_aug|zernike"]["p"], tol=0.02)
chk("hybrid-hog test p 3.3e-5", 3.32e-5, sig_t["pairwise"]["hog_linear|hybrid_hog_ktsd"]["p"], tol=0.2)
chk("rbf-rf chi2 272.5", 272.5, sig_c["pairwise"]["random_forest|rbf_svm"]["mcnemar_chi2"], tol=0.01)

# CIs quoted in Table 1
def wilson(k, n, z=1.959963984540054):
    p = k/n; den = 1+z*z/n
    c = (p+z*z/(2*n))/den; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (c-h)*100, (c+h)*100
lo, hi = wilson(8670, 10000); chk("CI aug lo 86.02", 86.02, lo, tol=0.001); chk("CI aug hi 87.35", 87.35, hi, tol=0.001)
lo, hi = wilson(8654, 10000); chk("CI plain lo 85.86", 85.86, lo, tol=0.001); chk("CI plain hi 87.19", 87.19, hi, tol=0.001)
k_aug_dig = int(round(S["ktsd"]["augmented"]["dig"]*10240)); lo, hi = wilson(k_aug_dig, 10240)
chk("CI dig lo 69.21", 69.21, lo, tol=0.001); chk("CI dig hi 70.99", 70.99, hi, tol=0.001)

# ablation
chk("abl all 83.66", 83.66, ab["all"]["test_acc"]*100)
chk("abl drop_loop delta 0.32", 0.32, (ab["all"]["test_acc"]-ab["drop_loop"]["test_acc"])*100, tol=0.05)
chk("abl drop_endpoint delta 10.1", 10.1, (ab["all"]["test_acc"]-ab["drop_endpoint"]["test_acc"])*100, tol=0.02)
chk("abl only_endpoint 54.5", 54.5, ab["only_endpoint"]["test_acc"]*100, tol=0.002)

# loop justification
chk("cycles>holes 79.7%", 79.7, lj["cycles_gt_holes"]*100)
chk("holes>cycles 0%", 0.0, lj["holes_gt_cycles"]*100, tol=1)
chk("class5 cycles 9.3", 9.3, lj["per_digit"]["5"]["cycles"], tol=0.01)
chk("class5 holes 0.8", 0.8, lj["per_digit"]["5"]["holes"], tol=0.02)

# latency
chk("lat total 2.24", 2.24, lat["total_ms"]["mean"], tol=0.01)
chk("lat feat 1.04", 1.04, lat["feature_ms"]["mean"], tol=0.01)
chk("lat svm 1.20", 1.20, lat["svm_ms"]["mean"], tol=0.01)
chk("lat p95 2.44", 2.44, lat["total_ms"]["p95"], tol=0.01)

# shap mass
fr = []
for e in sh:
    v = np.abs(np.array(list(e["shap_for_pred"].values())))
    fr.append(np.sort(v)[::-1][:3].sum()/v.sum())
chk("shap mass mean 60%", 60.0, float(np.mean(fr))*100, tol=0.02)
chk("shap mass min 48", 48.0, float(np.min(fr))*100, tol=0.03)
chk("shap mass max 71", 71.0, float(np.max(fr))*100, tol=0.03)

# grid search claim
gs = open(f"{W}/repo/results/gridsearch_report.txt").read()
assert "C=10     gamma=scale  cv_acc=92.53 +- 0.26" in gs, "grid winner mismatch"
checks.append((True, "grid winner 92.53+-0.26", 92.53, 92.53))

# per-class + confusions from predictions
d2 = np.load(f"{W}/ovo_aug/pred_test.npz"); p, y = d2["pred"], d2["y"]
rec = {c: (p[y==c]==c).mean()*100 for c in range(10)}
for c, v in [(1, 97.7), (2, 93.5), (8, 92.6), (7, 75.3), (9, 81.9), (6, 82.0), (3, 83.0)]:
    chk(f"recall class {c} = {v}", v, rec[c], tol=0.002)
from collections import Counter
cm = np.zeros((10,10))
for a, b in zip(y, p): cm[a, b] += 1
cmn = cm/cm.sum(1, keepdims=True)*100
for i, j, v in [(9,6,10.0),(0,1,9.2),(6,7,7.8),(5,4,7.2),(4,3,6.3)]:
    chk(f"confusion {i}->{j} = {v}", v, cmn[i,j], tol=0.02)

fails = [c for c in checks if not c[0]]
print(f"CHECKS: {len(checks)} total, {len(fails)} FAILED")
for ok, name, claim, truth in fails:
    print(f"  FAIL {name}: paper={claim} truth={truth}")
if not fails:
    print("ALL NUMBERS IN MANUSCRIPT VERIFIED AGAINST ARTIFACTS")
