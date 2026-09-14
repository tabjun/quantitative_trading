import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"]=["Noto Sans CJK KR","DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt

P="/tmp/claude-1002/-home-std-jun99120-personal-ai-project-tools-vscode/fc52cb5c-495a-4a72-a905-2a961a95d970/scratchpad/poster"
d=pd.read_csv(f"{P}/regime.csv")
LABS=["Q1","Q2","Q3","Q4","Q5"]
XT=["1시간 뒤 ±0.3%\n이내(가장 잔잔)","±0.3~0.4%","±0.4~0.6%\n(보통)","±0.6~0.9%",
    "±0.9% 이상\n(급등락, 최대 22%)"]
ORDER=["GARCH-t","LSTM","GRU","LightGBM","HAR-RV","naive"]
COL={"GARCH-t":"#2C7BB6","LSTM":"#D95F02","GRU":"#E8A33D","LightGBM":"#7B68EE",
     "HAR-RV":"#B8860B","naive":"#C85A3E"}
pv=d.pivot_table(index="모델",columns="구간",values="포착률",aggfunc="mean")[LABS]
pe=d.pivot_table(index="모델",columns="구간",values="상대오차",aggfunc="mean")[LABS]

fig,ax=plt.subplots(1,2,figsize=(17,6.8))
a=ax[0]
a.axhspan(1.0,2.75,color="#C85A3E",alpha=.05); a.axhspan(0.4,1.0,color="#2C7BB6",alpha=.05)
a.axhline(1.0,color="#333",lw=2,ls="--",zorder=1)
a.text(4.12,1.04,"완벽한 예측 = 1.0",fontsize=11,color="#333",ha="right")
a.text(0.10,2.62,"과대예측 구역",fontsize=12,color="#8B3A26",weight="bold")
a.text(0.10,0.44,"과소예측 구역",fontsize=12,color="#1B4F72",weight="bold")
for nm in ORDER:
    if nm not in pv.index: continue
    big = nm in ("GARCH-t","HAR-RV")
    a.plot(range(5),pv.loc[nm],marker="o",ms=9,lw=3.2 if big else 2,
           color=COL[nm],label=nm,alpha=1 if big else .7,zorder=3)
a.annotate(f"급변 구간에서 실제의 {pv.loc['HAR-RV','Q5']*100:.0f}%만 포착",
           xy=(4,pv.loc["HAR-RV","Q5"]),xytext=(1.45,0.63),fontsize=11.5,color="#8B6508",
           weight="bold",arrowprops=dict(arrowstyle="->",color="#8B6508",lw=1.8))
a.set_xticks(range(5)); a.set_xticklabels(XT,fontsize=11)
a.set_ylabel("포착률 = 예측 평균 ÷ 실제 평균",fontsize=13)
a.set_xlabel("1시간 뒤 실제 변동성 크기 구간(20%씩)",fontsize=13)
a.set_title("(R-1) 모든 모델이 평균으로 수축한다\n변동이 작을 땐 부풀리고, 변동이 클 땐 절반만 잡는다",
            fontsize=14,weight="bold",pad=12)
a.legend(fontsize=11,ncol=2,frameon=False,loc="upper right")
a.grid(alpha=.25); a.set_ylim(0.4,2.75)
for s in ("top","right"): a.spines[s].set_visible(False)

a=ax[1]; w=0.14
for i,nm in enumerate(ORDER):
    if nm not in pe.index: continue
    a.bar(np.arange(5)+(i-2.5)*w,pe.loc[nm],width=w,color=COL[nm],label=nm,
          alpha=1 if nm!="naive" else .6)
for j,c in enumerate(LABS):
    b=pe[c].idxmin()
    a.text(j,pe[c].max()+0.04,f"최우수\n{b}",ha="center",fontsize=10.5,
           weight="bold",color=COL[b])
a.set_xticks(range(5)); a.set_xticklabels(XT,fontsize=11)
a.set_ylabel("상대오차 (낮을수록 좋음)",fontsize=13)
a.set_xlabel("1시간 뒤 실제 변동성 크기 구간(20%씩)",fontsize=13)
a.set_title("(R-2) 구간마다 최우수 모델이 다르다\n단일 승자는 없다",fontsize=14,weight="bold",pad=12)
a.legend(fontsize=10,ncol=3,frameon=False,loc="upper left")
a.grid(alpha=.25,axis="y"); a.set_ylim(0,pe.values.max()*1.33)
for s in ("top","right"): a.spines[s].set_visible(False)

fig.suptitle("그림 R. 왜 단일 승자가 없는가 — 변동성 구간별 성능 분해 (8종목)",
             fontsize=16,weight="bold",y=1.005)
fig.tight_layout()
fig.savefig(f"{P}/fig_regime.png",dpi=150,bbox_inches="tight",facecolor="white")
print("완료")
