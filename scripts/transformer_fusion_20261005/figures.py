from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
FONT='C:/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
PRES=OUT/'presentation'
COLORS=['#4169a1','#8695a3','#2a947f','#e3a340','#9869b9','#d16e57']


def main():
    PRES.mkdir(parents=True,exist_ok=True);s=read(OUT/'summary.json')
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.1),gridspec_kw={'width_ratios':[1,1.2]})
    labels=[v[0] for v in VARIANTS];values=[s['reviewed'][v]['window_mean']['ADE15'] for v in VARIANTS]
    seeds=np.array([[x['ADE15'] for x in s['reviewed'][v]['per_seed'].values()] for v in VARIANTS])
    axes[0].bar(labels,values,color=COLORS,width=.65)
    for i,val in enumerate(values):
        axes[0].scatter([i]*3,seeds[i],s=21,color='#243544',zorder=4)
        axes[0].text(i,val+9,f'{val:.1f}',ha='center',fontsize=11)
    for method,color,label in [('hold_last','#555555','保持最后位置'),('old_comp_dual','#a56444','旧版')]:
        val=s['reviewed'][method]['window_mean']['ADE15'];axes[0].axhline(val,ls='--',color=color,lw=1.3,label=f'{label} {val:.1f}')
    axes[0].set_ylabel('平均轨迹误差 ADE（像素，越低越好）');axes[0].set_title('117 个窗口 × 3 个种子\n点表示单个种子，不是置信区间',fontsize=12)
    axes[0].set_ylim(0,max(values)*1.4);axes[0].legend(fontsize=9,loc='upper right')
    for v,c in zip(VARIANTS,COLORS):axes[1].plot(np.arange(1,16)/10,s['horizon'][v],label=NAMES[v],color=c,lw=2)
    axes[1].plot(np.arange(1,16)/10,s['horizon']['hold_last'],ls='--',color='#555555',label='保持最后位置')
    axes[1].set_xlabel('预测时刻（秒）');axes[1].set_ylabel('该时刻中心误差（像素）');axes[1].set_title('预测越远，误差如何变化',fontsize=12);axes[1].legend(fontsize=9,ncol=2)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('六组 Transformer：相同数据与训练预算下的比较',fontsize=17)
    fig.tight_layout(rect=[0,0,1,.94])
    for ext in ['png','pdf','svg']:fig.savefig(PRES/f'results_overview.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,9));ax.set_xlim(0,14);ax.set_ylim(0,9);ax.axis('off')
    ax.text(.2,8.75,'相机与行人信息：六种进入预测网络的方式',fontsize=22,weight='bold')
    ax.text(.2,8.25,'历史 6 点 / 0.5 秒 → 未来 15 点 / 1.5 秒；64 维、4 头；约 21 万可训练参数',fontsize=12,color='#536171')
    steps=[['原始行人 + 相机特征','拼接 → 联合 self-attention','公共轨迹输出层','未来图像位置'],
           ['原始行人','行人 self-attention','公共轨迹输出层','未来图像位置'],
           ['旋转补偿后行人','行人 self-attention','公共层 → 残差位置','独立相机 → 重投影'],
           ['原始行人 / 相机','各自 self-attention','双向 cross → 融合','公共层 → 图像位置'],
           ['原始行人 / 相机','各自 self-attention','行人 query 相机','公共层 → 图像位置'],
           ['补偿后行人 / 相机','各自 self-attention','行人 query → 残差','独立相机 → 重投影']]
    for r,(v,ss,c) in enumerate(zip(VARIANTS,steps,COLORS)):
        y=6.95-r*1.03;ax.text(.2,y+.21,v[0],fontsize=22,weight='bold',color=c)
        for j,text in enumerate(ss):
            x=.9+j*3.18;box=FancyBboxPatch((x,y),2.88,.66,boxstyle='round,pad=0.09',facecolor=c+'18',edgecolor=c,lw=1.3);ax.add_patch(box)
            ax.text(x+1.44,y+.33,text,fontsize=11.2,ha='center',va='center')
            if j<3:ax.annotate('',(x+3.07,y+.33),(x+2.98,y+.33),arrowprops=dict(arrowstyle='->',color='#677583',lw=1.3))
    ax.text(.2,.93,'相机特征 = 历史旋转 + 冻结相机 GRU 的历史隐状态；cross-attention 只读取历史。',fontsize=11.5)
    ax.text(.2,.52,'C/F：同一冻结相机模型预测未来旋转，再做固定几何重投影；F 不再是严格独立的行人分支。',fontsize=11.5)
    ax.text(.2,.11,'C/F 与直接输出组还存在坐标、监督目标、合成方式的差异；不将组间全部差异归因于 attention。',fontsize=11,color='#536171')
    for ext in ['png','pdf','svg']:fig.savefig(PRES/f'architecture_six_groups.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Scientific figures saved')


if __name__=='__main__':main()
