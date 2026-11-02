from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')


def main():
    s=read(OUT/'summary.json');h=read(OUT/'horizon_errors.json');p=OUT/'presentation';p.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    names=['hold','old_original','coupled_dual/full','separated_dual/full','separated_future/full']
    labels=['保持最后位置','保存的旧版','匹配耦合对照','分离双任务','分离只预测']
    colors=['#888888','#8973ab','#b48038','#267e94','#439b72']
    vals=[s['main'][n]['window_mean']['ADE15'] for n in names]
    std=[np.std([v['ADE15'] for v in s['main'][n]['per_seed'].values()]) for n in names]
    ax=axes[0,0];ax.bar(range(len(names)),vals,color=colors,yerr=std,capsize=4)
    ax.set_xticks(range(len(names)),labels,fontproperties=FONT,fontsize=9);ax.set_ylabel('ADE (pixels)');ax.set_ylim(0,205)
    ax.set_title('完整预测：117 个窗口，误差条为三个种子的标准差',fontproperties=FONT,fontsize=12)
    for i,v in enumerate(vals):ax.text(i,v+std[i]+3,f'{v:.1f}',ha='center',fontsize=10)
    ax=axes[0,1]
    for n,label,color in zip(names[:4],labels[:4],colors[:4]):ax.plot(np.arange(1,16)/10,h[n],label=label,color=color,linewidth=2)
    ax.set_xlabel('Forecast horizon (s)');ax.set_ylabel('Mean center error (pixels)');ax.grid(alpha=.2);ax.legend(prop=FONT,fontsize=9)
    ax.set_title('分离版终点误差降低，但平均误差尚未超过旧版',fontproperties=FONT,fontsize=12)
    ax=axes[1,0];names2=['coupled_dual','separated_dual','separated_future'];labels2=['匹配耦合对照','分离双任务','分离只预测']
    ordinary=[s['main'][n+'/full']['window_mean']['ADE15'] for n in names2]
    oracle=[s['main'][n+'/reference']['window_mean']['ADE15'] for n in names2]
    x=np.arange(3);ax.bar(x-.18,ordinary,.36,color='#b48038',label='预测未来旋转：正常推理');ax.bar(x+.18,oracle,.36,color='#267e94',label='参考未来旋转：诊断')
    ax.set_xticks(x,labels2,fontproperties=FONT);ax.set_ylabel('Image ADE (pixels)');ax.set_ylim(0,210)
    for xx,a,b in zip(x,ordinary,oracle):ax.text(xx-.18,a+3,f'{a:.1f}',ha='center');ax.text(xx+.18,b+3,f'{b:.1f}',ha='center')
    ax.legend(prop=FONT,fontsize=9);ax.set_title('固定行人输出，仅替换最终投影的相机旋转',fontproperties=FONT,fontsize=12)
    ax=axes[1,1];rec=s['history'];values=[rec['separated_dual']['controlled_noise']['visible_input_MAE'],rec['separated_dual']['controlled_noise']['visible_reconstruction_MAE'],rec['coupled_dual']['controlled_noise']['visible_reconstruction_MAE']]
    ax.bar(range(3),values,color=['#888888','#267e94','#b48038']);ax.set_xticks(range(3),['带噪输入','分离版重建','耦合版重建'],fontproperties=FONT)
    ax.set_ylabel('Visible historical point error (pixels)');ax.set_ylim(0,48)
    for i,v in enumerate(values):ax.text(i,v+1,f'{v:.1f}',ha='center')
    ax.set_title('历史重建尚未改善带噪观测',fontproperties=FONT,fontsize=12)
    fig.suptitle('相机与行人分离预测第一版 · 旋转分离，尚未分离平移',fontproperties=FONT,fontsize=16)
    fig.savefig(p/'results_overview.png',dpi=160);fig.savefig(p/'results_overview.pdf');plt.close(fig)
    print('Saved results_overview.png and PDF')


if __name__=='__main__':main()
