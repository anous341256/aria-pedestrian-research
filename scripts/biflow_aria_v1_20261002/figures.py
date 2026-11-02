from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
def main():
    p=OUT/'presentation';p.mkdir(exist_ok=True);s=read(OUT/'summary.json');h=read(OUT/'history_reconstruction.json')['summary'];names=['raw_future','raw_dual','comp_future','comp_dual'];labels=['原始／只预测','原始／双分支','补偿／只预测','补偿／双分支']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    vals=[s['reviewed'][n]['window_mean']['ADE15'] for n in names];std=[np.std([r['ADE15'] for r in s['reviewed'][n]['per_seed'].values()]) for n in names];colors=['#6886a6','#b998cb','#52a7a0','#246b63']
    axes[0].bar(range(4),vals,color=colors,yerr=std,capsize=5);axes[0].axhline(s['reviewed']['hold']['window_mean']['ADE15'],color='#b75638',linestyle='--',label='Last position')
    axes[0].set_xticks(range(4),labels,fontproperties=FONT,fontsize=10);axes[0].set_ylabel('ADE (pixels)');axes[0].set_title('1.5 秒预测：三个训练种子的均值与标准差',fontproperties=FONT)
    for j,v in enumerate(vals):axes[0].text(j,v+std[j]+5,f'{v:.1f}',ha='center')
    axes[0].legend();axes[0].set_ylim(0,330)
    hh=h['controlled_noise'];rv=[hh['raw_dual']['visible_input_MAE'],hh['raw_dual']['reconstruction_visible_MAE'],hh['comp_dual']['reconstruction_visible_MAE']];axes[1].bar(range(3),rv,color=['#b75638','#b998cb','#246b63']);axes[1].set_xticks(range(3),['带噪输入','原始坐标重建','补偿坐标重建'],fontproperties=FONT,fontsize=10);axes[1].set_ylabel('Historical visible-point error (pixels)');axes[1].set_title('历史修复：重建误差仍高于原输入',fontproperties=FONT)
    for j,v in enumerate(rv):axes[1].text(j,v+1,f'{v:.1f}',ha='center')
    axes[1].set_ylim(0,80);fig.savefig(p/'results_overview.png',dpi=160);fig.savefig(p/'results_overview.pdf');plt.close(fig)
    # A fixed diagnostic window; all displayed methods use seed 17.
    selection=read(p/'selection.json');key=selection['selected_reconstruction_failure'];meta=read(OUT/'samples.json');i=next(i for i,m in enumerate(meta) if m['key']==key);fold=next(f for f in read(OUT/'folds.json') if i in f['test_indices']);d=dict(np.load(OUT/'dataset.npz'));fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,name in enumerate(['raw_dual','comp_dual']):
        z=np.load(OUT/f'fold_{fold["fold"]}'/'seed_17'/name/'predictions_controlled_noise.npz');j=list(z['indices']).index(i);time=d['seconds'][i,:P];valid=z['input_valid'][j].astype(bool)
        for row in range(2):
            ax=axes[row,col];ax.plot(time,d['xy'][i,:P,row],'o-',color='black',label='Reviewed reference');ax.plot(time[valid],z['input_xy'][j,valid,row],'x',markersize=9,color='#b75638',label='Noisy observed');ax.plot(time,z['reconstruction'][j,:,row],'o--',color='#246b63',label='Reconstructed');ax.set_xlabel('History time (s)');ax.set_ylabel(['x (pixels)','y (pixels)'][row]);ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[0,col].set_title(['原始坐标双分支','旋转补偿双分支'][col],fontproperties=FONT)
    fig.suptitle(f'{key}：事后选取的历史重建失败示例，种子 17',fontproperties=FONT);fig.savefig(p/'history_reconstruction_failure.png',dpi=150);plt.close(fig)
    print('Figures saved')
if __name__=='__main__':main()
