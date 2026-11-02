from shared import *
from run import *
def main():
 data=dict(np.load(DATA/'dataset.npz'));cfg=read(OUT/'protocol.json');fold=read(OUT/'folds.json')[2];features,rotations=camera_features(data,2,17)
 source=ROOT/'outputs/transformer_expanded_v2_20261005/fold_2/seed_17/A_early_fusion';ck=torch.load(source/'checkpoint.pt',map_location=DEVICE,weights_only=False)
 m=Predictor('A_early_fusion',ck['ff_width']).to(DEVICE);m.load_state_dict(ck['model']);m.eval();z=np.load(source/'predictions_reviewed.npz')
 idx=np.array(fold['test_indices']);pos=[list(z['indices']).index(i) for i in idx]
 a=infer(m,data,idx,features,rotations);b=infer(m,data,z['indices'],features,rotations)
 record=dict(subset_batch_max_px=float(np.max(np.abs(a['pred']-z['pred'][pos]))),original_batch_max_px=float(np.max(np.abs(b['pred']-z['pred']))),old_batch=len(z['indices']),new_batch=len(idx))
 write(VERSION/'batch_replay_diagnostic.json',record);print(record)
if __name__=='__main__':main()
