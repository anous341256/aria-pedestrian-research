"""Post-run training/validation diagnostics only; never selects or tunes models."""
from common import *
from run import load,Predictor,torch,predict
def main():
    assert read(OUT/"training_complete.json")["passed"]
    d,inputs=load();rows=[]
    for arm in ARMS:
        for fold in read(OUT/"folds.json"):
            for s in [17,29,43]:
                dest=OUT/arm/f'fold_{fold["fold"]}/seed_{s}'
                ck=torch.load(dest/"checkpoint.pt",map_location="cuda",weights_only=False)
                model=Predictor().cuda();model.load_state_dict(ck["model"]);model.eval()
                idx=np.array(fold["train_indices"])
                image=predict(model,inputs,idx,arm)*2560+d["xy"][idx,15,None]
                ade=float(np.linalg.norm(image-d["xy"][idx,16:21],axis=-1).mean())
                hold=float(np.linalg.norm(d["xy"][idx,15,None]-d["xy"][idx,16:21],axis=-1).mean())
                log=read(dest/"training.json");selected=next(x for x in log if x["step"]==ck["step"])
                rows.append(dict(arm=arm,fold=fold["fold"],seed=s,step=ck["step"],train_ADE=ade,train_hold_ADE=hold,
                  validation_ADE=selected["validation_ADE"],validation_hold_ADE=log[0]["validation_ADE"]))
    result={}
    for arm in ARMS:
        rr=[r for r in rows if r["arm"]==arm]
        result[arm]={k:float(np.mean([r[k] for r in rr])) for k in ["train_ADE","train_hold_ADE","validation_ADE","validation_hold_ADE"]}
        result[arm]["selected_step0"]=sum(r["step"]==0 for r in rr)
        result[arm]["trained_selections"]=sum(r["step"]>0 for r in rr)
    write(OUT/"learning_diagnostics.json",dict(passed=True,fit_equal_averages=result,rows=rows,
      note="Train/validation counts and recording composition differ from held-out aggregate; diagnostic only. Validation is optimistically selected. No extra optimization or test-driven selection."))
    print(result,flush=True)
if __name__=="__main__":main()
