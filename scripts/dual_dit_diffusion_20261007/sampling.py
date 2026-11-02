"""Scene-shared camera randomness. Same history gives same draws across people."""
import hashlib
import numpy as np
import torch
from experiment import infer as ordinary_infer, fixed_noise, camera_keys, tensor, DEVICE

def infer(model,history,origin=None,noise=None,eta=0.,steps=20,rngseed=20261007):
    if not (eta and model.coordinates==3 and origin is None):
        return ordinary_infer(model,history,origin,noise,eta,steps,rngseed)
    model.eval()
    _,first,inverse=np.unique(np.asarray(history).reshape(len(history),-1),axis=0,return_index=True,return_inverse=True)
    keys=camera_keys(history)
    if noise is None:noise=fixed_noise(keys,3)
    result=[]
    for i in first:
        random_seed=int(hashlib.sha256((keys[i]+':innovations:'+str(rngseed)).encode()).hexdigest()[:15],16)
        generator=torch.Generator(device=DEVICE).manual_seed(random_seed)
        with torch.no_grad():
            y=model.sample(tensor(history[i:i+1]),tensor(noise[i:i+1]),steps=steps,eta=eta,generator=generator)
        result.append(y.cpu().numpy()[0])
    return np.stack(result)[inverse]
