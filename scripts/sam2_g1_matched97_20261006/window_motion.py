"""Preregistered operational G1 motion screen, applied before forecasts exist.

Thresholds are local screening priors, not a paper-defined G1 classifier.
Visual continuity/weak interaction and body coverage remain necessary.
"""
import numpy as np
def rotation_stats(R,seconds):
 def angle(a,b):
  r=np.swapaxes(a,-1,-2)@b
  return np.degrees(np.arccos(np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1)))
 speed=angle(R[:-1],R[1:])/np.diff(seconds)
 one=angle(R[:-10],R[10:])
 return dict(max_1s_rotation_deg=float(max(one)),omega_p95_deg_s=float(np.percentile(speed,95)))
def eligible(stats):
 return stats['max_1s_rotation_deg']<=20 and stats['omega_p95_deg_s']<=50
