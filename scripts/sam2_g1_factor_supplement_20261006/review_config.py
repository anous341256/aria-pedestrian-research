# Agent inspection of temporal RGB and mask crops. These are offline review
# decisions, not independent human ground truth. Raw outputs are preserved.
REVOKE={'60':{'N01':[2]},'65':{'N01':[20,21]}}
REPAIR={}
EXCLUDE={
 '58':{
  'N01':[(0,7,'Foreground person obscures lower body'),(40,47,'Foreground crossing obscures lower body')],
  'N04':[(18,66,'Rolling case persistently occludes lower body; unstable visible-mask extent')],
  'N07':[(35,45,'Incomplete lower-body mask during occlusion')],
 },
 '60':{
  'N01':[(0,4,'Blur and partial mask before clear observation')],
  'N02':[(19,21,'Sign occludes body'),(35,42,'Foreground/counter occludes lower body')],
 },
 '61':{
  'N01':[(3,22,'Counter occludes lower body')],
  'N02':[(38,79,'Counter occludes lower body')],
 },
 '62':{'N02':[(32,34,'Foreground person occludes target')]},
 '63':{'N05':[(20,26,'Foreground passer-by obscures lower body')]},
 '65':{
  'N01':[(18,25,'Severe blur and missing head in some masks')],
  'N04':[(51,63,'Foreground person occludes body')],
  'N05':[(60,67,'Partial body mask during foreground occlusion')],
 },
 '66':{'N04':[(28,44,'Foreground companion obscures part of lower body')]},
}
