"""Run the identical pilot background check on all expanded candidates."""
import sys
from prepare import ROOT,OUT
sys.path.insert(0,str(ROOT/'scripts/factor_feasibility_20260928'))
import background
background.OUT=OUT
if __name__=='__main__':background.main()
