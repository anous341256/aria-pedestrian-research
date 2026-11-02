import screen_short as s
from pathlib import Path
s.RECORDS=[sid for sid,rec in s.base.INV.items() if sid not in s.RECORDS and rec['mps_source'] and Path(rec['mps_source']).exists()]
# Keep first search record immutable; use a new output name via temporary wrapper.
old=s.OUT/'screen_short_candidates.json'
snapshot=old.read_bytes()
s.main()
(s.OUT/'screen_broader_candidates.json').write_bytes(old.read_bytes())
old.write_bytes(snapshot)
