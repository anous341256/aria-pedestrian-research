"""Resource-bounded independent inference shard; identical frozen computation."""
import run
run.torch.cuda.set_per_process_memory_fraction(.43)
run.main()
