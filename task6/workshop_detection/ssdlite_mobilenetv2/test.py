
import torch, sys
print("Python:", sys.version.splitlines()[0])
print("torch:", getattr(torch, "__version__", None))
print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by torch:", torch.version.cuda)
