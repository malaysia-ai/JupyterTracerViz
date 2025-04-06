# JupyterTracerViz

Visualize trace.json in Jupyter Notebook cell.

## Installation

```
pip3 install git+https://github.com/huseinzol05/JupyterTracerViz
```

## how to

I take example profiling from https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html,

```python3
import torch
import jupytertracerviz
from torchvision.models import resnet18

model = resnet18().cuda()
inputs = [torch.randn((5, 3, 224, 224), device='cuda') for _ in range(10)]

model_c = torch.compile(model)

def fwd_bwd(inp):
    out = model_c(inp)
    out.sum().backward()

# warm up
fwd_bwd(inputs[0])

with torch.profiler.profile() as prof:
    for i in range(1, 4):
        fwd_bwd(inputs[i])
        prof.step()

prof.export_chrome_trace("trace.json")
jupytertracerviz.visualize("trace.json", height = "800")
```

<img width="50%" src="pic1.png">