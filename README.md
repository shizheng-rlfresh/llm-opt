### ✨ Fine-tuning LLMs with `LoRA` and `Hessian-free` optimizers!

Demo Notebooks available

| Optimizer | Notebooks | Optimizer Implementation | Citation
|-------------------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------|---------------------------------------------------------|
| **Trust Region Newton-CG** | [gpt2-LoRA-TRCG](./examples/gpt2/gpt2-LoRA-TRCG.ipynb) | [optim/trcg.py](./optim/trcg.py) | [1][2]
| **AdaHessian** | [gpt2-LoRA-AdaHessian](./examples/gpt2/gpt2-LoRA-TRCG.ipynb) | [torch-optimizer](https://github.com/jettify/pytorch-optimizer?tab=readme-ov-file#adahessian) | [1][2]
| **OASIS** | [sdsf]() | [sdsf] | [1][2]
 
### Current status:
- A simple implementation of `Trust Region Newton-CG`, aka, `TRCG`; see [optim/trcg.py](https://github.com/shizheng-rlfresh/llm-opt/blob/main/optim/trcg.py) for details. TRCG is <ins>**Hessian-free** with only Hessian-vector product is needed, and no Hessian!</ins>. As LoRA brings down the side of models, ✨ let's give it a shot! 💪 
    - `Trust Region Newton-CG` is probably <ins>**the most underrated**</ins>  😖 optimizer in machine learning field. It is one of the best optimizers for solving nonconvex problems, e.g., deep neural networks. 
    - Coupled with `preconditioning`, `Trust Region Newton-CG` could yield even more promosing convergence property than most optimizers. 
    - A BIG UNKNOWN - its convergence in general `stochastic` setting is yet to proved ... that means, mini-batch training is not theoretically proved yet.
    - A BIG BUT - I loved it, and I can show many uses cases using naive `TRCG` to train DNN, e.g., CNN, GNN, etc. 

- Benchmark results of `TRCG` vs. `AdamW`, is shown here
<div style="display: flex; justify-content: center; margin-bottom: 20px;">
    <img src="./static/gpt2/trcg_gpt2_loss.jpg" alt="Image 1" style="width: 100%;">
</div>


### Ongoing stuff:


### download project
```bash
# create python venv, e.g.,
python -m venv .venv
source .venv/bin/activate
# clone git repository
git clone https://github.com/shizheng-rlfresh/llm-opt.git
# go to directory, and pip install required dependency
pip install -r requirements.txt
```










As we continue to expand the algorithms, we aim to provide easy and simple implementations and running examples on using **adaptive** algorithms.  

For `first-order` methods, we qualify an algorithm as `adaptive` if the tuning efforts of critical hyperparameters are nearly zero, such as **ones with no tuning required on learning rate** 

> e.g., `Adam` is not really a fully-adaptive algorithm ... simply because a global learning rate has to be supplied. It is known that one has to search for good learning-rate + some nice schedulers.

`Second-order` methods, in particular, `Hessian-free` methods, are known to exploit the loss lanscape with assistance of second-order information. With recent development in memory/cost-efficient fine-tuning mechanisms of LLMs, e.g., **LoRA** and **quantization**, it becomes possible to validate  "maybe we can try to use Hessian-free methods to fine-tune some LLMs?" 

### Requirement:
```bash
# main packages
pytorch==2.2.2
transformers==4.39.3
bitsandbytes==0.43.0 
peft==0.10.0

# system
os==linux # tested on RHEL, should work for most Linux dist
python>=3.10 
cuda==12.x # 11.8 or above should all work fine
GPU: T4, V100, etc w/ 16GB # at most as old as these guys
```