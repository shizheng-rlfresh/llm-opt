This repo demos fine-tuning / training LLMs using `adaptive` algorithms, including `first-order` and `second-order` methods. 

### Current status:


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

### download project
```bash
# create python venv, e.g.,
python -m venv .venv
# clone git repository
git clone https://github.com/shizheng-rlfresh/llm-opt.git
# go to directory, and pip install required dependency
pip install -r requirements.txt
```


As we continue to expand the algorithms, we aim to provide easy and simple implementations and running examples on using **adaptive** algorithms.  

For `first-order` methods, we qualify an algorithm as `adaptive` if the tuning efforts of critical hyperparameters are nearly zero, such as **ones with no tuning required on learning rate** 

> e.g., `Adam` is not really a fully-adaptive algorithm ... simply because a global learning rate has to be supplied. It is known that one has to search for good learning-rate + some nice schedulers.

`Second-order` methods, in particular, `Hessian-free` methods, are known to exploit the loss lanscape with assistance of second-order information. With recent development in memory/cost-efficient fine-tuning mechanisms of LLMs, e.g., **LoRA** and **quantization**, it becomes possible to validate  "maybe we can try to use Hessian-free methods to fine-tune some LLMs?" 
