# Value Gradient Sampler (VGS)
This is the official repository for the paper [Value Gradient Sampler: Sampling as Sequential Decision Making](https://www.arxiv.org/abs/2502.13280)


## Environment Setup
We recommend using Conda for environment setup. Follow these steps:
```bash
#clone project
git clone https://github.com/swyoon/value-gradient-sampler.git
cd value-gradient-sampler

#create environment using python
conda create -n vgs python=3.10 -y
conda activate vgs

# install pytorch and dependencies
pip install -r requirements.txt
```

## Training VGS
In our paper, we conduct experiments on **sampling from synthetic distributions (GMM & Funnel)** and **particle systems (DW-4, LJ-13)**.  
Details about each experiment can be found in the paper.  


To train the sampler on different distributions, use the following commands:  

- GMM 
    ```bash
    python train_sampler.py --config configs/gmm.yaml --device 0 --run YOUR_RUN_NAME --exp_num 0
    ```
- Funnel
    ```bash
    python train_sampler.py --config configs/funnel.yaml --device 0 --run YOUR_RUN_NAME --exp_num 0
    ```
- DW-4 
    ```bash
    python train_sampler.py --config configs/dw4.yaml --device 0 --run YOUR_RUN_NAME --exp_num 0
    ```
- LJ-13  
    ```bash
    python train_sampler.py --config configs/lj13.yaml --device 0 --run YOUR_RUN_NAME --exp_num 0
    ```

You can analyze the experimental results using **Weights & Biases (wandb).** 

## Evaluation  
To reproduce the metrics reported in our paper, please refer to the evaluation notebooks located in `notebooks/`.  
In these notebooks, you will need to specify the path to the model checkpoint, which can be found in `checkpoints/`.  


## Citation
If you find this repository useful for your research, please cite our paper:
```
@misc{yoon2025valuegradientsamplersampling,
      title={Value Gradient Sampler: Sampling as Sequential Decision Making}, 
      author={Sangwoong Yoon and Himchan Hwang and Hyeokju Jeong and Dong Kyu Shin and Che-Sang Park and Sehee Kwon and Frank Chongwoo Park},
      year={2025},
      eprint={2502.13280},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.13280}, 
}
```
