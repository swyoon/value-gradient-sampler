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


## **Training VGS**  
In our paper, we conduct experiments on **sampling from synthetic distributions (GMM & Funnel)** and **particle systems (DW-4, LJ-13)**.  
Details about each experiment can be found in the paper.  

To train the sampler on different distributions, use the following commands:  

- **GMM**  
    ```bash
    python train_sampler.py --config configs/gmm.yaml --device 0 --run RUN_NAME --exp_num 0
    ```
- **Funnel**  
    ```bash
    python train_sampler.py --config configs/funnel.yaml --device 0 --run RUN_NAME --exp_num 0
    ```
- **DW-4**  
    ```bash
    python train_sampler.py --config configs/dw4.yaml --device 0 --run RUN_NAME --exp_num 0
    ```
- **LJ-13**  
    ```bash
    python train_sampler.py --config configs/lj13.yaml --device 0 --run RUN_NAME --exp_num 0
    ```

The training configuration files, logs, and results are saved in the log directory:  
```
results/CONFIGFILE_NAME/RUN_NAME/ (e.g., 'results/gmm/test/')
```
You can analyze the experiment results using **Weights & Biases (wandb).**  


## **Evaluation**  
This section provides a brief overview of the evaluation metrics reported in our paper and explains how to obtain them using this repository. For a detailed description, please refer to our paper.  

### **Synthetic Distribution Experiments**  
For these experiments, we report the **Sinkhorn Distance ($\mathcal{W}^2_\gamma$)**, **Total Variation Distance - Energy (TVD-E)**, and **Average Standard Deviation Error Across the Marginals ($\Delta std$)**, evaluated using $10^5$ samples.  
These metrics are automatically recorded in the `results_{EXP_NUM}.txt` file, which is generated in the log directory after the training is completed.

### **Particle System Experiments** 
For these experiments, we report **Total Variation Distance - Energy & Interatomic Distance (TVD-E & TVD-D)** and **Wasserstein Distance ($\mathcal{W}^2$)**, evaluated using 2000 samples.  
The model checkpoint with the lowest validation TVD-E is automatically saved in the log directory during training. To evaluate the metrics using the saved checkpoint, use the evaluation notebooks located in notebooks/. You will need to set the variable `sampler_ckpt = PATH_TO_YOUR_CHECKPOINT`. By default, this variable is set to the provided model checkpoints in `checkpoints/`, which allows you to reproduce the exact metrics reported in our paper.


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
