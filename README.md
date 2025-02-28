# value-gradient-sampler
The official repository of ['Value Gradient Sampler: Sampling as Sequential Decision Making'](https://www.arxiv.org/abs/2502.13280)


## Environment
For environment setting we recommend the use of conda.
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

## Experiments
To run VGS experiment on Energy, e.g., GMM:

```
python train_sampler.py --config configs/gmm.yaml --device 0 --run dev --exp_num 0
```
## Reference

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
