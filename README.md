# CPD
PyTorch implementation for **Source-free domain adaptation with Class Prototype Discovery**. This repository is based on framework from [CAN](https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation) and modified part of the code. 

The installation can refer to CAN.

Please download datasets to experiments/datasets/ folder.

## Training:
```
CUDA_VISIBLE_DEVICES=X python tools/train_target.py --method CPD --cfg experiments/config/config_file.yaml --weights pre_trained_source_model_weights
```



## Citation
If you think our paper or code is helpful to you, we very much hope that you can cite our paper, thank you very much.

```
@article{zhou2024source,
  title={Source-free domain adaptation with Class Prototype Discovery},
  author={Zhou, Lihua and Li, Nianxin and Ye, Mao and Zhu, Xiatian and Tang, Song},
  journal={Pattern recognition},
  volume={145},
  pages={109974},
  year={2024},
  publisher={Elsevier}
}
```
