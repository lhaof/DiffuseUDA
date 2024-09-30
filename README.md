# DiffuseUDA
This respository is the official implimentation of the paper <!--[Diffuse-UDA](https://arxiv.org/pdf/2408.05985):--> Diffuse-UDA: Addressing Unsupervised Domain Adaptation in Medical Image Segmentation with Appearance and Structure Aligned Diffusion Models. This work introduce a effective method that reaches the comparable or exceeds the upper bound on the unsupervised domain adaptation task.

# Deployment
## Environment
Please refer to ``requirement.txt``.

## Data
All data or check points related to this project are available at [Google Drive](https://drive.google.com/drive/folders/1rdyNVFMCaFfRnXj3yAZVLo4koKpEgKMc?usp=drive_link).

## Training & Validation
Folder seg-xxx indicate the corresponding segmentation task on xxx dataset. Folder gen indicates ours diffuse-uda for generation. For training, you should follows [Diffuse-UDA](https://arxiv.org/pdf/2408.05985) that first generate the pseudo label on the target domain training data, then mix them with the source domain data to train the diffusion model, and use the label from source domain and pseudo label from target domain to sample the cases for training. Finally, you may use the generated data and source+target domain data to train the segmentation model again.

Please refer to ``xxx.sh'' inside the folders. 

<!--# Cite
If you find this code useful for your research, please consider cite the following paper:
```
@article{gong2024diffuse,
  title={Diffuse-UDA: Addressing Unsupervised Domain Adaptation in Medical Image Segmentation with Appearance and Structure Aligned Diffusion Models},
  author={Gong, Haifan and Wang, Yitao and Wang, Yihan and Xiao, Jiashun and Wan, Xiang and Li, Haofeng},
  journal={arXiv preprint arXiv:2408.05985},
  year={2024}
}
```
-->
