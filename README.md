# DiffMorphGrasp [CoRl 2026]

**DiffMorphGrasp: A Diffusion Framework Conditioned on Hand Morphology for Cross-Embodiment Grasp Generation**

<!-- 
[Zhiyuan Wu](https://georgewuzy.github.io/)<sup>1</sup>,
[Xiangyu Zhang](https://orcid.org/0009-0000-8690-514X)<sup>1</sup>,
[Zhuo Chen](https://zhuochenn.github.io/)<sup>1</sup>,
[Jiankang Deng](https://jiankangdeng.github.io/)<sup>2</sup>,
[Rolandos Alexandros Potamias](https://rolpotamias.github.io/)<sup>2</sup>,
[Shan Luo](https://shanluo.github.io/)<sup>1</sup>

<sup>1</sup> King's College London · <sup>2</sup> Imperial College London -->

[Paper](https://arxiv.org/pdf/2602.00915) · [arXiv](https://arxiv.org/abs/2602.00915) · [Project](https://georgewuzy.github.io/DiffMorphGrasp-website/)

![DiffMorphGrasp pipeline](assets/pipeline.png)

A diffusion framework conditioned on hand morphology for cross-embodiment grasp generation.

This release supports the three seen hands **Barrett, Allegro, and ShadowHand** and two unseen hands **Leap Hand and Inspire Hand** inference.

## Dependencies

Create a conda environment. The following commands use CUDA 12.4:

```bash
git clone https://github.com/GeorgeWuzy/DiffMorphGrasp.git
cd DiffMorphGrasp
conda create -n diffmorphgrasp python=3.8 pip
conda activate diffmorphgrasp
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```

Install PyTorch3D and the remaining Python dependencies:

```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8'
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/Silverster98/pointops.git@86c68b1e63092152b4ab14ff09b832c5d2d0dc97'
```

For grasp evaluation, download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym/download) from NVIDIA and install it separately:

```bash
pip install --no-deps -e /path/to/isaacgym/python
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Building PyTorch3D and pointops requires a CUDA toolkit and C++ compiler compatible with the installed PyTorch version.

## Data and checkpoint

Download the processed dataset from [OneDrive link](https://emckclac-my.sharepoint.com/:u:/g/personal/k24021603_kcl_ac_uk/IQDViMqWhPuET6V92Ujv1GsMASTW27e2b6qjGgO65kVU0aE?e=3b2wns) and the pretrained checkpoint from [OneDrive link](https://emckclac-my.sharepoint.com/:u:/g/personal/k24021603_kcl_ac_uk/IQA-QTOGWy5rQbUDSfLSQ8iiAYmxRHZUellwT3Ctq5sEgcs?e=flJVVZ). Place them as follows:

```text
checkpoints/pretrained.pth
data/MultiDex_filtered/
  object_pcds_nors.pkl
  allegro/allegro_new.pt
  barrett/barrett_new.pt
  shadowhand/shadowhand_new.pt
  object/<dataset>/<object>/<object>.stl
  object/<dataset>/<object>/<object>.urdf
```

The hand URDFs and meshes are included in `data/urdf/`. The object URDFs and
their referenced `.stl` meshes must be downloaded with the dataset. The default paths are configured in `configs/default.yaml`.

## Grasp Generation Inference

```bash
python sample.py
```

To select one hand or checkpoint:
```bash
python sample.py 'robot_name=[barrett]' checkpoint=checkpoints/pretrained.pth
python sample.py 'robot_name=[leaphand]'
python sample.py 'robot_name=[inspire]'
```

For a single-object run, add `'objects=[contactdb+apple]'`.
Set `penetration_optimization=true` to optimize hand-object penetration.

## Evaluate

```bash
bash scripts/evaluate_all.sh \
  outputs/samples/YOUR_TIMESTAMP
```

To evaluate one hand:
```bash
python scripts/evaluate.py \
  --input outputs/samples/YOUR_TIMESTAMP/barrett \
  --robot barrett
```

## Visualize

```bash
python scripts/visualize_grasp.py \
  --input outputs/samples/YOUR_TIMESTAMP/barrett/contactdb+apple.pt \
  --robot barrett --index 0 --output outputs/visualizations/apple.html
```

## Training

```bash
python train.py exp_dir=outputs/train_baseline
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

If you find this work helpful, please consider citing us using the following BibTeX entry:

```bibtex  
@article{wu2026unimorphgrasp,
  title={UniMorphGrasp: Diffusion Model with Morphology-Awareness for Cross-Embodiment Dexterous Grasp Generation},
  author={Wu, Zhiyuan and Zhang, Xiangyu and Chen, Zhuo and Deng, Jiankang and Potamias, Rolandos Alexandros and Luo, Shan},
  journal={arXiv preprint arXiv:2602.00915},
  year={2026}
}
```

## Contact

If you have any questions, feel free to contact me through email at [zhiyuan.1.wu@kcl.ac.uk](mailto:zhiyuan.1.wu@kcl.ac.uk).
