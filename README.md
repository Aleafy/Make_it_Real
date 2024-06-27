# <img src="assets/icon.png" style="vertical-align: -14px;" :height="50px" width="50px"> Make-it-Real

**[Make-it-Real: Unleashing Large Multimodal Model for Painting 3D Objects with Realistic Materials](https://arxiv.org/abs/2404.16829)**
</br>
[Ye Fang](https://github.com/Aleafy)\*,
[Zeyi Sun](https://github.com/SunzeY)\*,
[Tong Wu](https://wutong16.github.io/),
[Jiaqi Wang](https://myownskyw7.github.io/),
[Ziwei Liu](https://liuziwei7.github.io/),
[Gordon Wetzstein](https://web.stanford.edu/~gordonwz/),
[Dahua Lin](http://dahua.site/)

<p style="font-size: 0.6em; margin-top: -1em">*Equal Contribution</p>
<p align="center">
<a href="https://arxiv.org/abs/2404.16829"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://sunzey.github.io/Make-it-Real"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="https://www.youtube.com/watch?v=_j-t8592GCM"><img src="https://img.shields.io/static/v1?label=Demo&message=Video&color=orange"></a>
<a href="" target='_blank'>
<img src="https://visitor-badge.laobi.icu/badge?page_id=Aleafy.Make_it_Real&left_color=gray&right_color=blue">
</a>
</p>


![Demo](./assets/demo.gif)


## 📜 News
🚀 [2024/6/8] We release our [inference pipeline of Make-it-Real](#⚡-quick-start), including material matching and generation of albedo-only 3D objects.

🚀 [2024/6/8] [Material library annotations](#📦-data-preparation) generated by GPT-4V and [data engine](#⚡-quick-start) are released!

🚀 [2024/4/26] The [paper](https://arxiv.org/abs/2404.16829) and [project page](https://sunzey.github.io/Make-it-Real) are released!

## 💡 Highlights
- 🔥 We first demonstrate that **GPT-4V** can effectively **recognize and describe materials**, allowing our model to precisely identifies and aligns materials with the corresponding components of 3D objects.
- 🔥 We construct a **Material Library** containing thousands of materials with highly
detailed descriptions readily for MLLMs to look up and assign.
- 🔥 **An effective pipeline** for texture segmentation, material identification and matching, enabling the high-quality application of materials to
3D assets.

## 👨‍💻 Todo
- [ ] Evaluation for Existed and Model-Generated Assets (both code & test assets)
- [ ] More Interactive Demos (huggingface, jupyter) 
- [x] Make-it-Real Pipeline Inference Code
- [x] Highly detailed Material Library annotations (generated by GPT-4V) 
- [x] Paper and Web Demos

## 💾 Installation
Prepare basic modules for deep learning 3d modeling tool(kaolin), rendering engine(blender), and segmentation model. See details in [INSTALL.md](INSTALL.md).


## 📦 Data Preparation
 1. **Annotations**: in `data/material_lib/annotations` [folder](data/material_lib/annotations), include:
    - Highly-detailed descriptions by GPT-4V: offering thorough descriptions of the material’s visual characteristics and rich semantic information.
    - Category-tree: Divided into a hierarchical structure with coarse and fine granularity, it includes over 80 subcategories.
 2. **PBR Maps**: You can download the complete PBR data collection at [Huggingface](https://huggingface.co/datasets/gvecchio/MatSynth/tree/main), or download the data used in our project at [OpenXLab](https://openxlab.org.cn/datasets/YeFang/MatSynth/tree/main) (Recommended).
 3. **Material Images(optinal)**: You can download the material images file [here](https://drive.google.com/file/d/1ob7CV6JiaqFyjuCzlmSnBuNRkzt2qMSG/view?usp=sharing), to check and visualize the material appearance.

<pre>
Make_it_Real
└── data
    └── material_lib
        ├── annotations
        ├── mat_images
        └── pbr_maps
</pre>



## ⚡ Quick Start
#### Inference
```bash
python main.py --obj_dir <object_dir> --exp_name <unique_exp_name> --api_key <your_own_gpt4_api_key>
```
- To ensure proper network connectivity for GPT-4V, add proxy environment settings in [main.py](https://github.com/Aleafy/Make_it_Real/blob/feb3563d57fbe18abbff8d4abfb48f71cc8f967b/main.py#L18) (optional). Also, please verify the reachability of your [API host](https://github.com/Aleafy/Make_it_Real/blob/feb3563d57fbe18abbff8d4abfb48f71cc8f967b/utils/gpt4_query.py#L68).
- Result visualization (blender engine) is located in the `output/refine_output` dir. You can compare the result with that in `output/ori_output`. 

#### Annotation Engine

```bash
cd scripts/gpt_anno
python gpt4_query_mat.py
```
Note: Besides functinoning as annotation engine, you can also use this code ([gpt4_query_mat.py](https://github.com/Aleafy/Make_it_Real/blob/main/scripts/gpt_anno/gpt4_query_mat.py)) to test the GPT-4V connection simply.

<!-- [annotation code](scripts/gpt_anno) -->
<!-- #### Evalutation -->



## ❤️ Acknowledgments
- [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth/tree/main): a Physically Based Rendering (PBR) materials dataset, which offers extensive high-resolusion tilable pbr maps to look up.
- [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper): Wonderful text-guided texture generation model, and the codebase we built upon.
- [SoM](https://som-gpt4v.github.io/): Draw visual cues on images to facilate GPT-4V query better.
- [Material Palette](https://github.com/astra-vision/MaterialPalette): Excellent exploration of material extraction and generation, offers good insights and comparable setting.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@misc{fang2024makeitreal,
      title={Make-it-Real: Unleashing Large Multimodal Model for Painting 3D Objects with Realistic Materials}, 
      author={Ye Fang and Zeyi Sun and Tong Wu and Jiaqi Wang and Ziwei Liu and Gordon Wetzstein and Dahua Lin},
      year={2024},
      eprint={2404.16829},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
