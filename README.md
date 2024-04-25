# Make-it-Real

**[Make-it-Real: Unleashing Large Multimodal Model’s Ability for Painting 3D Objects with Realistic Materials](https://arxiv.org/abs/xxx)**
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
<a href="https://arxiv.org/abs/xxx"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://sunzey.github.io/Make-it-Real"><img src="https://img.shields.io/badge/Project-Website-red"></a>
</p>

Demo `Make-it-Real`: (Coming soon)
<!-- [![Hugging Face Spaces(Coming soon)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/xxx) -->


<video src="https://github.com/Aleafy/Make_it_Real/blob/main/assets/demo.mp4?raw=true" controls="controls">
您的浏览器不支持 video 标签。
</video>



## 📜 News
🚀 [2024/4/26] The [paper](https://arxiv.org/abs/xxx) and [project page](https://sunzey.github.io/Make-it-Real) are released!

## 💡 Highlights
- 🔥 **3.93%** improved zero-shot ImageNet classification accuracy when providing foreground alpha-map.
- 🔥 **Plug-in and play** with region focus in **any work** that use CLIP vision encoder.
- 🔥 **A strong visual encoder** as versatile tool when foreground mask is available.

## 👨‍💻 Todo
- [ ] Training code for Alpha-CLIP based on Open-CLIP
- [x] Evaluation code for Alpha-CLIP
- [x] Zero-shot evaluation for Imagenet-S Classification and REC tasks.
- [x] Web demo and local demo of Alpha-CLIP with LLaVA
- [x] Web demo and local demo of Alpha-CLIP with Stable Diffusion
- [x] Usage example notebook of Alpha-CLIP
- [x] Checkpoints of Alpha-CLIP

## ⚡ Quick Start

##   ⭐ Demos
<p align="center"> <a>  
<img src="./img/demo1.gif"  width="900" />
</a> </p>



## ❤️ Acknowledgments
- [CLIP](https://github.com/openai/CLIP): The codebase we built upon. Thanks for their wonderful work.
- [LAVIS](https://github.com/salesforce/LAVIS): The amazing open-sourced multimodality learning codebase, where we test Alpha-CLIP in [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and [BLIP-Diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion).
- [Point-E](https://github.com/openai/point-e): Wonderful point-cloud generation model, where we test Alpha-CLIP for 3D generation task.
- [LLaVA](https://github.com/haotian-liu/LLaVA): Wounderful MLLM that use CLIP as visual bacbone where we test the effectiveness of Alpha-CLIP.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@misc{sun2023alphaclip,
      title={Alpha-CLIP: A CLIP Model Focusing on Wherever You Want}, 
      author={Zeyi Sun and Ye Fang and Tong Wu and Pan Zhang and Yuhang Zang and Shu Kong and Yuanjun Xiong and Dahua Lin and Jiaqi Wang},
      year={2023},
      eprint={2312.03818},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of CLIP. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
