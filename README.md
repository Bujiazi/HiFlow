<div align="center">

<h1>HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance</h1>

<div>
    <a href="https://bujiazi.github.io/" target="_blank">Jiazi Bu*</a><sup></sup> | 
    <a href="https://github.com/LPengYang/" target="_blank">Pengyang Ling*</a><sup></sup> | 
    <a href="https://github.com/YujieOuO" target="_blank">Yujie Zhou*</a><sup></sup> | 
    <a href="https://panzhang0212.github.io/" target="_blank">Pan Zhang<sup>†</sup></a><sup></sup> | 
    <a href="https://wutong16.github.io/" target="_blank">Tong Wu</a><sup></sup> <br>
    <a href="https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en/" target="_blank">Xiaoyi Dong</a><sup></sup> |
    <a href="https://yuhangzang.github.io/" target="_blank">Yuhang Zang</a><sup></sup> |
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=sJkqsqkAAAAJ" target="_blank">Yuhang Cao</a><sup></sup> |
    <a href="http://dahua.site/" target="_blank">Dahua Lin</a><sup></sup> |
    <a href="https://myownskyw7.github.io/" target="_blank">Jiaqi Wang<sup>†</sup></a><sup></sup>
</div>
<br>
<div>
    <sup></sup>Shanghai Jiao Tong University, University of Science and Technology of China, <br> The Chinese University of Hong Kong, Shanghai Artificial Intelligence Laboratory
</div>
(*<b>Equal Contribution</b>)(<sup>†</sup><b>Corresponding Author</b>)
<br><br>

[![arXiv](https://img.shields.io/badge/arXiv-2504.06232-b31b1b.svg)](https://arxiv.org/abs/2504.06232) 
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://bujiazi.github.io/hiflow.github.io/)

---


<strong>HiFlow is a training-free and model-agnostic framework to unlock the resolution potential of pre-trained flow models.</strong>

<details><summary>📖 Click for the full abstract of HiFlow</summary>

<div align="left">

> Text-to-image (T2I) diffusion/flow models have drawn considerable attention recently due to their remarkable ability to deliver flexible visual creations. Still, high-resolution image synthesis presents formidable challenges due to the scarcity and complexity of high-resolution content. To this end,  we present **HiFlow**, a training-free and model-agnostic framework to unlock the resolution potential of pre-trained flow models. Specifically, HiFlow establishes a virtual reference flow within the high-resolution space that effectively captures the characteristics of low-resolution flow information, offering guidance for high-resolution generation through three key aspects: initialization alignment for low-frequency consistency, direction alignment for structure preservation, and acceleration alignment for detail fidelity. By leveraging this flow-aligned guidance, HiFlow substantially elevates the quality of high-resolution image synthesis of T2I models and demonstrates versatility across their personalized variants. Extensive experiments validate HiFlow's superiority in achieving superior high-resolution image quality over current state-of-the-art methods.
</details>
</div>

## 🎨 Gallery
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="__assets__/hiflow_teaser.png">
</div>
<br>

<div align="center">
👁️ For more visual results, go checkout our <a href="https://bujiazi.github.io/hiflow.github.io/" target="_blank">Project Page</a>.
</div>

</div>



## 💻 Overview
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="__assets__/hiflow_pipeline.png">
</div>
<br>

HiFlow constructs reference flow from low-resolution sampling trajectory to offer initiation alignment, direction alignment, and acceleration alignment, enabling flow-aligned high-resolution image generation. Specifically, HiFlow involves a cascade generation paradigm: First, a virtual reference flow is constructed in the high-resolution space based on the step-wise estimated clean samples of the low-resolution sampling flow. Then, during high-resolution synthesizing, the reference flow offers guidance from sampling initialization, denoising direction, and moving acceleration, aiding in achieving consistent low-frequency patterns, preserving structural features, and maintaining high-fidelity details.

## 🔧 Installations
### Setup repository and conda environment

```bash
git clone https://github.com/Bujiazi/HiFlow.git
cd HiFlow

conda create -n hiflow python=3.10
conda activate hiflow

pip install -r requirements.txt
```
### (Optional) Prepare LoRA models
HiFlow can be seamlessly integrated with various LoRA models. 

<table class="center">
    <tr>
    <td><img src="__assets__/aidmafluxpro_1.png"></td>
    <td><img src="__assets__/aidmafluxpro_2.png"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/832683/flux-pro-11-style-lora-extreme-detailer-for-flux-illustrious">aidmaFLUXPro</a> (More and Finer Details)</p> 

<table class="center">
    <tr>
    <td><img src="__assets__/hyrea_1.png"></td>
    <td><img src="__assets__/hyrea_2.png"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/939882/realistic-hyrea-flux-lora">Realistic HyRea</a> (Hyper Realistic Style)</p> 

<table>
    <tr>
    <td><img src="__assets__/wukong_1.png"></td>
    <td><img src="__assets__/wukong_2.png"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/681691/black-myth-wukong-flux">Black Myth Wukong</a> (T2I Customization)</p>




## 🎈 Quick Start
### Perform high-resolution image generation with Flux.1.0-dev
```bash
python run_hiflow.py
```
Model downloading is automatic.


## 🖋 News
- Support LoRA! (2025.5.11)
- Code (V1.0) and project page are released! (2025.4.17)
- Paper is available on arXiv! (2025.4.8)

## 🏗️ Todo
- [x] 🚀 Release the HiFlow code and project page
- [x] 🚀 Release paper

## 📎 Citation 

If you find our work helpful, please consider giving a star ⭐ and citation 📝 
```bibtex
@article{bu2025hiflow,
  title={HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance},
  author={Bu, Jiazi and Ling, Pengyang and Zhou, Yujie and Zhang, Pan and Wu, Tong and Dong, Xiaoyi and Zang, Yuhang and Cao, Yuhang and Lin, Dahua and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2504.06232},
  year={2025}
}
```

## 📣 Disclaimer

This is official code of HiFlow.
All the copyrights of the demo images and audio are from community users. 
Feel free to contact us if you would like remove them.

## 💞 Acknowledgements
The code is built upon the below repositories, we thank all the contributors for open-sourcing.
* [Flux](https://github.com/black-forest-labs/flux)
* [I-Max](https://github.com/PRIS-CV/I-Max)
* [DiffuseHigh](https://github.com/yhyun225/DiffuseHigh)


