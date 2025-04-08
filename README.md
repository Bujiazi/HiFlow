<div align="center">

<h1>HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance</h1>

<div>
    <a href="https://bujiazi.github.io/" target="_blank">Jiazi Bu*</a><sup></sup> | 
    <a href="https://github.com/LPengYang/" target="_blank">Pengyang Ling*</a><sup></sup> | 
    <a href="https://github.com/YujieOuO" target="_blank">Yujie Zhou*</a><sup></sup> | 
    <a href="https://panzhang0212.github.io/" target="_blank">Pan Zhang<sup>†</sup></a><sup></sup> | 
    <a href="https://wutong16.github.io/" target="_blank">Tong Wu</a><sup></sup> |
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

---

<strong>HiFlow is a training-free and model-agnostic framework to unlock the resolution potential of pre-trained flow models.</strong>

<details><summary>📖 Click for the full abstract of HiFlow</summary>

<div align="left">

> Text-to-image (T2I) diffusion/flow models have drawn considerable attention recently due to their remarkable ability to deliver flexible visual creations. Still, high-resolution image synthesis presents formidable challenges due to the scarcity and complexity of high-resolution content. To this end,  we present **HiFlow**, a training-free and model-agnostic framework to unlock the resolution potential of pre-trained flow models. Specifically, HiFlow establishes a virtual reference flow within the high-resolution space that effectively captures the characteristics of low-resolution flow information, offering guidance for high-resolution generation through three key aspects: initialization alignment for low-frequency consistency, direction alignment for structure preservation, and acceleration alignment for detail fidelity. By leveraging this flow-aligned guidance, HiFlow substantially elevates the quality of high-resolution image synthesis of T2I models and demonstrates versatility across their personalized variants. Extensive experiments validate HiFlow's superiority in achieving superior high-resolution image quality over current state-of-the-art methods.
</details>
</div>

</div>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="__assets__/hiflow_teaser.png">
</div>
<br>

## 💻 Overview
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="__assets__/hiflow_pipeline.png">
</div>
<br>

HiFlow constructs reference flow from low-resolution sampling trajectory to offer initiation alignment, direction alignment, and acceleration alignment, enabling flow-aligned high-resolution image generation. Specifically, HiFlow involves a cascade generation paradigm: First, a virtual reference flow is constructed in the high-resolution space based on the step-wise estimated clean samples of the low-resolution sampling flow. Then, during high-resolution synthesizing, the reference flow offers guidance from sampling initialization, denoising direction, and moving acceleration, aiding in achieving consistent low-frequency patterns, preserving structural features, and maintaining high-fidelity details. Such flow-aligned guidance from the sampling trajectory facilitates better merging of the structure synthesized at the low-resolution scale and the details synthesized at the high-resolution scale, enabling superior generation. 



## 🖋 News
- Paper is available on arXiv!

## 🏗️ Todo
- [ ] 🚀 Release the HiFlow code
- [x] 🚀 Release paper



