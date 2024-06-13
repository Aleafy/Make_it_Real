### 💾 Installation

1. Install basic modules: torch and packages in requirements.txt
   
   ```bash
   git clone https://github.com/Aleafy/Make_it_Real.git
   cd Make_it_Real

   conda create -n mkreal python=3.8 
   pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
   pip install -r requirements.txt
   ```

2. Install rendering utils: Blender and Kaolin
   
   Download [blender-3.2.2-linux-x64.tar.xz_original](https://download.blender.org/release/Blender3.2/) or [blender_3.2.2_with_installed_modules](https://drive.google.com/file/d/1PKbCS7VymPo_xVYavT0CDd42OmXbfurY/view?usp=sharing)

   ```bash
   tar -xvf blender-3.2.2-linux-x64.tar.xz
   export PATH=$PATH:path_to_blender/blender-3.2.2-linux-x64
   ```

   and Kaolin
   ```bash
   pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.10.1_cu111.html
   ```
   
3. Install segment & mark utils:
   ```
   # install Semantic-SAM
   pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package
   # install Deformable Convolution for Semantic-SAM
   cd som/ops && sh make.sh && cd ..
   # install detectron2
   python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
   ```
   Get model weight through `bash som/ckpts/download_ckpt.sh`