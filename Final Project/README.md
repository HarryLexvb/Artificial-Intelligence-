# Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. This was my final project of the artificial intelligence course. 

SV2TTS is a deep learning framework in three stages. In the first stage, one creates a digital representation of a voice from a few seconds of audio. In the second and third stages, this representation is used as reference to generate speech given arbitrary text.

### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |

## Heads up
Like everything else in Deep Learning, this repo is quickly getting old. Many other open-source repositories or SaaS apps (often paying) will give you a better audio quality than this repository will. If you care about the fidelity of the voice you're cloning, and its expressivity, here are some personal recommendations of alternative voice cloning solutions:
- Check out [CoquiTTS](https://github.com/coqui-ai/tts) for an open source repository that is more up-to-date, with a better voice cloning quality and more functionalities.
- Check out [paperswithcode](https://paperswithcode.com/task/speech-synthesis/) for other repositories and recent research in the field of speech synthesis.
- Check out [Resemble.ai](https://www.resemble.ai/) (disclaimer: I work there) for state of the art voice cloning with little hassle.

## Setup

### 1. Install Requirements
1. Both Windows and Linux are supported. A GPU is recommended for training and for inference speed, but is not mandatory.
2. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`, but this is optional.
3. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
4. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.

### 2. Complete installation on windows operating system 
1. How to install ffmpeg
- Go to the official FFmpeg website at https://www.ffmpeg.org/ and go to the download section.
- Scroll down until you find the download links for the Windows version. Typically, there are links for static versions and links for versions with installers. Static versions are ZIP files that contain all the FFmpeg executables, while installer versions are executable files that guide you through the installation process.
- If you prefer to use a static version, click on the corresponding download link and save the ZIP file to your computer. Unzip the ZIP file to a location of your choice.
- If you prefer to use the installer version, click on the corresponding download link and run the downloaded file. Follow the installer instructions to complete the installation process. During the installation, you can choose the location where FFmpeg will be installed.

Once you have completed the installation, you will be able to use FFmpeg from the command line or by integrating with other applications. Be sure to add the location of FFmpeg executable files to your system's PATH so that you can access them from any location on the command line.

Remember that FFmpeg is a powerful and versatile tool for audio and video processing, but its use requires additional knowledge. See the official FFmpeg documentation for more information on how to use it and the available commands.

2. How to install pytorch
```bash
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```  
- Verify the installation 
```python
import torch
print(torch.__version__)
```  
- If you experience this type of error
```
C:\Users\harry>pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
Looking in links: https://download.pytorch.org/whl/torch_stable.html
ERROR: Could not find a version that satisfies the requirement torch==1.9.0+cpu (from versions: 2.0.0, 2.0.0+cpu, 2.0.0+cu117, 2.0.0+cu118, 2.0.1, 2.0.1+cpu, 2.0.1+cu117, 2.0.1+cu118)
ERROR: No matching distribution found for torch==1.9.0+cpu
```
- You can solve it in the following way
```bash
pip install torch torchvision torchaudio
```
3. Installation of the rest of the requirements
```bash
pip install --upgrade pip
```
```bash
pip install inflect
```
```bash
pip install librosa
```
```bash
pip install matplotlib
```
```bash
pip install pyqt5
```
```bash
pip install scikit-learn
```
```bash
pip install sounddevice
```
```bash
pip install soundfile
```
```bash
pip install tqdm
```
```bash
pip install umap-learn
```
```bash
pip install Unidecode
```
```bash
pip install visdom
```
```bash
python -m visdom.server
```
```bash
pip install webrtcvad
```
- If you experience this type of error
```
Collecting webrtcvad
  Using cached webrtcvad-2.0.10.tar.gz (66 kB)
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: webrtcvad
  Building wheel for webrtcvad (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [9 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build\lib.win-amd64-cpython-311
      copying webrtcvad.py -> build\lib.win-amd64-cpython-311
      running build_ext
      building '_webrtcvad' extension
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for webrtcvad
  Running setup.py clean for webrtcvad
Failed to build webrtcvad
ERROR: Could not build wheels for webrtcvad, which is required to install pyproject.toml-based projects
```
- You can solve it in the following way
- The error message indicates that you need to have Microsoft Visual C++ 14.0 or higher installed on your system in order to build the "webrtcvad" package wheel. Here is how to proceed:

- Make sure you have Python installed on your Windows system. You can download the latest version of Python from the official website (https://www.python.org) and follow the installation instructions.

- Install the Microsoft Visual C++ compiler tools. You can download them from the following link: https://visualstudio.microsoft.com/visual-cpp-build-tools/

- Click on the "Download Build Tools" button to download the installer.
Run the installer and select the "C++ build tools" option during installation.
Follow the instructions and complete the installation process.
Once you have installed the Visual C++ Build Tools, open a new command prompt and try to install the "webrtcvad" package again using the following command:
```bash
pip install webrtcvad
```
### 3. (Optional) Download Pretrained Models
Pretrained models are now downloaded automatically. If this doesn't work for you, you can manually download them [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

### 4. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.

### 5. (Optional) Download Datasets
For playing with the toolbox alone, I only recommend downloading [`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz). Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where `<datasets_root>` is a directory of your choosing. Other datasets are supported in the toolbox, see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets). You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.

### 6. Launch the Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  

depending on whether you downloaded any datasets. If you are running an X-server or if you have the error `Aborted (core dumped)`, see [this issue](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/11#issuecomment-504733590).






