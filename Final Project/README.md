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
`pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
