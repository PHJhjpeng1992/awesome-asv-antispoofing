# Awesome ASV Anti-Spoofing [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md) 

## Table of contents

* [Overview](#Overview)
* [Publications](#Publications)
* [Software](#Software)
  * [Framework](#Framework)
  * [Evaluation](#Evaluation)
  * [Clustering](#Clustering)
  * [Speaker embedding](#Speaker-embedding)
  * [Audio feature extraction](#Audio-feature-extraction)
  * [Audio data augmentation](#Audio-data-augmentation)
  * [Other sotware](#Other-software)
* [Datasets](#Datasets)
  * [Spoofing datasets](#Diarization-datasets)
  * [Phisical access training sets](#Phisical-access-training-sets)
  * [Logical access training sets](#Logical-access-training-sets)
  * [Augmentation noise sources](#Augmentation-noise-sources)
* [Conferences](#Conferences)
* [Leaderboards](#Leaderboards)
* [Other learning materials](#Other-learning-materials)
  * [Books](#Books)
  * [Tech blogs](#Tech-blogs)
  * [Video tutorials](#Video-tutorials)
* [Products](#Products)
* [Project](#Project)
* [Standards](#Standards)

## Overview

This is a curated list of awesome ASV(Automatic Speaker Verification) Anti-Spoofing papers, libraries, datasets, and other resources.

The purpose of this repo is to organize the world’s resources for voice anti-spoofing, and make them universally accessible and useful.

To add items to this page, simply send a pull request. ([contributing guide](CONTRIBUTING.md))

## Publications

### Special topics

#### Review & survey papers

* [Advances in anti-spoofing: From the perspective of ASVspoof challenges](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/advances-in-antispoofing-from-the-perspective-of-asvspoof-challenges/6B5BB5B75A49022EB869C7117D5E4A9C), 2020
* [Countermeasures to Replay Attacks: A Review](https://www.tandfonline.com/doi/abs/10.1080/02564602.2019.1684851?journalCode=titr20), 2020
* [Introduction to Voice Presentation Attack Detection and Recent Advances](https://arxiv.org/pdf/1901.01085), 2019
* [An Investigation of Deep-Learning Frameworks for Speaker Verification Anti-spoofing](http://crss.utdallas.edu/Publications/spoof_jp.pdf), 2017
* [Spoofing and countermeasures for speaker verification A survey](https://www.sciencedirect.com/science/article/abs/pii/S0167639314000788), 2015

#### Fast & light anti-spoofing

* [Void: A fast and light voice liveness detection system](https://www.usenix.org/conference/usenixsecurity20/presentation/ahmed-muhammad), 2020
* [Audio Replay Attack Detection with Deep Learning Frameworks](https://www.researchgate.net/publication/319185301_Audio_Replay_Attack_Detection_with_Deep_Learning_Frameworks), 2017

#### Anti-spoofing with phoneme

* [Phoneme Specific Modelling and Scoring Techniques for Anti Spoofing System](https://ieeexplore.ieee.org/document/8682411), 2019
* [The SYSU System for the Interspeech 2015 Automatic Speaker Verification Spoofing and Countermeasures Challenge](https://arxiv.org/pdf/1507.06711), 2015

#### Anti-spoofing with brain

* [The Crux of Voice (In)Security: A Brain Study of Speaker Legitimacy Detection](https://www.ndss-symposium.org/ndss-paper/the-crux-of-voice-insecurity-a-brain-study-of-speaker-legitimacy-detection/), 2019

#### Anti-spoofing with fieldprint

* [The Catcher in the Field: A Fieldprint based Spoofing Detection for Text-Independent Speaker Verification](https://dl.acm.org/doi/10.1145/3319535.3354248), 2019
* [You Can Hear But You Cannot Steal: Defending against Voice Impersonation Attacks on Smartphones](https://cse.buffalo.edu/~lusu/papers/ICDCS2017Si.pdf), 2017

#### Anti-spoofing with articulatory gesture

* [Hearing Your Voice is Not Enough: An Articulatory Gesture Based Liveness Detection for Voice Authentication](https://acmccs.github.io/papers/p57-zhangA.pdf), 2017
* [VoiceLive: A Phoneme Localization based Liveness Detection for Voice Authentication on Smartphones](http://www.winlab.rutgers.edu/~yychen/papers/VoiceLive%20A%20Phoneme%20Localization%20based%20Liveness%20Detection%20for%20Voice%20Authentication%20on%20Smartphones.pdf), 2016

#### Anti-spoofing with Multi-task
* [Adversarial Multi-Task Learning for Speaker Normalization in Replay Detection](https://ieeexplore.ieee.org/document/9054322), 2020
* [Multi-task learning of deep neural networks for joint automatic speaker verification and spoofing detection](https://ieeexplore.ieee.org/document/89023289), 2019
* [Anti-Spoofing Speaker Verification System with Multi-Feature Integration and Multi-Task Learning](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1698.pdf), 2019
* [Replay spoofing detection system for automatic speaker verification using multi-task learning of noise classes](https://arxiv.org/pdf/1808.09638), 2018

#### Anti-spoofing with smarthome

* [Protecting Voice Controlled Systems Using Sound Source Identification Based on Acoustic Cues](https://arxiv.org/pdf/1811.07018), 2018

#### Challenges
* [ASVspoof 2021](https://www.asvspoof.org), 2021
* [ASVspoof 2019](https://www.asvspoof.org/index2019.html), 2019
* [ASVspoof 2017](https://www.asvspoof.org/index2017.html), 2017
* [BTAS 2016](http://ieee-biometrics.org/btas2016/competitions.html), 2016
* [ASVspoof 2015](https://www.asvspoof.org/index2015.html), 2015

### Spoofing attack type 

#### Self trigger attack

* [Your Voice Assistant is Mine: How to Abuse Speakers to Steal Information and Control Your Phone](https://arxiv.org/pdf/1407.4923), 2014

* [A11y Attacks: Exploiting Accessibility in Operating Systems](http://wenke.gtisc.gatech.edu/papers/a11y.pdf), 2014

#### Inaudible voice command attack

*[IEMI Threats for Information Security: Remote Command Injection on Modern Smartphones](https://ieeexplore.ieee.org/document/7194754), 2015

* [DolphinAttack: Inaudible Voice Commands](https://acmccs.github.io/papers/p103-zhangAemb.pdf), 2017

#### Hidden voice command attack

* **White-box attack** 
  * [Hidden Voice Commands](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_carlini.pdf), 2016

  * [Audio Adversarial Examples: Targeted Attacks on Speech-to-Text](https://arxiv.org/pdf/1801.01944), 2017

  * [Adversarial Attacks Against Automatic SpeechRecognition Systems via Psychoacoustic Hiding](https://arxiv.org/pdf/1808.05665.pdf)， 2018
* **Black-box attack**

  * [CommanderSong: A Systematic Approach for Practical Adversarial Voice Recognition](https://arxiv.org/pdf/1801.08535)， 2018

  * [SirenAttack: Generating Adversarial Audio for End-to-End Acoustic Systems](https://nesa.zju.edu.cn/download/SirenAttack%20Generating%20Adversarial%20Audio%20for%20End-to-End%20Acoustic%20Systems_AsiaCCS.pdf)， 2019

* **Gray-box attack**

  * [Adversarial Music: Real World Audio AdversaryAgainst Wake-word Detection System](https://proceedings.neurips.cc/paper/2019/file/ebbdfea212e3a756a1fded7b35578525-Paper.pdf)， 2019

#### Voice conversion attack

* [Voice conversion versus speaker verification:an overview](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/DDAB15B28710960D07547DE06A98C695/S2048770314000171a.pdf/voice-conversion-versus-speaker-verification-an-overview.pdf), 2014

#### Speech synthesis attack

* [Evaluation of Speaker Verification Security and Detection of HMM-Based Synthetic Speech](https://ieeexplore.ieee.org/document/6205335), 2012

#### Replay attack

* [A study on replay attack and anti-spoofing for text-dependent speaker verification](https://ieeexplore.ieee.org/document/7041636), 2014
* [A Study on Replay Attack and Anti-Spoofing for Automatic Speaker Verification](https://arxiv.org/pdf/1706.02101), 2017

#### Impostor attack

* [Can a Professional Imitator Fool a GMM-Based Speaker Verification System?](https://infoscience.epfl.ch/record/83202), 2005

* [I-Vectors Meet Imitators: On Vulnerability of Speaker Verification Systems Against Voice Mimicry](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_0930.pdf), 2013

### Other

#### 2021

* [Data Quality as Predictor of Voice Anti-Spoofing Generalization](https://arxiv.org/abs/2103.14602)

#### 2020

* [End-to-end anti-spoofing with RawNet2](https://arxiv.org/pdf/2011.01108)
* [Residual networks for resisting noise: analysis of an embeddings-based spoofing countermeasure](https://karkirowle.github.io/files/Odyssey2020_spoofingResNet_Halpern_et_al.pdf)
* [Detecting Replay Attacks Using Multi-Channel Audio: A Neural Network-Based Method](https://arxiv.org/pdf/2003.08225)
* [An analysis of speaker dependent models in replay detection](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/1B7A246707EF8DDB5B2618327299380C/S2048770320000098a.pdf/an_analysis_of_speaker_dependent_models_in_replay_detection.pdf)
* [Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals](https://arxiv.org/pdf/2007.05979)
* [Dynamically Mitigating Data Discrepancy with Balanced Focal Loss for Replay Attack Detection](https://arxiv.org/pdf/2006.14563)
* [Voice Spoofing Detection Corpus for Single and Multi-order Audio Replays](https://arxiv.org/pdf/1909.00935)
* [An Ensemble Based Approach for Generalized Detection of Spoofing Attacks to Automatic Speaker Recognizers](https://ieeexplore.ieee.org/document/9054558)
* [Defense against adversarial attacks on spoofing countermeasures of ASV](https://arxiv.org/pdf/2003.03065)
* [Multiple Points Input For Convolutional Neural Networks in Replay Attack Detection](https://ieeexplore.ieee.org/document/9053303)

#### 2019

* [Auditory Inspired Spatial Differentiation for Replay Spoofing Attack Detection](https://ieeexplore.ieee.org/document/8683693)
* [Attention-Based LSTM Algorithm for Audio Replay Detection in Noisy Environments](https://pdfs.semanticscholar.org/6945/3fac2454bf77af1119155cff24243e3385ce.pdf?_ga=2.60651435.727780602.1620370656-1450628680.1620370656)
* [Cross-domain replay spoofing attack detection using domain adversarial training](https://x-lance.sjtu.edu.cn/papers/2019/hjw77-wang-is2019-2.pdf)
* [Transmission Line Cochlear Model Based AM-FM Features for Replay Attack Detection](https://ieeexplore.ieee.org/document/8682771)
* [Adversarial Attacks on Spoofing Countermeasures of automatic speaker verification](https://arxiv.org/pdf/1910.08716)
* [Replay Spoofing Countermeasure Using Autoencoder and Siamese Network on ASVspoof 2019 Challenge](https://arxiv.org/pdf/1910.13345)

#### 2018

* [Independent Modelling of Long and Short Term Speech Information for Replay Detection](https://www.researchgate.net/publication/328828011_Independent_Modelling_of_Long_and_Short_Term_Speech_Information_for_Replay_Detection)
* [Voice livness detection based on pop-noise detector with phoneme information for speaker verification](https://www.researchgate.net/publication/310760372_Voice_livness_detection_based_on_pop-noise_detector_with_phoneme_information_for_speaker_verification)
* [An end-to-end spoofing countermeasure for automatic speaker verificationusing evolving recurrent neural networks](http://hectordelgado.me/wp-content/uploads/Delgado2018c.pdf)
* [Deep Siamese Architecture Based Replay Detection for Secure VoiceBiometric](https://isca-speech.org/archive/Interspeech_2018/pdfs/1819.pdf)
* [Use of Claimed Speaker Models for Replay Detection](https://ieeexplore.ieee.org/document/8659510?denied=)
* [Replay Attacks Detection Using Phase and Magnitude Features with Various Frequency Resolutions](https://ieeexplore.ieee.org/document/8706628)
* [Performance evaluation of front- and back-end techniques for ASV spoofingdetection systems based on deep features](https://www.isca-speech.org/archive/IberSPEECH_2018/pdfs/IberS18_P1-6_Gomez-Alanis.pdf)
* [Modulation Dynamic Features for the Detection of Replay Attacks](https://www.researchgate.net/publication/327389405_Modulation_Dynamic_Features_for_the_Detection_of_Replay_Attacks)

#### 2017

* [Audio Replay Attack Detection Using High-Frequency Features](https://www.researchgate.net/publication/319185216_Audio_Replay_Attack_Detection_Using_High-Frequency_Features)
* [Replay Attack Detection Using DNN for Channel Discrimination](https://www.isca-speech.org/archive/Interspeech_2017/abstracts/1377.html)
* [Investigating the use of Scattering Coefficients for Replay Attack Detection](https://ieeexplore.ieee.org/document/8282211)
* [Constant Q cepstral coefficients: a spoofing countermeasure for automatic speaker verification](http://www.eurecom.fr/en/publication/5146/download/sec-publi-5146.pdf)

#### 2016

* [Anti-spoofing Methods for Automatic Speaker Verification System](https://link.springer.com/chapter/10.1007/978-3-319-52920-2_17)
* [Overview of BTAS 2016 Speaker Anti-spoofing Competition](http://publications.idiap.ch/downloads/papers/2017/Korshunov_BTAS_2016.pdf)
* [Voice Liveness Detection for Speaker Verification based on a Tandem Single/Double-channel Pop Noise Detector](http://www.odyssey2016.com/papers/pdfs_stamped/80.pdf)
* [Cross-Database Evaluation of Audio-Based Spoofing Detection Systems](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/1326.PDF)
* [Spoofing detection from a feature representationperspective](https://dr.ntu.edu.sg/bitstream/10356/89643/1/ICASSP2016_final.pdf)
* [Spoofing Speech Detection using Temporal Convolutional Neural Network](https://ieeexplore.ieee.org/document/7820738)

#### 2015

* [Robust Deep Feature for Spoofing Detection - The SJTU System for ASVspoof 2015 Challenge](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_2097.pdf)
* [A Comparison of Features for Synthetic Speech Detection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.717.6743&rep=rep1&type=pdf)

#### 2014

* [Anti-spoofing: voice databases](https://www.eurecom.fr/~evans/papers/pdfs/template.pdf)

#### 2013

* [Vulnerability evaluation of speaker verification under voice conversionspoofing: the effect of text constraints](http://www.cs.joensuu.fi/pages/tkinnu/webpage/pdf/spoofing_attack_IS2013.pdf)

#### 1999
* [Vulnerability In Speaker Verification - A Study Of Technical Impostor Techniques](https://www.isca-speech.org/archive/archive_papers/eurospeech_1999/e99_1211.pdf)

## Software

### Framework

| Link | Language | Description |
| ---- | -------- | ----------- |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) ![GitHub stars](https://img.shields.io/github/stars/speechbrain/speechbrain?style=social) | Python & PyTorch | SpeechBrain is an open-source and all-in-one speech toolkit based on PyTorch. |
| [SIDEKIT](https://projets-lium.univ-lemans.fr/sidekit/) | Python | SIDEKIT is an open source package allow a rapid prototyping of an end-to-end speaker recognition system. |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |
| [kaldi-asr](https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16) [![Build Status](https://travis-ci.com/kaldi-asr/kaldi.svg?branch=master)](https://travis-ci.com/kaldi-asr/kaldi) | C++ & Bash | A toolkit for speech & speaker recognition, intended for use by researchers and professionals.  |
| [Alize LIA_SpkDet](https://alize.univ-avignon.fr/) | C++ | ALIZÉ is an opensource platform for speaker recognition. LIA_SpkSeg is the tools for model training,feature normalization,socre normalization,etc. |
| [SPEAR Toolkit (based on BOB) ](https://pypi.org/project/bob.bio.spear/) | python | This package is part of the signal-processing and machine learning toolbox Bob. |
| [MSRidentity Toolbox ](https://www.microsoft.com/en-us/download/details.aspx?id=52279) | Matlab | This toolbox contains a collection of MATLAB tools and routines that can be used for research and development in speaker recognition. [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/09/MSR-Identity-Toolbox-v1_1.pdf)|

### Evaluation

* [t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification](https://arxiv.org/pdf/1804.09618) [[MATLAB](https://www.asvspoof.org/asvspoof2019/tDCF_matlab_v1.zip)]&[[Python](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip)]
* [Asvspoof 2021 Evaluation Plan](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)

### Clustering
| Link | Language | Description |
| ---- | -------- | ----------- |
| [sklearn.cluster](https://scikit-learn.org/stable/modules/clustering.html) [![Build Status]( https://api.travis-ci.org/scikit-learn/scikit-learn.svg?branch=master)](https://travis-ci.org/scikit-learn/scikit-learn) | Python | scikit-learn clustering algorithms. |
| [PLDA](https://github.com/RaviSoji/plda) ![GitHub stars](https://img.shields.io/github/stars/RaviSoji/plda?style=social) | Python | Probabilistic Linear Discriminant Analysis & classification, written in Python. |
| [PLDA](https://github.com/mrouvier/plda) ![GitHub stars](https://img.shields.io/github/stars/mrouvier/plda?style=social) | C++ | Open-source implementation of simplified PLDA (Probabilistic Linear Discriminant Analysis). |
| [Auto-Tuning Spectral Clustering](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.git) ![GitHub stars](https://img.shields.io/github/stars/tango4j/Auto-Tuning-Spectral-Clustering?style=social) | Python | Auto-tuning Spectral Clustering method that does not need development set or supervised tuning. |

### Speaker embedding

| Link | Method | Language | Description |
| ---- | ------ | -------- | ----------- |
| [resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) ![GitHub stars](https://img.shields.io/github/stars/resemble-ai/Resemblyzer?style=social) | d-vector | Python & PyTorch | PyTorch implementation of generalized end-to-end loss for speaker verification, which can be used for voice cloning and diarization. |
| [Speaker_Verification](https://github.com/Janghyun1230/Speaker_Verification) ![GitHub stars](https://img.shields.io/github/stars/Janghyun1230/Speaker_Verification?style=social) | d-vector | Python & TensorFlow | Tensorflow implementation of generalized end-to-end loss for speaker verification. |
| [PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification) ![GitHub stars](https://img.shields.io/github/stars/HarryVolek/PyTorch_Speaker_Verification?style=social) | d-vector | Python & PyTorch | PyTorch implementation of "Generalized End-to-End Loss for Speaker Verification" by Wan, Li et al. With UIS-RNN integration. |
| [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) ![GitHub stars](https://img.shields.io/github/stars/CorentinJ/Real-Time-Voice-Cloning?style=social) | d-vector | Python & PyTorch | Implementation of "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis" (SV2TTS) with a vocoder that works in real-time. |
| [deep-speaker](https://github.com/philipperemy/deep-speaker) ![GitHub stars](https://img.shields.io/github/stars/philipperemy/deep-speaker?style=social) | d-vector |Python & Keras | Third party implementation of the Baidu paper Deep Speaker: an End-to-End Neural Speaker Embedding System. |
| [x-vector-kaldi-tf](https://github.com/hsn-zeinali/x-vector-kaldi-tf) ![GitHub stars](https://img.shields.io/github/stars/hsn-zeinali/x-vector-kaldi-tf?style=social) | x-vector | Python & TensorFlow & Perl | Tensorflow implementation of x-vector topology on top of Kaldi recipe. |
| [kaldi-ivector](https://github.com/idiap/kaldi-ivector) ![GitHub stars](https://img.shields.io/github/stars/idiap/kaldi-ivector?style=social) | i-vector | C++ & Perl |  Extension to Kaldi implementing the standard i-vector hyperparameter estimation and i-vector extraction procedure. |
| [voxceleb-ivector](https://github.com/swshon/voxceleb-ivector) ![GitHub stars](https://img.shields.io/github/stars/swshon/voxceleb-ivector?style=social) | i-vector |Perl | Voxceleb1 i-vector based speaker recognition system. |
| [pytorch_xvectors](https://github.com/manojpamk/pytorch_xvectors) ![GitHub stars](https://img.shields.io/github/stars/manojpamk/pytorch_xvectors?style=social) | x-vector | Python & PyTorch | PyTorch implementation of Voxceleb x-vectors. Additionaly, includes meta-learning architectures for embedding training. Evaluated with speaker diarization and speaker verification. |
| [ASVtorch](https://gitlab.com/ville.vestman/asvtorch) | i-vector | Python & PyTorch | ASVtorch is a toolkit for automatic speaker recognition. |

### Audio feature extraction

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [LibROSA](https://github.com/librosa/librosa) ![GitHub stars](https://img.shields.io/github/stars/librosa/librosa?style=social) | Python | Python library for audio and music analysis. https://librosa.github.io/ |
| [python_speech_features](https://github.com/jameslyons/python_speech_features) ![GitHub stars](https://img.shields.io/github/stars/jameslyons/python_speech_features?style=social) | Python | This library provides common speech features for ASR including MFCCs and filterbank energies. https://python-speech-features.readthedocs.io/en/latest/ |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |

### Audio data augmentation

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) ![GitHub stars](https://img.shields.io/github/stars/LCAV/pyroomacoustics?style=social) | Python | Pyroomacoustics is a package for audio signal processing for indoor applications. It was developed as a fast prototyping platform for beamforming algorithms in indoor scenarios. https://pyroomacoustics.readthedocs.io |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ![GitHub stars](https://img.shields.io/github/stars/DavidDiazGuerra/gpuRIR?style=social) | Python | Python library for Room Impulse Response (RIR) simulation with GPU acceleration |
| [rir_simulator_python](https://github.com/sunits/rir_simulator_python) ![GitHub stars](https://img.shields.io/github/stars/sunits/rir_simulator_python?style=social) | Python | Room impulse response simulator using python |

### Other software
| Link  | Language | Description |
| ----  | -------- | ----------- |
| [Rawnet2](https://github.com/Jungjee/RawNet) ![GitHub stars](https://img.shields.io/github/stars/Jungjee/RawNet?style=social) | Python & Bash | End-to-End Neural Anti-spoofing. |
| [ReMASC](https://github.com/YuanGongND/ReMASC) ![GitHub stars](https://img.shields.io/github/stars/YuanGongND/ReMASC?style=social) | Python | Realistic Replay Attack Corpus for Voice Controlled Systems. |
| [Attentive-Filtering-Network](https://github.com/jefflai108/Attentive-Filtering-Network) ![GitHub stars](https://img.shields.io/github/stars/jefflai108/Attentive-Filtering-Network?style=social) | Python & Bash | University of Edinbrugh-Johns Hopkins University's system for ASVspoof 2017 Version 2.0 dataset. |

## Datasets

### Spoofing datasets

| Audio | Type | Language | Pricing | Additional information |
| ----- | ------------------------ | -------- | ------- | ---------------------- |
| [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) | [PA(16.44Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/3336/PA.zip?sequence=4&isAllowed=y) , [LA(7.116Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y) | en | Free | [Evaluation Plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf)
| [ASVspoof 2017](https://datashare.ed.ac.uk/handle/10283/3055) | PA-[Train(200.7Mb)](https://datashare.ed.ac.uk/bitstream/handle/10283/3055/ASVspoof2017_V2_train.zip?sequence=10&isAllowed=y), [Dev(133.7Mb)](https://datashare.ed.ac.uk/bitstream/handle/10283/3055/ASVspoof2017_V2_dev.zip?sequence=5&isAllowed=y), [Eval(1.065Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/3055/ASVspoof2017_V2_eval.zip?sequence=6&isAllowed=y) | en | Free | [Evaluation Plan](https://datashare.ed.ac.uk/bitstream/handle/10283/3055/asvspoof2017_evalplan_v1.1.pdf?sequence=3&isAllowed=y)
| [SAS Corpus](https://datashare.ed.ac.uk/handle/10283/2741) | LA-[SS_LARGE-16k (7.591Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/SS_LARGE-16k.tar.gz?sequence=2&isAllowed=y), [SS_LARGE-48k (7.798Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/SS_LARGE-48k.tar.gz?sequence=3&isAllowed=y), [SS_MARY_LARGE (7.303Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/SS_MARY_LARGE.tar.gz?sequence=4&isAllowed=y), [SS_SMALL-16k (7.582Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/SS_SMALL-16k.tar.gz?sequence=5&isAllowed=y), [SS_SMALL-16k (7.582Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/SS_SMALL-16k.tar.gz?sequence=5&isAllowed=y), [SS_SMALL-48k (7.788Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/SS_SMALL-48k.tar.gz?sequence=6&isAllowed=y), [VC_C1 (10.00Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_C1.tar.gz?sequence=7&isAllowed=y), [VC_EVC (6.518Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_EVC.tar.gz?sequence=8&isAllowed=y), [VC_FESTVOX (10.04Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_FESTVOX.tar.gz?sequence=9&isAllowed=y), [VC_FS (10.15Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_FS.tar.gz?sequence=10&isAllowed=y), [VC_GMM (9.830Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_GMM.tar.gz?sequence=11&isAllowed=y), [VC_KPLS (9.703Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_KPLS.tar.gz?sequence=12&isAllowed=y), [VC_LSP (9.616Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_LSP.tar.gz?sequence=13&isAllowed=y), [VC_TVC (6.489Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/VC_TVC.tar.gz?sequence=14&isAllowed=y), [human (3.229Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/782/human.tar.gz?sequence=15&isAllowed=y) | en | Free | [LICENSE](https://datashare.ed.ac.uk/bitstream/handle/10283/782/license_text?sequence=17&isAllowed=y)
| [ASVspoof 2015](https://datashare.ed.ac.uk/handle/10283/853) | LA-[Data - Part aa (7.543Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/853/wav_data.aa.tar.gz?sequence=6&isAllowed=y),[Data - Part ab (7.543Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/853/wav_data.ab.tar.gz?sequence=7&isAllowed=y),[Data - Part ac (7.331Gb)](https://datashare.ed.ac.uk/bitstream/handle/10283/853/wav_data.ac.tar.gz?sequence=8&isAllowed=y) | en | Free | [LICENSE](https://datashare.ed.ac.uk/bitstream/handle/10283/853/license_text?sequence=9&isAllowed=y)

### Phisical access training sets
* [ASV2019 Training set](https://datashare.ed.ac.uk/bitstream/handle/10283/3336/PA.zip?sequence=4&isAllowed=y)
* [ASV2017 Training set](https://datashare.ed.ac.uk/bitstream/handle/10283/3055/ASVspoof2017_V2_train.zip?sequence=10&isAllowed=y)
### Logical access training sets
* [ASV2019 Training set](https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y)
* ASV2015 Training set [Part aa](https://datashare.ed.ac.uk/bitstream/handle/10283/853/wav_data.aa.tar.gz?sequence=6&isAllowed=y),[ab](https://datashare.ed.ac.uk/bitstream/handle/10283/853/wav_data.ab.tar.gz?sequence=7&isAllowed=y),[ac](https://datashare.ed.ac.uk/bitstream/handle/10283/853/wav_data.ac.tar.gz?sequence=8&isAllowed=y)
### Augmentation noise sources

| Name | Utterances | Pricing | Additional information |
| ---- | ---------- | ------- | ---------------------- |
| [AudioSet](https://research.google.com/audioset/) | 2M | Free | A large-scale dataset of manually annotated audio events. |
| [MUSAN](https://www.openslr.org/17/) | N/A | Free | MUSAN is a corpus of music, speech, and noise recordings. |

### Speaker Verification training sets

| Name | Utterances | Speakers | Language | Pricing | Additional information |
| ---- | ---------- | -------- | -------- | ------- | ---------------------- |
| [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) | 6K+ | 630 | en | $250.00 | Published in 1993, the TIMIT corpus of read speech is one of the earliest speaker recognition datasets. |
| [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) | 43K+ | 109 | en | Free | Most were selected from a newspaper plus the Rainbow Passage and an elicitation paragraph intended to identify the speaker's accent. |
| [LibriSpeech](http://www.openslr.org/12) | 292K | 2K+ | en | Free | Large-scale (1000 hours) corpus of read English speech. |
| [Multilingual LibriSpeech (MLS)](http://openslr.org/94/) | ? | ? | en, de, nl, es, fr, it, pt, po | Free | Multilingual LibriSpeech (MLS) dataset is a large multilingual corpus suitable for speech research. The dataset is derived from read audiobooks from LibriVox and consists of 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish. |
| [LibriVox](https://librivox.org/) | 180K | 9K+ | Multiple | Free | Free public domain audiobooks. LibriSpeech is a processed subset of LibriVox. Each original unsegmented utterance could be very long. |
| [VoxCeleb 1&2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | 1M+ | 7K | Multiple | Free | VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube. |
| [The Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/) | 5K | 879 | en, de, nl | Free | Volunteer readers reading Wikipedia articles. |
| [CN-Celeb](http://www.openslr.org/82/) | 130K+ | 1K | zh | Free | A Free Chinese Speaker Recognition Corpus Released by CSLT@Tsinghua University. |
| [BookTubeSpeech](https://users.wpi.edu/~jrwhitehill/BookTubeSpeech/index.html) | 8K | 8K | en | Free | Audio samples extracted from BookTube videos - videos where people share their opinions on books - from YouTube. The dataset can be downloaded using [BookTubeSpeech-download](https://github.com/wq2012/BookTubeSpeech-download). |
| [DeepMine](http://data.deepmine.ir/en/index.html) | 540K | 1850 | fa, en | Unknown | A speech database in Persian and English designed to build and evaluate speaker verification, as well as Persian ASR systems. |
| [NISP-Dataset](https://github.com/iiscleap/NISP-Dataset) | ? | 345 | hi, kn, ml, ta, te (all Indian languages) | Free | This dataset contains speech recordings along with speaker physical parameters (height, weight, ... ) as well as regional information and linguistic information. |

## Conferences

| Conference/Workshop | Frequency | Page Limit  | Organization | Blind Review |
| ------------------- | --------- | ----------  | ------------ | ------------ |
| ICASSP              | Annual    | 4 + 1 (ref) | IEEE         | No           |
| InterSpeech         | Annual    | 4 + 1 (ref) | ISCA         | No           |
| APSIPA              | Annual    | 4 + 1 (ref) | IEEE         | Yes           |
| Odyssey     | Biennial  | 8 + 2 (ref) | ISCA         | No           |
| SLT                 | Biennial  | 6 + 2 (ref) | IEEE         | Yes          |
| ASRU                | Biennial  | 6 + 2 (ref) | IEEE         | Yes          |
| WASPAA              | Biennial  | 4 + 1 (ref) | IEEE         | No           |

## Other learning materials

### Books

* [Handbook of Biometric Anti-Spoofing](https://link.springer.com/book/10.1007%2F978-3-319-92627-8)

### Tech blogs

* [Can You Fool Voice Biometrics?](https://www.nice.com/engage/blog/can-you-fool-voice-biometrics-2359/) by [Lior Artzi](https://www.nice.com/engage/blog/author/lior-artzi/)

* [ID R&D and Synaptics First to Deploy Voice Biometrics on NPU for Smart Home Applications](https://www.synaptics.com/company/blog/IDRD) by Vineet Ganju

### Video tutorials

## Products

| Company | Product |
| ------- | ------- |
| Pindrop | [Deep Voice Engine](https://www.pindrop.com/technologies/deep-voice/) |
| ID R&D | [IDLive™ Voice](https://www.idrnd.ai/voice-anti-spoofing/) |
| VoiceAI | [Voiceprint recognition API](https://market.aliyun.com/products/57124001/cmapi00039301.html#sku=yuncode3330100008)
| Kriston | [Voiceprint API,SDK](http://www.shengwenyun.com/)

## Project

* [OCTAVE](https://www.octave-project.eu/)

## Standards

* [Information security technology — Security requirements of voiceprint recognition data(Exposure draft)](https://www.tc260.org.cn/file/2021-04-28/846cb81c-bc4e-4c8e-aad7-191c9a676884.docx), 2021

* [Technical specifications for voiceprint recognition based security application for mobile finance (JR / t0164-2018)](https://sv.cebnet.com.cn/upload/regulation1101/regulation1101.pdf), 2018