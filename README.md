# Awesome ASV Anti-Spoofing [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md) 

## Table of contents

* [Overview](#Overview)
* [Publications](#Publications)
* [Software](#Software)
  * [Framework](#Framework)
  * [Evaluation](#Evaluation)
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

#### Anti-spoofing with brain

* [The Crux of Voice (In)Security: A Brain Study of Speaker Legitimacy Detection](https://www.ndss-symposium.org/ndss-paper/the-crux-of-voice-insecurity-a-brain-study-of-speaker-legitimacy-detection/), 2019

#### Anti-spoofing with fieldprint

* [The Catcher in the Field: A Fieldprint based Spoofing Detection for Text-Independent Speaker Verification](https://dl.acm.org/doi/10.1145/3319535.3354248), 2019

#### Anti-spoofing with articulatory gesture

* [You Can Hear But You Cannot Steal: Defending against Voice Impersonation Attacks on Smartphones](https://cse.buffalo.edu/~lusu/papers/ICDCS2017Si.pdf), 2017
* [VoiceLive: A Phoneme Localization based Liveness Detection for Voice Authentication on Smartphones](http://www.winlab.rutgers.edu/~yychen/papers/VoiceLive%20A%20Phoneme%20Localization%20based%20Liveness%20Detection%20for%20Voice%20Authentication%20on%20Smartphones.pdf), 2016

#### Anti-spoofing with Multi-task
* [Adversarial Multi-Task Learning for Speaker Normalization in Replay Detection](https://ieeexplore.ieee.org/document/9054322), 2020
* [Multi-task learning of deep neural networks for joint automatic speaker verification and spoofing detection](https://ieeexplore.ieee.org/document/89023289), 2019
* [Anti-Spoofing Speaker Verification System with Multi-Feature Integration and Multi-Task Learning](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1698.pdf), 2019

#### Challenges
* [ASVspoof 2021](https://www.asvspoof.org), 2021
* [ASVspoof 2019](https://www.asvspoof.org/index2019.html), 2019
* [ASVspoof 2017](https://www.asvspoof.org/index2017.html), 2017
* [BTAS 2016](http://ieee-biometrics.org/btas2016/competitions.html), 2016
* [ASVspoof 2015](https://www.asvspoof.org/index2015.html), 2015

### Other

#### 2021

* [Data Quality as Predictor of Voice Anti-Spoofing Generalization](https://arxiv.org/abs/2103.14602)

#### 2020

* [Detecting Replay Attacks Using Multi-Channel Audio: A Neural Network-Based Method](https://arxiv.org/pdf/2003.08225)
* [An analysis of speaker dependent models in replay detection](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/1B7A246707EF8DDB5B2618327299380C/S2048770320000098a.pdf/an_analysis_of_speaker_dependent_models_in_replay_detection.pdf)
* [Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals](https://arxiv.org/pdf/2007.05979)
* [Dynamically Mitigating Data Discrepancy with Balanced Focal Loss for Replay Attack Detection](https://arxiv.org/pdf/2006.14563)
* [Voice Spoofing Detection Corpus for Single and Multi-order Audio Replays](https://arxiv.org/pdf/1909.00935)

#### 2019

* [Auditory Inspired Spatial Differentiation for Replay Spoofing Attack Detection](https://ieeexplore.ieee.org/document/8683693)
* [Attention-Based LSTM Algorithm for Audio Replay Detection in Noisy Environments](https://pdfs.semanticscholar.org/6945/3fac2454bf77af1119155cff24243e3385ce.pdf?_ga=2.60651435.727780602.1620370656-1450628680.1620370656)
* [Cross-domain replay spoofing attack detection using domain adversarial training](https://x-lance.sjtu.edu.cn/papers/2019/hjw77-wang-is2019-2.pdf)
* [Transmission Line Cochlear Model Based AM-FM Features for Replay Attack Detection](https://ieeexplore.ieee.org/document/8682771)
* [Adversarial Attacks on Spoofing Countermeasures of automatic speaker verification](https://arxiv.org/pdf/1910.08716)
* [Replay Spoofing Countermeasure Using Autoencoder and Siamese Network on ASVspoof 2019 Challenge](https://arxiv.org/pdf/1910.13345)


