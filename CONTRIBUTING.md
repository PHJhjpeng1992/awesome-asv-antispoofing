# Contributing

Your contributions are always welcome!

Just send a pull request, and I will review and merge it.

## Guidance

### General

For new items, please follow the existing format.

### Publications

Only include complete papers that are directly related to voice anti-spoofing.

For example, these will **NOT** be accepted:
* Course project reports.
* One pager description of a diarization system submitted to DIHARD challenge.
* Commercial system technical document.
* Media post.
* Low quality paper without experiments and evaluations.
* Publications not directly related to speaker diarization:
  * Pure ML papers.
  * Speaker recognition papers.

### Software

* A **Framework** is a software that has all the necessary features to perform
  voice anti-spoofing, including audio processing, feature extraction,
  speaker analysis and clustering, etc.
* A **Evaluation** software must be able to produce voice anti-spoofing related
  metrics that are *permutation invariant*, such as Equal Error Rate
  (EER).
* A **Clustering** software must correspond to a clustering algorithm that has
  been used by at least one anti-spoofing publication.

### Dataset

A anti-spoofing dataset must contain utterances with multiple speakers
**speaking in turn**, and each utterance must have
**time-stamped speaker annotations**.

A anti-spoofing dataset must consisting of **bonafide and replay samples** that can effectively be used to evaluate the perfomance of anti-spoofing algorithm in multi-scnearios.



