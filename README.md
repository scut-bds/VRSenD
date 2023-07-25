# VRED

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href="support os"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href="https://github.com/scut-bds/VRSenD/graphs/contributors"><img src="https://img.shields.io/github/contributors/scut-bds/VRSenD?color=9ea"></a>
    <a href="https://github.com/scut-bds/VRSenD/commits"><img src="https://img.shields.io/github/commit-activity/m/scut-bds/VRSenD?color=3af"></a>
    <a href="https://github.com/scut-bds/VRSenD/issues"><img src="https://img.shields.io/github/issues/scut-bds/VRSenD?color=9cc"></a>
    <a href="https://github.com/scut-bds/VRSenD/stargazers"><img src="https://img.shields.io/github/stars/scut-bds/VRSenD?color=ccf"></a>
</p>




虚拟现实全景图二分类情感分析数据集

Compared with non-immersive environments and semi-immersive environments, the sense of presence and interaction of fully immersive virtual environments enables users to be more deeply involved in the evoked materials and more reliably induce a variety of emotional states in the interaction with them. But publicly available emotional datasets in virtual environments are scarce. The purpose of this repository is to create a large-scale VR panoramic emotion data set that can effectively induce different emotional responses of subjects, and to design the production process of this data set and emotion labeling paradigm in strict reference to the existing emotional experiment standards, so as to provide a scientific and rigorous reference template for subsequent research.



This paper exposes a Virtual Reality Omni-directional Image Sentiment Dataset (VRSenD) in a virtual reality scenario. The data set contains the original data of 1140 panoramic images viewed by 26 subjects and the pre-processed data, which are stored in Excel format for external access and processing. Among them, the channel number and size of 1140 panoramic images were uniformly processed into RGB three-channel and 2000×1000 resolution, respectively, and stored in ERP projection format. It has been proved that the contents of the panorama emotion data set can effectively induce different types of emotions, and each panorama map corresponds to an effective binary discrete emotion label of positive or negative. Contains the Panorama data folder, user data folder, data set description file, and data set license file.



The production process of the panorama emotion data set is as follows:
(1) The virtual reality scene materials that can effectively induce different emotions of the subjects are selected through the network collection and pre-experiment;
(2) The HMD emotion labeling system was built through Unity3D engine, which could collect emotion valence validity/arousal and discrete labels while the subjects watched the panorama.
(3) GUI discrete emotion labeling system was used to collect binary emotion labeling data of users.
(4) The efficiency of annotation was verified by Python script statistics, and the classification of panorama (positive and negative) emotion labels were determined according to the voting results.

Some examples are shown in following, with positive samples at the top and negative samples at the bottom.
![image](https://user-images.githubusercontent.com/34803816/230023139-a65d32a0-caaa-4bd4-97ce-439d5f778f2f.png)
