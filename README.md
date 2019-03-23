# siamese-tracker-road-trip
基于孪生网络的单目标跟踪论文汇总

|   Tracker   | AUC-CVPR2013 | Precision-CVPR2013 | AUC-OTB100 | Precision-OTB100 | AUC-OTB50 | Precision-OTB50 |  FPS  |
| :---------: | :----------: | :----------------: | :--------: | :--------------: | :-------: | :-------------: | :--------: |
|    SINT+    |    0.655     |       0.882        |     -      |        -         |     -     |        -        |  4    |
|    SINT     |    0.625     |       0.848        |     -      |        -         |     -     |        -        |   4    |

-------
## Trackers
- **2016_CVPR_SINT**
    * **SINT**:R. Tao, E. Gavves, and A. W. Smeulders. Siamese instance search for tracking. In IEEE Conference on Computer Vision and Pattern Recognition, 2016[[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Tao_Siamese_Instance_Search_CVPR_2016_paper.pdf)][[code](https://github.com/taotaoorange/SINT)][[project](https://taotaoorange.github.io/projects/SINT/SINT_proj.html)]  
        ##### Contributions
    	- Propose to learn a ***generic matching function for tracking***, from external video data, to robustly handle the common appearance variations an object can undergo in video sequences.
    	- Present a ***tracker*** based on the learnt generic matching function which reaches state-of-the-art tracking performance.
    	- Design a ***two-stream Siamese network*** specifically for tracking to learn the matching function.
        ##### Pipeline
        ![pipeline](image/SINT/pipeline.png)
        ##### Candidate Sampling
    	- Use the ***radius sampling strategy*** to generate candidate boxes. At each sample location, generate three scaled versions of the initial box with the scales being $${\frac {\sqrt{2}} {2}, 1, \sqrt{2} }$$
        ##### SINT+
    	- The ***sampling range*** is adaptive to the image resolution, set to be 30/512 ∗ w in this experiment, where w is the image width.
    	- Given the pixels covered by the predicted box in the previous frame and the estimated ***optical flow***, remove the candidate boxes that contain less than 25% of those pixels in the current frame.
-----
- **2016_ECCV_Siamese-fc**
- **2017_CVPR_CFNet**
- **2017_ICCV_DSiam**
- **2017_Siamese_Survey**
- **2018_CVPR_RASNet**
- **2018_CVPR_SA-Siam**
- **2018_CVPR_SiameseRPN**
- **2018_CVPR_SINT++**
- **2018_ECCV_DaSiamRPN**
- **2018_ECCV_Siam-BM**
- **2018_ECCV_SiamFC-tri**
- **2018_ECCV_StructSiam**
- **2019_CVPR_CIR**
- **2019_CVPR_C-RPN**
- **2019_CVPR_SiamMask**
- **2019_CVPR_SiamRPN++**

## Survey
* **2017**: R. Pflugfelder. An in-depth analysis of visual tracking with siamese neural networks. arXiv:1707.00569, 2017[[paper](https://arxiv.org/pdf/1707.00569.pdf)]  

## TODO  
- [ ] add road trip figure
- [ ] add link for paper&code&project
- [ ] add core analyses
- [ ] add benchmark comparison

## License
MIT
