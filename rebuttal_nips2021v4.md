【这里我们需要陈述整篇文章的贡献点，增加对R3的质疑的一个简要总结，主要包括nolvety 以及实验性能的问题】

We thank all reviewers for their valuable comments.  This paper proposes a unified framework that..., and all reviewers are positive and interested in this novel settings, as far as we know, we are the first to rethinking pretraining for semantic segmentation. In the following, we answer the concerns of each reviewer one by one. 

We aruge that there exists task gap when using classification based pretrained model like ImageNet for segmentation task, and hope to make good advantages of available annotations for segmentation pretraining. 

### **To Reviewer wmH9:**[这个是一对一回答，直接贴在每一个问题下面]

**Q1: [Discussion about multi-dataset training]:** MDP is different from type 2 method (multi-dataset training with unified label space) in both motivation and application scenarios. 1）type 2 is designed to train a unified network that is able to segment all objects from multi-dataset, while MDP is designed for pretraining, which acts as ImageNet pretrained model that is usually used in current computer vision tasks. 2) In order to facilitate training, type 2 method needs **manually integrating the label space from multi-dataset, which is labor-consuming and inevitable to introduce noise**.  3) as demonstrated in Mseg [a], tranining a unified segmentation network usually suffers lower performance comparing with that training for each dataset separately. The reason is that it is hard to optimze the pixel-level objective as the number of classes grows, and inevitabely suffers from long tail issue, i.e., the classes with rare samples are usually underfitted. While MDP maintains one prototype for each class regardless of the number of images. This ensures the contribution of small datasets and classes with rare data be not ignored.  4) In our preliminary experiments, we follow the same fine-tuned settings as in MDP, and treat multi-dataset training as pretranined model, but achieve much lower performance than ImageNet pretrained model, which motivates us to design MDP as pretrained strategies. 

[a]  Mseg: A composite dataset for multi-domain semantic segmentation.CVPR, 2020.

**Q2: [Adding this pre-training strategy on a well-pretraind backbone]:**  Good question, using pretrained backbone can de facto improve the performance, especially when the number of images used for segmentation pretraining is relative small (e.g., VOC+ADE20K, around 35k images). As shown in the table below, when using imagenet pretrained model for initialization, the performance of the pretraining strategy ( VOC+ADE20K, 100 epochs)  can be boosted from 73.32% to 75.70% when fine-tuned on VOC. Note that this paper targets at uncovering the effect of task related pretraining and simply decouple the influence of the pretrained model for fair comparisons. On the other hand, we think the advantage of ImageNet pretraining is negligible when the number of annotations used for pretraining is relatively large like using COCO, we will add the results in the revised version. 

| Method |              Pretrained Dataset              | VOC mIoU |
| :----- | :------------------------------------------: | :------: |
| MDP    |                  VOC+ADE20K                  |  73.32   |
| MDP    | ImageNet(first stage), VOC+ADE20K(2nd stage) |  75.70   |

### **To Reviewer Hbtb:**

**Q1: [The techniques for the cross dataset learning seems too heuristic. It may not generalize well to other settings]:**  

Thanks for your comments. The motivation of this paper is that 1) we think it is not optimal to make use of the ImageNet pretrained model for segmentation task due to the task gap that the pretraining is based on global classification while the downstream task is for local pixel level prediction, and aim at obtaining pretrainned model specific for segmentation. 2) Considering that the pixel-level annotation is time consuming and the scale is far less than that of image-level and bounding-box level ones, we hope to make use of off-the-shelf  annotations by unifying the pixel-level anotation cross datasets. **As far we we know, we are the first to unify multi-dataset for jointly pretraining, while without cubersome human internvention such as label mapping**. 

As for generalization, MDP is a general framwork, and can be easily extended to other tasks such as classification and detection, as long as we design a prototype for each class embedding with image-level or bounidng box level labels,  and the domain gap can be relieved via cross dataset mixing and sparse coding equipped in MDP.  Experiments also demonstrate the effectiveness of MDP. Notably, we test the transfer ability on COCO detection and instance segmentation using Mask R-CNN and the results are shown in the table below. Although our pre-trained model does not completely match the downstream model (detetion and instance segmentation), our method still surpasses the supervised and self-supervised ImageNet pretraining methods that make use of image-level features for model pretraining, which indicates that the task gap matters when considering pretraining strategy.

| Method     |  Pretrained Dataset  | Pretrained Task | Box AP for object detection | Mask AP for instance segmentation |
| :--------- | :------------------: | :-------------: | :-------------------------: | :-------------------------------: |
| Supervised |       ImageNet       |   image-level   |            38.9             |               35.4                |
| MoCo v2    |       ImageNet       |   image-level   |            39.3             |               35.7                |
| MDP        | VOC, ADE20K and COCO |   pixel-level   |            39.8             |               36.0                |

**Q2: [Using only COCO in the pretraining]:** 

Thanks for your comments. We compare the results of different pre-training settings for 100 epochs, as shown in the table below, COCO pre-training do achieves good performance since the COCO dataset is three times larger than the combination of other two datasets. However, we want to point out that:

2. This does not weaken our proposed MDP since it is meaningful  when the number of images across dataset is comparable, or as long as no one dataset  dominates the whole datasets. Notably, the advantges of MDP can be demonstrated via comparing the results of using only VOC and VOC+ADE20K (first and second row in the table below), where the number of images between VOC and ADE20K are comparable.
2. MDP is flexible and can be applied with only one dataset for pretraining. For comparison, we also list the result of directly using COCO for pretraining (third row) with customized pixel-level cross entropy loss, and the performance decreases a lot, which demonstrates the effetiveness of MDP (a special case with only one dataset).

| Method     | Pretrained Dataset | VOC mIoU |
| :--------- | :----------------: | :------: |
| MDP        |     VOC (10k)      |  70.92   |
| MDP        |  VOC, ADE20K(35k)  |  73.32   |
| Supervised |     COCO(110k)     |  68.48   |
| MDP        |     COCO(110k)     |  76.85   |

**Q3: [Considering the number of mask label instances, the COCO dataset may surpass Imagenet in terms of scale.]**

Thanks for pointing out this. Considering the annotation cost, we agree that it is unfair to simply compare the number of images that used for ImageNet pretraining and our MDP due to different annotation granularities. The highlight is that it is more suitable to design task specific pretrained model for segmentation, and the number of images needed for pretraining can be substaintially decreased, as shown in Table 1 in the original paper, using VOC+ADE20K(around 35k images) for pretraining, MDP obtains better performance than ImanegNet pretrained model with class labels (ADE20K:+1.47%, Cityscapes: +1.29%).  We have adjusted the expression in the revised version accordingly.

**Q4: [The sparse coding part needs more analysis and justification]**

The sparse coding module is designed to consider the inter-class similarity especially for cross datasets for better transferability and we find that the pixle-to-prototype cross dataset sparse coding is beneficial to increase the intra-class diversity, and is **especially beneficial for downstream tasks that have non-overlap labels as the pretraining stage**. In particualr, we adapt two different settings：CSC-1) Not pushing the pixel and its top-k similar prototype far away;  CSC2) pulling the pixel and its top-k similar prototype close (sparse coding). The results are shown in the table below. CSC1 does not directly push the class of VOC away from other similarity class of VOC or other datasets, which makes its results on VOC worse. However, the results on Cityscapes are still comparable. This indicates that even if the features between the categories are not completely pushed away, making the features within one category distinguishable can still maintain a good accuracy in the unseen downstream. CSC2 2 increases intra-classes discrimination while ensuring inter-class discrimination. This makes the performance of VOC still comparable, and the performance of Cityscapes been further improved. During our experiment, the training loss in the finetune stage of CSC2 2 is much lower than that in CSC1, which indicates that our sparse coding scheme is effective.【你这里的setting1 还是奇奇怪怪的，所有的pushing都没有？到底想表达啥意思】

| Method        | Pretrained Dataset | VOC mIoU | Cityscapes mIoU |
| :------------ | :----------------: | :------: | :-------------: |
| MDP           |   VOC and ADE20K   |  71.98   |      75.37      |
| MDP with CSC1 |   VOC and ADE20K   |  70.88   |      75.50      |
| MDP with CSC2 |   VOC and ADE20K   |  71.84   |      76.53      |



### **To Reviewer qZDT:**

**Q1: [Technical novelty and contribution]:** We think the main innovation of our work lies in the proposed Multi-dataset pretraining framework. Most of the previous work was based on ImageNet pre-training model and did not consider the difference between the pretraining task and the downstream task at all. Our work is the first to focus on this problem. It can integrate and utilize multiple datasets with completely different label spaces to learn discriminable features and has versatility and wide application value. 

In addition, our methods have been improved and innovatively applied to the problems of our framework. We believe that this part of the contribution cannot be ignored.  Among these methods, the pixel-to-prototype contrastive learning we proposed is mainly to alleviate the huge storage space overhead in the case of multiple datasets.  On the one hand,  continuing to use pixel-to-pixel or pixel-to-region contrastive learning will cause huge storage overhead, which is unbearable in a multi-datasets scenario because a sufficient number of features need to be stored to reflect the overall category distribution of the multi-datasets. On the other, due to the limitation of GPU memory when calculating similarity,  pixel-to-pixel or region-to-pixel contrastive loss cannot be calculated efficiently under the multi-datasets setting. Our pixel-to-prototype method also achieved better performance in experiments. We think this is because compared to pixel embeddings, our class prototype is more stable and less susceptible to mislabeling.  In addition, muti-dataset pretraining introduces a new long-tail problem due to the difference in the number of samples between datasets. Our method can also alleviate this problem since only one prototype for each class is stored regardless of the number of images containing the class. We will further add more experiments about these benefits.

Similarly, our region-level mixing and pixel-level mixing are also innovative applied in new scenarios, which can alleviate the domain gap between datasets at both image and class levels. Different from [a], we have introduced different level of mixing operations on images to further boost diversity.  We also consider the gap of multiple datasets at the class level and generate the "intermediate class" of the two classes from different datasets through pixel-level class mixing. This is also a problem that is not involved in other work, and we can alleviate it through uncomplicated methods. 【这一部分一定要谨慎，问题， 分条陈述， 不要引用原文，直接把观点列上，后续有必要要放在整个rebuttal的开头】

[a] DACS: Domain Adaptation via Cross-domain Mixed Sampling, WACV'21

**Q2: [Cross-class sparse coding does not seem reasonable]:** 

同review2 Q4

**Q3: [The dataset comparison is not fair as the paper uses pixel-level annotations]** 

Thanks for pointing out this. 

前面有一部分同review2 Q3.

The pretraining and downstream tasks include some of the same datasets, mainly due the the lack of available segmention dataset. In order to better understand MDP, we also evaluate the performance on Cityscapes, where the model does not see any images over this dataset during pretraining, and the results are surprisingly promising (add results).  

**Q4: [The authors mention that they do not unify the label space, but it is not reflected]:** Thanks for pointing out this. The label unification in our paper means **mapping the labels between different datasets to a unified label space to tackle the label inconsistency among datasets, which requires manual definition and alignment**. For example, mapping the classes ’wall-brick’, ’wall-concrete’ and ’wall-panel’ in COCO to ’wall’ in ADE20K and to ’background’ in VOC.  While MDP just concatenates the label space of different datasets without any manual intervention. This allows our method to efficiently deal with multi-datasets problems. In fact, we also tried the label rough unification on the basis of MDP in our prelimilary experiments (only the data preprocessing is slightly different) but we did not see any improvement. We analyze that it is the feature-remapping in the fine-tuning stage that automatically realizes the label re-unification.

| Method | Pretrained Dataset | Label Unification | VOC mIoU |
| :----- | :----------------: | :---------------: | :------: |
| MDP    |   VOC and ADE20K   |        No         |  71.93   |
| MDP    |   VOC and ADE20K   |        Yes        |  71.91   |

**Q5: [The improvement is marginal considering the improvement of simply using COCO pre-trained weight]:** 

**The reviewer may misunderstand our settings**.The semantic segmentation pipelines like Deeplab[a] can indeed achieve around 2% improvement by using the COCO pre-trained weights, but **are totally different settings as in our paper**:

1. First, the COCO pretraining is based on **a well pretrained model using ImageNet **, while our MDP is trained from scratch.  We find that MDP can also achieve better performance when initialized with ImageNet pretrained model when the number of  pretraining is relatively small (e.g., using VOC and ADE20K for pretraining), please refer to Q2 for R1 for more details. 
2. Second, the COCO pretraining undergos deliberately preprocessing to adapt to VOC, i.e.,  only the images containing the classes defined in VOC are selected for pretraining and the COCO classes that not defined in VOC are all treated as background class.  In contrast, our method does not perform any selection and label processing operation on COCO data.

In order to better understand MDP, we also further explore the results of using similar settings: 1) Training on the whole COCO dataset from scratch, without label processing operation;  2) Training on the whole COCO dataset without label processing operation based on ImageNet pretraining. For fair comparsions, the COCO training lasts 80k iteration and the finetune step remains the same, and results are shown in the table below. The performance is much lower than our method, and we also notice that: 1) ImageNet pretraining is necessary. Training COCO from scratch and then finetuning on VOC obtain worse performance. 2) Data selection and label processing are important. We notice that the loss of our exploration setting (without label processing) is relatively low at the beginning of the finetune stage, but then it has been fluctuating at a higher value for a long time. We think this is due to the label space gap between COCO and VOC. Compared to the above setting, our method aims at making the features distinguishable, which results in much better feature representation in pretraining.

| Method                      | Pretrained Dataset | VOC mIOU |
| :-------------------------- | :----------------: | :------: |
| Supervised training on COCO |        None        |  68.48   |
| Supervised training on COCO |      ImageNet      |  74.51   |

**Q6: [The ablation study in Table 2  and Table 3  does not seem complete and are all worse than ImageNet pretraining]:** Both Tabels are ablation studies and only using VOC (ADE20K) with 100 epochs for efficiency, so the results are not comparable with ImageNet pretraining. Note that it is the relative numbers that matters to help conduct experiments over larger dataset and with more training epochs. Notably, the goal of this paper is to validate the effectivness of MDP, and the parameters are simply followed from the small dataset experiments, we do not carefully tune the parameters for better performance . We think it is resonable for research considering the complexity of the complete model.  

**Q7: [Some terms are quite vague and cannot provide the audience a good motivation and a big picture of the proposed method]**  Thanks for your suggestions, we have carefully proofread the paper in the revised version.

[a] Rethinking atrous convolution for semantic image segmentation. arxiv,2017

---------------------
### **To Reviewer 6obc:**

**Q1: [Discussion about DenseCL, PixPro and CAC]:**  Both DenseCL, PixPro and CAC introduce contrastive loss at pixel level. However, due to the lack of annotations, their methods need to define which pixels are positive samples of each other. For a pixel in the Images, DenseCL regards its most similar pixel on the other view as the positive sample while the remaining pixels as negative samples. PixPro considers the spatial relationship and treats pixels within a certain spatial range of the current pixel as positive samples.  The above two methods are target at ImageNet-based pretraining, while CAC is for semi-supervised semantic segmentation. Their pixel-level contrast learning is only performed on unlabeled images, and only pixels at corresponding positions in different views are regarded as positive sample pairs. CAC additionally adds directionality to pixel-level contrast learning through the confidence of the semi-supervised prediction. These methods obviously introduce a large number of false supervision signals and cannot be extended over multiple images.  Different from these methods, our method can directly use pixel-level labels to distinguish positive and negative samples,  which does not introduce too much noise, so we can achieve better performance with fewer data. Since the main purpose of our work is to explore the possibility of joint pre-training with multiple datasets, our improvement to the pixel-level understanding method is also mainly to adapt to this new scenario. We extend the pixel-level contrast to prototype-level and make cross datasets feature learning feasible.



**Q2: [Some of the performance comparisons are not that convincing]:** Our evaluation is based on MMsegmentation, which is a representative framework recognized by the community. **Our evaluation strategy strictly follows the setting of MMsegmentation**. The results look not well because we have made some modifications to the network structure in order to better verify the effectiveness of the method:

1. We use a standard ResNet-50 backbone other than ResNet-50 v1c In order to compare our results with ImageNet pretraining methods.

2. We remove the auxiliary head to better show the effect of backbone pre-training, while the auxiliary head is very helpful for the segmentation performance improvement.

We admit that changing to a better backbone or adding some additional tricks can further improve the accuracy,  but this is out of the scope of this paper.



**Q3: [Will the results be further boosted with longer training iterations]：** Yes, the results will be further boosted with longer training iterations.  However, as we target to demonstrating the effectiveness of multi-dataset pretraining, we do not further training the model for convenience in our paper. The results in the table below may shed light on the boosting of longer training, where we pretrain our model on the VOC dataset and use pixel-to-prototype contrastive loss. With the increase of epochs, we view better experimental results. 

| Method             | Pretrained Dataset | Epoch | VOC mIoU |
| :----------------- | :----------------: | :---: | :------: |
| Pixel-to-Prototype |        VOC         |  100  |  69.72   |
| Pixel-to-Prototype |        VOC         |  200  |  70.56   |
| Pixel-to-Prototype |        VOC         |  400  |  72.26   |

### **To Reviewer  LjH9:**

**Q1: [The presentation is not very good]:** Thanks for your detailed comments, we have carefully proofread the paper in the revised version.

**Q2: [Pixel-to-Pixel baseline]:** Thanks for your suggestion. we add an experiment to test the pixel-to-pixel baseline on three different downstream. The model was pre-trained for 100 epochs using VOC and ADE20K and the results are reported in the table below. The results reveal that our MDP achieves performance gain on all three datasets. We have added these results in the revised version.


| Method         | Pretrained Dataset | VOC mIoU | ADE20K mIoU | Cityscpases mIoU |
| :------------- | :----------------: | :------: | :---------: | :--------------: |
| Pixel-to-Pixel |   VOC and ADE20K   |  71.19   |    38.81    |      77.17       |
| MDP            |   VOC and ADE20K   |  73.32   |    40.15    |      77.75       |

**Q3: [How does a simple merging strategy address such a complex problem when dealing with multiple datasets]:** 

同review3 Q4

**Q4: [In region-level mixing, how are binary masks constructed]：** The mask $M$ can be represented by bounding box coordinates $\mathbf{B}=\left(r_{x}, r_{y}, r_{w}, r_{h}\right)$.  Among them, $r_{x}$ and $r_{y}$ are uniformly sampled within the width scope and height scope of the image respectively to determine the start points of the bounding box:

$r_{x} \sim$ Unif $(0, W)$, $r_{y} \sim$ Unif $(0, H)$

$r_{w}$ and $r_{h}$ determine the end points of the bounding box, and they are set according to a combination ratio $\lambda$ to make the cropped area ratio $\frac{r_{w} r_{h}}{W H}=1-\lambda$:

$r_{w}=W \sqrt{1-\lambda}$, $r_{h}=H \sqrt{1-\lambda}$

### **To Reviewer  9Qx8 :**

**Q1: [Why the pixel-to-pixel loss in Equation (2) cannot simply be formulated over multiple images as well]:** 

We have tried to extend pixel-to-pixel loss over multiple images, but our model was collapsed under this setting. We discuss the difficulty of extending pixel-to-pixel loss and analysis why the collapse occurs below.

1. Calculating pixel-to-pixel loss across images needs to store the pixels of all the images in the memory bank, which is hard to tolerate.
2. An alternative is to only perform pixel-to-pixel contrastive learning on the images in a batch. However, the similarity comparison of all pixels requires large GPU memory,  so it cannot be performed under a large batch size. 
3. If we set a small batch size, the images in the batch may come from totally different datasets and a large number of inappropriate negative samples will be introduced, which finally leads to the training collapse.

**Q2: [How does the merge of the label spaces work]:** 

同review3 Q4

**Q3: [Wording and expression need to be revised]：**  Thanks for your detailed comments, we have carefully proofread the paper in the revised version. Some more appropriate expressions have also been updated.

As to some other points:

In Ln 160,  "dirty data" means some pixels which have false labels. These “dirty data” may be caused by incorrect labeling of the dataset itself, or it may be introduced by the downsampling operation.

In Ln 169，"interactive" means pixels will be also pushed away or pulled close to the prototype of classes that do not exist in the current image.

**Q4: [Training a network on all four datasets with dataset-specific classification heads as a pre-training step]：** Thanks for your suggestion!  We think this is a result worth exploring and have added related experiments. For a fair comparison, we also pretrain the multi-head model over 100 epochs on VOC and ADE20k datasets, using the same learning rate and learning rate decay settings as MMsegmention. It can be seen from the table below that the performance of this scheme is far lower than our method (71.19% vs 73.32%). We think this is because this scheme does not model the correlation between the various datasets since each head is independent of each other.  In contrast, our MDP can perform joint learning across datasets, which makes our features more discriminative and obtains better performance. We have added these results in the revised version.

| Method     | Pretrained Dataset | VOC mIoU |
| :--------- | :----------------: | :------: |
| Multi-head |   VOC and ADE20K   |  71.19   |
| MDP        |   VOC and ADE20K   |  73.32   |

**Q5: [Evaluation on COCO-stuff-val ]：** Thanks for pointing out that. We did not report the test results on COCO because there is no community-recognized evaluation code on COCO-stuff. In fact, we also have our own implemented 80k iterations finetune results on the COCO dataset. As shown in the table below, our method still boosts supervised ImageNet pretraining results on COCO for a large margin, from 35.25% to 38.37%.

| Method     |  Pretrained Dataset  | COCO mIoU |
| :--------- | :------------------: | :-------: |
| Scratch    |          -           |   25.02   |
| Supervised |       ImageNet       |   35.25   |
| MDP        | VOC, ADE20K and COCO |   38.37   |

**Q6: [Segmentation annotations are much richer than ImageNet image-level annotations]：** 

同review2 Q3

We mentioned in Ln261 that our method is more efficient because the number of images we used is much smaller than that of ImageNet pretraining. Our code has not been fully optimized yet, but the current running time of a batch is basically the same as moco v2. Taking into account the reduction in the number of images, we think our method is more efficient. We will provide the detailed numbers in the revised version.

**Q7: [The sparse coding part needs more analysis and justification]：** 

同review2 Q4

他还要求额外补一组ade20k的结果(39.71vs39.47)

### **To Reviewer RRFs :**

(Q1和Q2可以整合到一起，作为对没有搭建一个公平baseline的回应)

**Q1: [Pretraining on segmentation dataset in sequential way: COCO->ADE20K->PASCAL VOC]:** We believe that the sequential training method is contrary to our intention of jointly training on multi-datasets, and since the final stage of pretraining is only performed on a specific dataset, its poor generalization ability is predictable. We add an experiment on this setting and the results reported in the table below confirm our view. Using the Sequential way achieves worse performance on all of the three downstream. Even on VOC, the results of sequential training are not good because of the overfitting problem.

| Method         |  Pretrained Dataset  | VOC mIOU | ADE20K mIoU | Cityscpaes MIOU |
| :------------- | :------------------: | :------: | :---------: | :-------------: |
| MDP            | VOC, ADE20K and COCO |  77.79   |    42.69    |      80.64      |
| Sequential way | VOC, ADE20K and COCO |  75.68   |    41.44    |      78.47      |

**Q2: [Co-training a shared model on segmentation datasets]:** 

同review6 Q4

**Q3: [Pretraining on COCO boosts the mIoU by +3% on PASCAL according to DeepLabV3]：** 

同review3 Q5

**Q4: [Compare to the multi dataset training when using any simple unified label spaces]：** 

同review2 Q4