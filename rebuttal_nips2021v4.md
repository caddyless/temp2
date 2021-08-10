We thank all reviewers for their valuable comments. The highlight is that it is not optimal in current computer vision tasks that makes use of the ImageNet pretrained model for downstream tasks like segmentation, since there exists task gap when using classification based pretrained model for segmentation task. Considering that the pixel-level annotations is time-consuming, we hope to make good advantages of the available, fragmented annotations for segmentation pretraining, while does not require any human intervention. As far as we know, we are the first to rethinking pretraining for semantic segmentation, and all reviewers are positive and interested in this novel settings. In the following, we answer the concerns of each reviewer one by one. 

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
2. MDP is flexible  and can be applied with only one dataset for pretraining. For comparison, we also list the result of directly using COCO for pretraining (third row) with customized pixel-level cross entropy loss, and the performance decreases a lot, which demonstrates the effetiveness of MDP (a special case with only one dataset).

| Method     | Pretrained Dataset | VOC mIoU |
| :--------- | :----------------: | :------: |
| MDP        |     VOC (10k)      |  70.92   |
| MDP        |  VOC, ADE20K(35k)  |  73.32   |
| Supervised |     COCO(110k)     |  68.48   |
| MDP        |     COCO(110k)     |  76.85   |

**Q3: [Considering the number of mask label instances, the COCO dataset may surpass Imagenet in terms of scale.]**

Thanks for pointing out this. Considering the annotation cost, we agree that it is unfair to simply compare the number of images that used for ImageNet pretraining and our MDP due to different annotation granularities. The highlight is that it is more suitable to design task specific pretrained model for segmentation, and the number of images needed for pretraining can be substaintially decreased, as shown in Table 1 in the original paper, using VOC+ADE20K(around 35k images) for pretraining, MDP obtains better performance than ImanegNet pretrained model with class labels (VOC: +2.16% ADE20K:+3.33%).  We have adjusted the expression in the revised version accordingly.

**Q4: [The sparse coding part needs more analysis and justification]**

The cross dataset sparse coding (CSC) module is designed to consider the inter-class similarity and intra-class diversity,  which we find is **especially beneficial for downstream tasks that have non-overlap labels as the pretraining stage**. In particualr, we adapt two different settings：MDP without CSC and the proposed MDP training strategy, and the results are shown in the table below (Table 4 in the original paper). MDP increases intra-classes diversity via pulling each pixel-level embeddings to its corresponding top-k class prototypes,  although the VOC results are comparable, the performance of Cityscapes can be further improved. Notably, **the training loss in the finetune stage of MDP is much lower than that without CSC for pretraining**, which indicates that our sparse coding scheme is effective.

| Method          | Pretrained Dataset | VOC mIoU | Cityscapes mIoU |
| :-------------- | :----------------: | :------: | :-------------: |
| MDP without CSC |   VOC and ADE20K   |  71.98   |      75.37      |
| MDP             |   VOC and ADE20K   |  71.84   |      76.53      |


### **To Reviewer qZDT:**

**Q1: [Technical novelty and contribution]:**

As for the contribution, this paper for the first time proposes a unified framwork that enables multi-dataset pretraining for segmentation. We aruge that there exists task gap when using classification based pretrained model like ImageNet for segmentation task, and considering that the pixel-level anotations is time-consuming, we hope to make good advantages of the available, fragmented annotations for segmentation pretraining, while does not require any human intervention. 

As for the tencnical details, although some modules are not firstly propsoed in our paper, we adapt them to facilitate MDP training. 

1) **For contrastive loss**: We adjust the original contrastive loss with pixel-level lables , and propose a pixel-to-prototype mapping to effectively model intra-class compactness and inter-class separability. Note that the original contrastive loss is designed under unsupervised setting, and the correspondence can only be ensured from different augmentations of the same image/pixel, while we are able to effieicntly model pixel-to-prototype mapping via pixel-level labels, which we find is beneficial for fast convergence and better transferablity. 
2) **For prototype generation and update**: we design an efficient strategy to maintain the prototype for each class. Considering thsat the pixel-level embeddings is considerably huge comparing with image-level ones, we rely on a two-step average strategy to efficiently store and dynamically update the class prototype, which is also benefical to alleviate the class imbalance problem especially when using multiple datasets and obtain more robust class embeddings.
3) **For cross dataset domain gap**: we propose region-level mixing and pixel-level mixing to alleviate the domain gap between datasets at both image and class levels. Different from [a], we introduced different level of mixing operations on images to further boost diversity.  We also consider the gap of multiple datasets at the class level and generate the "intermediate class" of the two classes from different datasets through pixel-level class mixing. This is also a problem that is not involved in other work, and we can alleviate it through such simple and efficient method. 

Integrating the above modules produces a powerful and general framework for multi-dataset pretraining, and we hope such training strategy would helpful to the community for designing pretrained models for downstream tasks. 

[a] DACS: Domain Adaptation via Cross-domain Mixed Sampling, WACV'21

**Q2: [Cross-class sparse coding does not seem reasonable]:** 

同review2 Q4

**Q3: [The dataset comparison is not fair as the paper uses pixel-level annotations]** 

Thanks for pointing out this. The pretraining and downstream tasks include some of the same datasets, mainly due the the lack of available segmention dataset. In order to better understand MDP, we also evaluate the performance on Cityscapes, where the model does not see any images over this dataset during pretraining, and the results are surprisingly promising (add results).  

**Q4: [The authors mention that they do not unify the label space, but it is not reflected]:** Thanks for pointing out this. The label unification in our paper means **mapping the labels between different datasets to a unified label space to tackle the label inconsistency among datasets, which requires manual definition and alignment**. For example, mapping the classes ’wall-brick’, ’wall-concrete’ and ’wall-panel’ in COCO to ’wall’ in ADE20K and to ’background’ in VOC.  While MDP just concatenates the label space of different datasets without any manual intervention. This allows our method to efficiently deal with multi-datasets problems. In fact, we also tried the label rough unification on the basis of MDP in our prelimilary experiments (only the data preprocessing is slightly different) but we did not see any improvement. We analyze that it is the feature-remapping in the fine-tuning stage that automatically realizes the label re-unification.

| Method | Pretrained Dataset | Label Unification | VOC mIoU |
| :----- | :----------------: | :---------------: | :------: |
| MDP    |   VOC and ADE20K   |        No         |  71.93   |
| MDP    |   VOC and ADE20K   |        Yes        |  71.91   |

**Q5: [The improvement is marginal considering the improvement of simply using COCO pre-trained weight]:** 

**The reviewer may misunderstand our settings**.The semantic segmentation pipelines like Deeplab can indeed achieve around 2% improvement by using the COCO pre-trained weights, but **are totally different settings as in our paper**:

1. First, the COCO pretraining is based on **a well pretrained model using ImageNet **, while our MDP is trained from scratch.  We find that MDP can also achieve better performance when initialized with ImageNet pretrained model when the number of  pretraining is relatively small (e.g., using VOC and ADE20K for pretraining), please refer to Q2 for R1 for more details. 
2. Second, the COCO pretraining undergos deliberately preprocessing to adapt to VOC, i.e.,  only the images containing the classes defined in VOC are selected for pretraining and the COCO classes that not defined in VOC are all treated as background class.  In contrast, our method does not perform any selection and label processing operation on COCO data.

We also evaluate the performance of training on the whole COCO dataset from scratch, without label preprocessing or make use of ImageNet pretrained model for fair comparison. The performance is much lower than MDP, as shown in the table below. 

| Method                                     | Pretrained Dataset |  VOC  |
| :----------------------------------------- | :----------------: | :---: |
| Supervised (customized cross entropy loss) |        COCO        | 68.48 |
| MDP                                        |        COCO        | 76.85 |

**Q6: [The ablation study in Table 2  and Table 3  does not seem complete and are all worse than ImageNet pretraining]:** Both Tabels are ablation studies and only using VOC (ADE20K) with 100 epochs for efficiency, so the results are not comparable with ImageNet pretraining. Note that it is the relative numbers that matters to help conduct experiments over larger dataset and with more training epochs. Notably, the goal of this paper is to validate the effectivness of MDP, and the parameters are simply followed from the small dataset experiments, we do not carefully tune the parameters for better performance . We think it is resonable for research considering the complexity of the complete model.  

**Q7: [Some terms are quite vague and cannot provide the audience a good motivation and a big picture of the proposed method]**  Thanks for your suggestions, we have carefully proofread the paper in the revised version.

---------------------

### **To Reviewer 6obc:**

**Q1: [Discussion about DenseCL, PixPro and CAC]:**  Both DenseCL and PixPro introduce contrastive loss at pixel-level. However, due to the lack of annoataions, their methods need to define which pixels are positive samples of each other. For a pixel in the Images, DenseCL regards the most similar pixel as the positive sample of the pixel while the remaining pixels as negative samples. PixPro considers the spatial relationship and treats pixels within a certain spatial range of the pixel as positive samples.  CAC is for semi-supervised semantic segmentation. In fact, it was published after our paper was submitted. Their pixel-level contrast learning is only performed on unsupervised images, and only pixels at corresponding positions on different views are used as positive sample pairs. CAC additionally adds directionality to pixel-level contrast learning through the confidence of the semi-supervised prediction. These methods obviously introduce a large number of false supervision signals and cannot be extended across images.  Our method directly uses pixel-wise labels to divide positive and negative samples, and extends the method to cross-image contrast learning. Therefore, our method only needs less data to achieve better performance. Our method can also be easily extended semi-supervised setting.

**不要引用原文，这里可以加上数据挖掘的部分**



**Q2: [Some of the performance comparisons are not that convincing]:** Our evaluation is based on MMsegmentation, which is a representative framework recognized by the community. Our evaluation strategy strictly follows the setting of MMsegmentation. The results looks not well because we have made some modifications to the network framework in order to better verify the effectiveness of the method:

1. We use the a standard ResNet-50 backbone other than ResNet-50 v1c In order to compare our results with Imagenet pretraining.

2. We remove the auxiliary head to better show the effect of backbone pre-training, while the auxiliary head is very helpful for the segmentation performance improvement.

We admit that changing to a better backbone or adding some additional tricks can further improve the accuracy,  but this is out of the scope of this paper.

**Q3: [Will the results be further boosted with longer training iterations]：** Yes, the results will be further boosted with longer training iterations.  However, as we traget at demostrating the effectiveness of multi-dataset pretraining, we do not further training the model for convenience in our paper. The results in the table below may shed light on the boosting of longer training, where we pretrain our model on VOC dataset and use pixel-to-prototype contrastive loss. With the increase of epochs, we view better experimental results. 

| Method             | Pretrained Dataset | Epoch | VOC mIoU |
| :----------------- | :----------------: | :---: | :------: |
| Pixel-to-Prototype |        VOC         |  100  |  69.72   |
| Pixel-to-Prototype |        VOC         |  200  |  70.56   |
| Pixel-to-Prototype |        VOC         |  400  |  72.26   |

### **To Reviewer  LjH9:**

**Q1: [The presentation is not very good]:** Thanks for your detailed comments, we have carefully proofread the paper in the revised version.

**Q2: [Pixel-to-Pixel baseline]:** Thanks for your suggestion. we add an experiment to test the results of pixel-to-pixel baseline on three different downstream. The model was pre-trained for 100 epochs using VOC and ADE20K dataset and the results are reported in the table below. Our MDP achieves performance gain on both of three datasets. We have added this results in the revised version.


| Method         | Pretrained Dataset | VOC mIoU | ADE20K mIoU | Cityscpases mIoU |
| :------------- | :----------------: | :------: | :---------: | :--------------: |
| Pixel-to-Pixel |   VOC and ADE20K   |  71.19   |    38.81    |      77.17       |
| MDP            |   VOC and ADE20K   |  73.32   |    40.15    |      77.75       |

**Q3: [How does a simple merging strategy address such a complex problem when dealing with multiple datasets]:** 

同review3 Q4

**Q4: [In region-level mixing, how are binary masks constructed]：** The mask $M$ can be represented by bounding box coordinates $\mathbf{B}=\left(r_{x}, r_{y}, r_{w}, r_{h}\right)$.  Among them, $r_{x}$ and $r_{y}$ are uniformly sampled within the width scope and height scope of the image respectively to determine the start point of the bounding box:

$r_{x} \sim$ Unif $(0, W)$, $r_{y} \sim$ Unif $(0, H)$

$r_{w}$ and $r_{h}$ determine the end point of the bounding box, and it is set according to a combination ratio $\lambda$ to make the cropped area ratio $\frac{r_{w} r_{h}}{W H}=1-\lambda$:

$r_{w}=W \sqrt{1-\lambda}$, $r_{h}=H \sqrt{1-\lambda}$

### **To Reviewer  9Qx8 :**

**Q1: [Why the pixel-to-pixel loss in Equation (2) cannot simply be formulated over multiple images as well]:** 

We have tried to expand pixel-to-pixel loss across images, but our model was collapsed under this setting. We discuss the difficulty of expanding pixel-to-pixel loss across images and analysis why the collapse occurs below.

1. Calculating pixel-to-pixel loss across images needs to store all the pixels of all the images in the memory bank, which is hard to tolerate.
2. An alternative is to only perform pixel-to-pixel contrastive learning on the images in a batch. However, the similarity comparison of all pixels requires large GPU memory,  so it cannot be performed under a large batch size. 
3. If we set a small batch size, the images in the batch may come from totally different datasets and a large number of inappropriate negative samples will be introduced, which leads to the training collapse.

**Q2: [How does the merge of the label spaces work]:** 

同review3 Q4

**Q3: [Wording and expression need to be revised]：**  Thanks for your detailed comments, we have carefully proofread the paper in the revised version. Some more appropriate expressions have also been updated.

As to some other points:

In Ln 160,  "dirty data" means some pixels which have false labels. These “dirty data” may be caused by incorrect labeling of the data setitself, or it may be introduced by downsampling operation.

In Ln 169， "interactive" means pixels will be also pushed away or pulled close to class prototypes that classes are not in the current image.

**Q4: [Training a network on all four datasets with dataset-specific classification heads as a pre-training step]：** Thanks for your suggestion!  We think this is a result worth exploring and have added related experiments. Based on VOC and ADE20K dataset, we use the same learning rate and learning rate decay settings as MMsegmention, and unify the training period to 100 epochs for easy comparison. It can be seen from the table below that the performance of this scheme is far lower than our method (71.19% vs 73.32%). We think this is because this scheme does not model the correlation between the various datasets, and each head is independent of each other.  Our MDP can perform joint learning across datasets, which makes our features more discriminative. We have added this results in the revised version.

| Method     | Pretrained Dataset | VOC mIoU |
| :--------- | :----------------: | :------: |
| Multi-head |   VOC and ADE20K   |  71.19   |
| MDP        |   VOC and ADE20K   |  73.32   |

**Q5: [Evaluation on COCO-stuff-val ]：** Thanks for pointing out that. We did not report the test results on COCO because MMsegmentaion did not provide COCO evaluation code and we worry that the results of our report are not representative（浩航师兄觉得没必要提这个）. In fact, we also have our own implemented finetune results of 80k iterations on the COCO dataset. As shown in the table below, our method also boosts supervised ImageNet pretraining results on COCO for a large margin, from 35.25% to 38.37%.

| Method     |  Pretrained Dataset  | COCO mIoU |
| :--------- | :------------------: | :-------: |
| Scratch    |          -           |   25.02   |
| Supervised |       ImageNet       |   35.25   |
| MDP        | VOC, ADE20K and COCO |   38.37   |

**Q6: [Segmentation annotations are much richer than ImageNet image-level annotations]：** 

同review2 Q3

We mentioned in Ln261 that our method is more efficient because the number of images we used is much smaller than that of ImageNet pretraining. Our code has not been fully optimized, but the current running time of a batch is basically the same as moco v2. Taking into account the reduction in the number of images, our method is more efficient. We will provide the detailed numbers in the revised version.

**Q7: [The sparse coding part needs more analysis and justification]：** 

同review2 Q4

他还要求额外补一组ade20k的结果(39.71vs39.47)

### **To Reviewer RRFs :**

(Q1和Q2可以整合到一起，作为对没有搭建一个公平baseline的回应)

**Q1: [Pretraining on segmentation dataset in sequential way: COCO->ADE20K->PASCAL VOC]:** We believe that the sequential training method is contrary to our intention of joint training on multi-datasets, and since the final stage of pretrain is only performed on a specific dataset, its poor generalization ability is predictable. We add an experiment on this setting and the results reported in the table below confirm our view. Using the Sequential way achieves worse performance on all of the three downstream. Even in the downstream of VOC, the results of sequential training are not good because of overfitting the training set.

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
