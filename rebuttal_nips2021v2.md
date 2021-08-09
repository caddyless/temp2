We thank all reviewers for the valuable comments.  We have carefully proofread the paper in the revised version, and answer the reviewers' concerns below.

### **To Reviewer wmH9:**

**Q1: [Discussion about multi-dataset training]:** Our method cannot directly compare the performance with the multi-dataset training (type 2.) method, because this type of method is not designed for pre-training, but instead trains a unified downstream model by manually integrating the label space. This type of method usually has the following characteristics: 1) Massive manual operations are required for label unification. 2)The unified model can achieve good results in each downstream, but they are all worse than the results of training each downstream separately.  We believe that this type of method does not make full use of the advantages of multi-datasets, and their process is too cumbersome.

To further answer your question,  we have designed a new pre-training scheme based on multi-dataset training, that is, co-training a shared model with dataset-specific classification heads and then finetuning at different downstream. This scheme is a lit bit like [33] but without ImageNet pre-training and subsequent label unification for a fair comparison. The results are shown in the table below.  It can be seen that if manual label integration is not performed, the performance of this scheme is far lower than our method (71.19% vs 73.32%) because this scheme cannot model the relationship between multi-datasets.  The above discussion proves the effectiveness of our MDP pipeline towards multi-dataset training.（这一段要加吗？会不会有点奇怪？但没了感觉太单薄了...）

| Method     | Pretrained Dataset |  VOC mIoU |
| :--------- | :----------------: | :-------: |
| Multi-head |   VOC and ADE20K   |   71.19   |
| MDP        |   VOC and ADE20K   |   73.32   |

**Q2: [Adding this pre-training strategy on a well-pretraind backbone]:**  Thanks for pointing out this. We add our pre-training strategy on supervised ImageNet pretrained models. We pretrain our model on VOC and ADE20K  dataset for 100 epochs. The results are shown in the table below. The results reveal that using a two-stages pretraining strategy can boost the one-stage pre-training for a large margin, from 73.32% to 75.7%. We will added more ablation study about it in the revised version.

| Method     |               Pretrained Dataset               | VOC mIoU  |
| :--------- | :--------------------------------------------: | :---: |
| MDP        |                 VOC and ADE20K                 | 73.32 |
| MDP        | ImageNet(1st stage), VOC and ADE20K(2nd stage) | 75.70 |

### **To Reviewer Hbtb:**

**Q1: [The techniques for the cross dataset learning seem too heuristic]:** We admit that our model is mainly for semantic segmentation tasks and our method may not work if the dataset gap for pre-training is too large (e.g. adding some medical segmentation datasets). However, our model can also achieve great performance in other downstream, and our method can be further adapted to scenarios such as data mining. From this perspective, we think our model still has great generalization ability. As an example, we test the transfer learning accuracy on COCO detection and COCO instance segmentation using Mask R-CNN and the results are shown in the table below. Although our pre-training model does not completely match the downstream model and can only load part of the parameters, our method still surpasses the supervised and self-supervised ImageNet pretraining method. The results prove the generalization ability of MDP pretraining models.

| Method     |  Pretrained Dataset  | Box AP for object detection | Mask AP for instance segmentation |
| :--------- | :------------------: | :-------------------------: | :-------------------------------: |
| Supervised |       ImageNet       |            38.9             |               35.4                |
| MoCo v2    |       ImageNet       |            39.3             |               35.7                |
| MDP        | VOC, ADE20K and COCO |            39.8             |               36.0                |

**Q2: [Using only COCO in the pretraining]:** (浩航师兄觉得这一块coco太高了...我的解释写的不太好...但是我们感觉也不太好改）Thanks for your suggestions! We compare the results of three different settings over 100 epochs pre-training in the following table: 1) MDP pretraining using only VOC dataset(~10k images), 2) MDP pretraining using VOC and ADE20K dataset (~30k images) 3) MDP pretraining using only COCO dataset(~100k images). The COCO pre-training results do achieve high performance since the COCO dataset is three times larger than the combination of other two datasets. However, we want to emphasize the advantages of our method when the amount of data is comparable. Compared with the first setting using only VOC dataset, our 'VOC+ADE20K' configuration has achieved a huge performance improvement. At the same time, this configuration has surpassed the results of supervised ImageNet pretraining on the ADE20k and Cityscapes downstream, shown in Tab. 1 of our paper. We think these results can prove the benefits of our MDP.

| Method | Pretrained Dataset | VOC mIoU  |
| :----- | :----------------: | :---: |
| MDP    |        VOC         | 70.92 |
| MDP    |   VOC and ADE20K   | 73.32 |
| MDP    |        COCO        | 76.85 |

**Q3: [Considering the number of mask label instances, the COCO dataset may surpass Imagenet in terms of scale.]** 

Yes, we admit that the data scale can not be comparable to the scale of data annotation and maybe we should use a more appropriate expression here. However, it is precise because of the difficulty of labeling that a single large-scale pixel-level labeled dataset is difficult to obtain. Our method provides a co-pretraining strategy using data from different sources and from the perspective of data amount, our performance is very outstanding.  In addition, we can also utilize the class prototype from a small amount of pixel-level labeled data to perform data mining using unlabeled images.  The results below show toy experiment results using no-label COCO data, which reviews that using a smaller amount of unlabeled images (compared to the scale of ImageNet) can further improve the performance.

| Method |       Pretrained Dataset       | VOC mIoU  |
| :----- | :----------------------------: | :---: |
| MDP    |         VOC and ADE20K         | 73.32 |
| MDP    | VOC, ADE20K and COCO(no label) | 74.74 |

**Q4: [The sparse coding part needs more analysis and justification]**  （稀疏编码我强行解释了下...）Since taxonomy from different datasets differs, the sparse coding part is designed by utilizing inter-class similarity across different datasets to increase the discrimination of embeddings with the same label and to make the re-mapping easier in the downstream fine-tuning.  In short, we hope to use sparse coding to make features of the same class distinguishable intraclass. The weight of the loss of sparse coding is low so that the constraint only makes the feature diverse and does not occupy the dominant position.

During the experiment, we adopted two different settings：1）Not pushing the pixel and its top-k similar prototype far away;  2) pulling the pixel and its top-k similar prototype close (as mentioned in the paper). The results are shown in the table below. Setting 1 does not directly push the class of VOC away from other similarity class of VOC or other datasets, which makes its results on VOC worse. However, the results on Cityscapes are still comparable. This indicates that even if the features between the categories are not completely pushed away, making the features within the categories distinguishable can still maintain a good accuracy rate in the unseen downstream. Setting 2 increases intra-classes discrimination while ensuring inter-class discrimination. This makes the performance of VOC still comparable, and the performance of Cityscapes has been further improved. During our experiment, the loss in the finetune stage of Setting 2 has also been lower during the finetune process, which indicates that our sparse coding scheme is effective.

| Method        | Pretrained Dataset | VOC mIoU | Cityscapes mIoU |
| :------------ | :----------------: | :------: | :-------------: |
| MDP           |   VOC and ADE20K   |  71.98   |      75.37      |
| MDP with CSC1 |   VOC and ADE20K   |  70.88   |      75.50      |
| MDP with CSC2 |   VOC and ADE20K   |  71.84   |      76.53      |



### **To Reviewer qZDT:**

**Q1: [Technical novelty and contribution]:** We think the main innovation of our work lies in the proposed Multi-dataset pretraining framework. Most of the previous work was based on ImageNet pre-training model and did not consider the difference between the pretraining task and the downstream task at all. Our work is the first to focus on this problem. It can integrate and utilize multiple datasets with completely different label spaces to learn discriminable features and has versatility and wide application value. 

In addition, our methods have been improved and innovatively applied to the problems of our framework. We believe that this part of the contribution cannot be ignored.  Among these methods, the pixel-to-prototype contrastive learning we proposed is mainly to alleviate the huge storage space overhead in the case of multiple datasets.  On the one hand, as mentioned in the paper Ln150-155,  continuing to use pixel-to-pixel or pixel-to-region contrastive learning will cause huge storage overhead, which is unbearable in a multi-datasets scenario because a sufficient number of features need to be stored to reflect the overall category distribution of the multi-datasets. On the other, due to the limitation of GPU memory when calculating similarity,  pixel-to-pixel or region-to-pixel contrastive loss cannot be calculated efficiently under the multi-datasets setting. Our pixel-to-prototype method also achieved better performance in experiments. We think this is because compared to pixel embeddings, our class prototype is more stable and less susceptible to mislabeling.  In addition, muti-dataset pretraining introduces a new long-tail problem due to the difference in the number of samples between datasets. Our method can also alleviate this problem, as mentioned in the paper Ln156-158, only one prototype for each class is stored. We will further add more experiments about these benefits.

Similarly, our region-level mixing and pixel-level mixing are also innovative applied in new scenarios, which can alleviate the domain gap between datasets at both image and class levels. Different from [c], we have introduced different level of mixing operations on images to further boost diversity.  We also consider the gap of multiple datasets at the class level and generate the "intermediate class" of the two classes from different datasets through pixel-level class mixing. This is also a problem that is not involved in other work, and we can alleviate it through uncomplicated methods.

**Q2: [Cross-class sparse coding does not seem reasonable]:** 

同review2 Q4

**Q3: [The dataset comparison is not fair as the paper uses pixel-level annotations]** 

同review2 Q3

As to the probelm of including the dataset of the final testing data，we discuss about it in the Ln262-267 and Ln299-306 of paper, we will also add more unseen downstream in the future.

**Q4: [The authors mention that they do not unify the label space, but it is not reflected]:** The label unification in our paper means mapping the labels between different datasets to a unified label space to tackle the label inconsistency between datasets, which requires a lot of manual definition and alignment. However, just as indicated in Ln 134, our method just concatenates the label space of different data sets without any manual integration and alignment. This allows our method to better deal with multi-datasets problems. In fact, we also tried the label rough unification on the basis of MDP at the early stages (only the data preprocessing is slightly different from now) but we did not see any improvement. We analyze that it is the feature-remapping in the fine-tuning stage that automatically realizes the label re-unification.

| Method | Pretrained Dataset | Label Unification | VOC mIoU  |
| :----- | :----------------: | :---------------: | :---: |
| MDP    |   VOC and ADE20K   |        No         | 71.93 |
| MDP    |   VOC and ADE20K   |        Yes        | 71.91 |

**Q5: [The improvement is marginal considering the improvement of simply using COCO pre-trained weight]:** 

We think your description is inappropriate and the comparison is unfair. The semantic segmentation pipelines like Deeplab [] can indeed achieve around 2% improvement by using the COCO pre-trained weight, but:

1. The COCO pretraining is based on a well ImageNet-pretrained model while our MDP is trained from scratch.  Their improvement is based on the application of ImageNet while our pipeline is not. 
2. The COCO data for pretraining is after careful selection and processing. Only the images containing the classes defined in VOC are selected and the COCO classes not defined in VOC are all treated as background classes.  In contrast, our method does not perform any selection and label processing operation on COCO data.

We also further explored the results using some similar settings: 1) Training on the whole COCO dataset without label processing operation from scratch. 2) Training on the whole COCO dataset without label processing operation based on ImageNet pretraining.  The COCO training lasts 80k iteration and the finetune is conducted on VOC for 40k iteration. Results are shown in the table below. The performance is far from satisfactory compared to our method, and we also notice that: 1) ImageNet pretraining is necessary. Training COCO from scratch and then finetuning on VOC obtain worse performance. 2) Data selection and label processing are important. We notice that the loss of our exploration setting (without label processing) is relatively low at the beginning of the finetune stage, but then it has been fluctuating at a higher value for a long time. We think this is due to the label space gap between COCO and VOC. Compared to the above setting, our method aims at making the features distinguishable, which results in much better performance.

| Method                      | Pretrained Dataset | VOC mIOU  |
| :-------------------------- | :----------------: | :---: |
| Supervised training on COCO |         -          | 68.48 |
| Supervised training on COCO |      ImageNet      | 74.51 |

**Q6: [The ablation study in Table 2  and Table 3  does not seem complete and are all worse than ImageNet pretraining]:** Table 2 studies on hyper-parameters and the type of memory bank using only VOC dataset so the results are worse tham ImageNet pretraining. We think that the setting of hyper-parameters can be explored in a single dataset for efficiency since directly adding all segmentation datasets for ablation study is too expensive.  Table 3 studies the cross-dataset mixing strategies. We have reported the results of using VOC and ADE20K in the paper and we believe that the huge improvement in the results can prove the effectiveness of our strategies. From another perspective, the results gap between using a single dataset like VOC and using ImageNet also proves the effectiveness of our MDP.

**Q7: [Some terms are quite vague and cannot provide the audience a good motivation and a big picture of the proposed method.]**  Thanks for your detailed comments, we have carefully proofread the paper in the revised version. Some more appropriate expressions have also been updated.



---------------------

### **To Reviewer 6obc:**

**Q1: [Discussion about DenseCL, PixPro and CAC]:**  Both DenseCL and PixPro introduce contrast learning methods to pixel-level. However, due to their unsupervised settings, their methods need to define which pixels are positive samples of each other. For a pixel in the Images, DenseCL regards the most similar pixel as the positive sample of the pixel while the remaining pixels as negative samples. PixPro considers the spatial relationship and treats pixels within a certain spatial range of the pixel as positive samples.  CAC is for semi-supervised semantic segmentation. In fact, it was published after our paper was submitted. Their pixel-level contrast learning is only performed on unsupervised images, and only pixels at corresponding positions on different views are used as positive sample pairs. CAC additionally adds directionality to pixel-level contrast learning through the confidence of the semi-supervised prediction. These methods obviously introduce a large number of false supervision signals and cannot be extended across images.  In fact, the scheme of directly using pixel labels for supervised contrastive learning has also been explored and it is mentioned in Ln72-76 of the paper.

**Q2: [Some of the performance comparisons are not that convincing]:** Our evaluation is based on MMsegmentation, which is a representative framework recognized by the community. Our evaluation strategy strictly follows the setting of MMsegmentation. The results looks not well because we have made some modifications to the network framework in order to better verify the effectiveness of the method (also mentioned in Ln239-247 of the paper):

1. We use the a standard ResNet-50 backbone other than ResNet-50 v1c In order to compare our results with Imagenet pretraining.

2. We removed the auxiliary head to better show the effect of backbone pre-training, while the auxiliary head is very helpful for the segmentation performance improvement.

We admit that changing to a better backbone or adding some additional tricks can further improve the accuracy,  but this is out of the scope of this paper.

**Q3: [Will the results be further boosted with longer training iterations]：** Yes, the results will be further boosted with longer training iterations. However, as our implementation aims to show the effectiveness of multi-dataset pretraining, we do not further training the model for convenience in our paper.  The results in the table below may shed light on the boosting of longer training, where we pretrain our model on VOC dataset and use pixel-to-prototype contrastive loss. With the increase of epochs, we view better experimental results.

| Method             | Pretrained Dataset | Epoch | VOC mIoU  |
| :----------------- | :----------------: | :---: | :---: |
| Pixel-to-Prototype |        VOC         |  100  | 69.72 |
| Pixel-to-Prototype |        VOC         |  200  | 70.56 |
| Pixel-to-Prototype |        VOC         |  400  | 72.26 |

### **To Reviewer  LjH9:**

**Q1: [The presentation is not very good]:** Thanks for your detailed comments, we have carefully proofread the paper in the revised version. Some more appropriate expressions have also been updated.

**Q2: [Pixel-to-Pixel baseline]:** Thanks for your suggestion. we add an experiment to test the results of pixel-to-pixel baseline on three different downstream. The model was pre-trained for 100 epochs using VOC and ADE20K dataset and the results are reported in the table below. Our MDP achieves performance gain on both of three datasets. We have added this results in the revised version.

| Method         | Pretrained Dataset | VOC mIoU | ADE20K mIoU | 
| :------------- | :----------------: | :------: | :---------: | 
| Pixel-to-Pixel |   VOC and ADE20K   |  71.19   |    38.81    | 
| MDP            |   VOC and ADE20K   |  73.32   |    40.15    | 

**Q3: [How does a simple merging strategy address such a complex problem when dealing with multiple datasets]:** 

同review3 Q4

**Q4: [In region-level mixing, how are binary masks constructed]：**The mask $M$ can be represented by bounding box coordinates $\mathbf{B}=\left(r_{x}, r_{y}, r_{w}, r_{h}\right)$.  Among them, $r_{x}$ and $r_{y}$ are uniformly sampled within the width scope and height scope of the image respectively to determine the start point of the bounding box:

$r_{x} \sim$ Unif $(0, W)$, $r_{y} \sim$ Unif $(0, H)$

$r_{w}$ and $r_{h}$ determine the end point of the bounding box, and it is set according to a combination ratio $lambda$ to make the cropped area ratio $\frac{r_{w} r_{h}}{W H}=1-\lambda$:

$r_{w}=W \sqrt{1-\lambda}$, $r_{h}=H \sqrt{1-\lambda}$

（涉及公式回头得看下怎么处理）

### **To Reviewer  9Qx8 :**

**Q1: [Why the pixel-to-pixel loss in Equation (2) cannot simply be formulated over multiple images as wel]:** Calculating pixel-to-pixel loss across images needs to store all the pixels of all the images in the memory bank, which is hard to tolerate. An alternative is to only perform pixel-to-pixel contrastive learning on the images in a batch. However, the comparison of all pixels requires large GPU memory,  so it cannot be performed under a large batch size. If we set a small batch size, the images in the batch may come from totally different datasets and a large number of inappropriate negative samples will be introduced, which makes the training collapse in our experiment. 

**Q2: [How does the merge of the label spaces work]:** 

同review3 Q4

**Q3: [Wording and expression need to be revised]：**  Thanks for your detailed comments, we have carefully proofread the paper in the revised version. Some more appropriate expressions have also been updated.

额外要写的：

“I would argue that this is not a "unified" model because the categories are not unified as in other papers like [30][33]”

Line 160: What is "dirty data"? 

Line 169: What is "interactive" about this?

**Q4: [Training a network on all four datasets with dataset-specific classification heads as a pre-training step]：** Thanks for your suggestion!  We think this is a result worth exploring and have added related experiments. Based on VOC and ADE20K dataset, we use the same learning rate and learning rate decay settings as MMsegmention, and unify the training period to 100 epochs for easy comparison. It can be seen from the table below that the performance of this scheme is far lower than our method (71.19% vs 73.32%). We think this is because this scheme does not model the correlation between the various datasets, and each head is independent of each other.  Our MDP can perform joint learning across datasets, which makes our features more discriminative. We have added this results in the revised version.

| Method     | Pretrained Dataset | VOC mIoU |
| :--------- | :----------------: | :-------: |
| Multi-head |   VOC and ADE20K   |   71.19   |
| MDP        |   VOC and ADE20K   |   73.32   |

**Q5: [Evaluation on COCO-stuff-val? ]：** Thanks for pointing out that. We did not report the test results on COCO because MMsegmentaion did not provide COCO evaluation code and we worry that the results of our report are not representative. In fact, we also have our own implemented finetune results of 80k iterations on the COCO dataset. As shown in the table below, our method also boosts supervised ImageNet pretraining results on COCO for a large margin, from 35.25% to 38.37%.

| Method     |  Pretrained Dataset  | COCO mIoU |
| :--------- | :------------------: | :-------: |
| Scratch    |          -           |   25.02   |
| Supervised |       ImageNet       |   35.25   |
| MDP        | VOC, ADE20K and COCO |   38.37   |

**Q6: [Segmentation annotations are much richer than ImageNet image-level annotations]：** 

同review2 Q3

We mentioned in Ln261 that our method is more efficient because the number of images we use is much smaller than that of ImageNet pretraining. Our code has not been fully optimized, but the current running time of a batch is basically the same as moco v2. Taking into account the reduction in the number of images, our method is more efficient. We will provide the detailed numbers in the revised version.

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
