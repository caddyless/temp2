【这里我们需要陈述整篇文章的贡献点，增加对R3的质疑的一个简要总结，主要包括nolvety 以及实验性能的问题】

We thank all reviewers for their valuable comments.  This paper proposes a unified framework that..., and all reviewers are positive and interested in this novel settings, as far as we know, we are the first to rethinking pretraining for semantic segmentation. In the following, we answer the concerns of each reviewer one by one. 

We aruge that there exists task gap when using classification based pretrained model like ImageNet for segmentation task, and hope to make good advantages of available annotations for segmentation pretraining. 

### **To Reviewer wmH9:**[这个是一对一回答，直接贴在每一个问题下面]

**Q1: [Discussion about multi-dataset training]:** Our method cannot directly compare the performance with the multi-dataset training (type 2 in Fig. 1) method, because this type of method is not designed for pre-training, but instead trains a unified downstream model by manually integrating the label space. This type of method usually has the following characteristics: 1) Massive manual operations are required for label unification. 2)The unified model can achieve good results in each downstream, but they are all worse than the results of training each downstream separately.  We believe that this type of method does not make full use of the advantages of multi-datasets, and their process is too cumbersome. 

MDP is different from type 2 method (multi-dataset training with unified label space) in both motivation and application scenarios. 1）type 2 is designed to train a unified network that is able to segment all objects from multi-dataset, while MDP is designed for pretraining, which acts as ImageNet pretrained model that is usually used in current computor vision tasks. 2) In order to facilitate training, type 2 method needs manually integrating the label space from multi-dataset, which is labor-consuming and inevitable to introduce noise.  3) as demonstrated in mseg(\cite), tranining a unified segmentation network usually obtains lower performance comparing with training for each dataset seprately, mainly because the optimization competetion and lable inconsistency from diverse dataset. It is hard to optimze the objective when the number of classes grows, and inevitabely suffers from long tail issue, while MDP maintains one prototype for each class regardless of the number of images. This ensures the contribution of small datasets and classes with rare data be not ignored. Experiments demonstate the effectiveness of MDP. 

To further answer your question,  we have designed a new pre-training scheme based on multi-dataset training, that is, co-training a shared model with dataset-specific classification heads and then finetuning at different downstream. This scheme is a lit bit like [33] but without ImageNet pre-training and subsequent label unification for fair comparison. The results are shown in the table below.  It can be seen that if manual label integration is not performed, the performance of this scheme is far lower than our method (71.19% vs 73.32%) because this scheme cannot model the relationship between multi-datasets.  The above discussion proves the effectiveness of our MDP pipeline towards multi-dataset training.（这一段要加吗？会不会有点奇怪？但没了感觉太单薄了...（浩航师兄建议直接删了））-这里是multi-head的预训练，其实和作者提问的不是一个问题，建议删掉

| Method     | Pretrained Dataset | VOC mIoU |
| :--------- | :----------------: | :------: |
| Multi-head |     VOC+ADE20K     |  71.19   |
| MDP        |     VOC+ADE20K     |  73.32   |

[33] Xingyi Zhou, Vladlen Koltun, and Philipp Krähenbühl. Simple multi-dataset detection. arXiv preprint arXiv:2102.13086, 2021.

[mseg]

**Q2: [Adding this pre-training strategy on a well-pretraind backbone]:**  Good question, using pretrained backbone can de

facto improve the performance, especially when the number of images used for segmentation pretraining is relative small (e.g., VOC+ADE20K, around 35k images). As shown in the table below, when using imagenet pretrained model for initialization, the performance of the pretraining strategy ( VOC+ADE20K, 100 epochs)  can be boosted from 73.32% to 75.7% when fine-tuned on VOC. In this paper, we target at uncovering the effect of task related pretraining and simply decouple the influence of pretrained model for fair comparisons. On the other hand, we think the advantage of ImageNet pretraining is negligible when the number of annotations used for pretraining is relatively large like using COCO.  

| Method |                Pretrained Dataset                | VOC mIoU |
| :----- | :----------------------------------------------: | :------: |
| MDP    |                    VOC+ADE20K                    |  73.32   |
| MDP    | ImageNet(first stage), VOC and ADE20K(2nd stage) |  75.70   |

### **To Reviewer Hbtb:**

**Q1: [The techniques for the cross dataset learning seem too heuristic. It may not generalize well to other settings]:** Thanks for your questions. We believe that the generation ability needs to be analyzed from two aspects, that is,  the generalization of the method and the generalization of the model. We admit that our method is mainly for semantic segmentation tasks and our method may not work if the dataset gap for pre-training is too large (e.g. simultaneously using some medical segmentation datasets and nature segmentation datasets for pretraining).  However, the core idea of our method can be applied in many domains, such as the class prototype and cross-dataset mixing. We can also pre-define class prototypes for tasks such as classification and body parsing, and reduce the domain gap by processing across datasets. As to the model generalization, our model can also achieve great performance in other downstream, and our model can be further adapted to scenarios such as data mining. From this perspective, we think our model still has great generalization ability. As an example, we test the transfer learning accuracy on COCO detection and COCO instance segmentation using Mask R-CNN and the results are shown in the table below. Although our pre-training model does not completely match the downstream model and can only load part of the parameters, our method still surpasses the supervised and self-supervised ImageNet pretraining method. -[你这个说的太绕了，我都不知道你在说啥]

| Method     |  Pretrained Dataset  | Box AP for object detection | Mask AP for instance segmentation |
| :--------- | :------------------: | :-------------------------: | :-------------------------------: |
| Supervised |       ImageNet       |            38.9             |               35.4                |
| MoCo v2    |       ImageNet       |            39.3             |               35.7                |
| MDP        | VOC, ADE20K and COCO |            39.8             |               36.0                |

答：

就是他其实在问的泛化能力只是方法的泛化能力，我补充写了下模型的泛化能力，如果觉得没必要的话，可以只有方法泛化能力，核心大概是：

Thanks for your questions.  We admit that our method is mainly for semantic segmentation tasks and our method may not work if the dataset gap for pre-training is too large (e.g. simultaneously using some medical segmentation datasets and nature segmentation datasets for pretraining).  However, the core idea of our method can be applied in many domains, such as the class prototype and cross-dataset mixing. We can also pre-define class prototypes for tasks such as classification and body parsing, and reduce the domain gap by processing across datasets.  and our method can be further adapted to scenarios such as data mining by utilizing pre-define class prototypes .

**Q2: [Using only COCO in the pretraining]:** Thanks for your suggestions! We compare the results of three different settings over 100 epochs pre-training in the following table: 1) MDP pretraining using only VOC dataset(~10k images), 2) MDP pretraining using VOC and ADE20K dataset (~30k images) 3) Supervised pretraining on COCO dataset(~100k images). 4) MDP pretraining using only COCO dataset(~100k images).  The table shows that:

1. The COCO pre-training results do achieve high performance. However, compared with the supervised results, we think this is mainly due to our pre-training method, which enables the advantage of larger data scale (three times larger than the combination of other two datasets) to be reflected. 
2. We want to emphasize the advantages of our method when the amount of data is comparable. Compared with the first setting using only VOC dataset, our 'VOC+ADE20K' configuration has achieved a huge performance improvement. At the same time, this configuration has surpassed the results of supervised ImageNet pretraining on the ADE20k and Cityscapes downstream, shown in Tab. 1 of our paper【直接在这里面引用，文中类似引用原文的都一样，不要让reviewer再去看】. We think these results can prove the benefits of MDP.【你也没有测试，其实最好测一下几个数据集合的下游性能，还有所有三个数据集合合在一起的性能】

| Method     | Pretrained Dataset | VOC mIoU |
| :--------- | :----------------: | :------: |
| MDP        |        VOC         |  70.92   |
| MDP        |   VOC and ADE20K   |  73.32   |
| Supervised |        COCO        |  68.48   |
| MDP        |        COCO        |  76.85   |

**Q3: [Considering the number of mask label instances, the COCO dataset may surpass Imagenet in terms of scale.]** Yes, we admit that the data scale can not be comparable to the scale of data annotation and maybe we should use a more appropriate expression here. However, it is precise because of the difficulty of labeling that a single large-scale pixel-level labeled dataset is difficult to obtain. Our method provides a co-pretraining strategy using data from different sources and from the perspective of data amount, our performance is very outstanding.  In addition, we can also utilize the class prototype from a small amount of pixel-level labeled data to perform data mining using unlabeled images.  The results below show toy experiment results using no-label COCO data. The unlabeled images are feed into the momentum encoder to obtain pseudo-label according to its similarity with class prototypes and the rest of the process is consistent with MDP. The results review that using a smaller amount of unlabeled images (compared to the scale of ImageNet) can further improve the performance.-【这个其实很好回答，我来写吧】

【这里放数据挖掘其实没啥用，建议删掉】

Thanks for pointing out this. Considering the annotation cost, we agree that it is unfair to simply compare the number of images that used for ImageNet pretraining and our MDP due to different annotation granularities. The highlight is that it is more suitable to design task specific pretrained model for segmentation, and the number of images needed for pretraining can be substaintially decreased, as shown in Table 1 in the original paper, using VOC+ADE20K(around 35k images) for pretraining, MDP obtains better performance than ImanegNet pretrained model with class class labels (VOC: ADE20K: results).  We have adjusted the expression in the revised version accordingly.

| Method |       Pretrained Dataset       | VOC mIoU |
| :----- | :----------------------------: | :------: |
| MDP    |         VOC and ADE20K         |  73.32   |
| MDP    | VOC, ADE20K and COCO(no label) |  74.74   |

**Q4: [The sparse coding part needs more analysis and justification]** （这一块基本就是自己分析了。。。）Since taxonomy from different datasets differs, the sparse coding part is designed by utilizing inter-class similarity across different datasets to increase the discrimination of embeddings with the same label and to make the re-mapping easier in the downstream fine-tuning.  In short, we hope to use sparse coding to make features of the same class distinguishable intraclass. The weight of the loss of sparse coding is low so that the constraint only makes the feature diverse and does not occupy the dominant position.

**We claim that the sparse coding is especially beneficial for downstream tasks that have non-overlap labels as the pretraining stage, and this is a common case in real application, and this can be demonstrated when using VOC and ADE20K for pretraining, while cityscapes for downstream task**. 

During the experiment, we adopted two different settings：CSC1）Not pushing the pixel and its top-k similar prototype far away;  CSCS2) pulling the pixel and its top-k similar prototype close (as mentioned in the paper). The results are shown in the table below. Setting 1 does not directly push the class of VOC away from other similarity class of VOC or other datasets, which makes its results on VOC worse. However, the results on Cityscapes are still comparable. This indicates that even if the features between the categories are not completely pushed away, making the features within the categories distinguishable can still maintain a good accuracy rate in the unseen downstream. Setting 2 increases intra-classes discrimination while ensuring inter-class discrimination. This makes the performance of VOC still comparable, and the performance of Cityscapes has been further improved. During our experiment, the loss in the finetune stage of Setting 2 has also been lower during the finetune process, which indicates that our sparse coding scheme is effective.【废话太多了，加你直接分条陈述，1）2）3）...， 还有你表格里面CSC1,2 啥意思】

答：cscs1和csc2对应上面的两种方案，因为直接在coco下游测结果体现不出来，所以就不得已拿了两种方法对比解释一下，这里感觉还不好分条

| Method        | Pretrained Dataset | VOC mIoU | Cityscapes mIoU |
| :------------ | :----------------: | :------: | :-------------: |
| MDP           |   VOC and ADE20K   |  71.98   |      75.37      |
| MDP with CSC1 |   VOC and ADE20K   |  70.88   |      75.50      |
| MDP with CSC2 |   VOC and ADE20K   |  71.84   |      76.53      |



### **To Reviewer qZDT:**

**Q1: [Technical novelty and contribution]:** We think the main innovation of our work lies in the proposed Multi-dataset pretraining framework. Most of the previous work was based on ImageNet pre-training model and did not consider the difference between the pretraining task and the downstream task at all. Our work is the first to focus on this problem. It can integrate and utilize multiple datasets with completely different label spaces to learn discriminable features and has versatility and wide application value. 

In addition, our methods have been improved and innovatively applied to the problems of our framework. We believe that this part of the contribution cannot be ignored.  Among these methods, the pixel-to-prototype contrastive learning we proposed is mainly to alleviate the huge storage space overhead in the case of multiple datasets.  On the one hand,  continuing to use pixel-to-pixel or pixel-to-region contrastive learning will cause huge storage overhead, which is unbearable in a multi-datasets scenario because a sufficient number of features need to be stored to reflect the overall category distribution of the multi-datasets. On the other, due to the limitation of GPU memory when calculating similarity,  pixel-to-pixel or region-to-pixel contrastive loss cannot be calculated efficiently under the multi-datasets setting. Our pixel-to-prototype method also achieved better performance in experiments. We think this is because compared to pixel embeddings, our class prototype is more stable and less susceptible to mislabeling.  In addition, muti-dataset pretraining introduces a new long-tail problem due to the difference in the number of samples between datasets. Our method can also alleviate this problem since only one prototype for each class is stored regardless of the number of images containing the class. We will further add more experiments about these benefits.

Similarly, our region-level mixing and pixel-level mixing are also innovative applied in new scenarios, which can alleviate the domain gap between datasets at both image and class levels. Different from [c], we have introduced different level of mixing operations on images to further boost diversity.  We also consider the gap of multiple datasets at the class level and generate the "intermediate class" of the two classes from different datasets through pixel-level class mixing. This is also a problem that is not involved in other work, and we can alleviate it through uncomplicated methods. 【这一部分一定要谨慎，问题， 分条陈述， 不要引用原文，直接把观点列上，后续有必要要放在整个rebuttal的开头】

**Q2: [Cross-class sparse coding does not seem reasonable]:** 

同review2 Q4

**Q3: [The dataset comparison is not fair as the paper uses pixel-level annotations]** 

同review2 Q3

As to the problem of including the dataset of the final testing data，we also evaluate the performance on Cityscapes in the paper , where the model does not see any images over this dataset during pretraining, and the results are surprisingly promising.  We have also conducted performance evaluations of different finetune iterations and have verified that our model has indeed learned more discriminative features.  We will  add more unseen downstream in the future.

**Q4: [The authors mention that they do not unify the label space, but it is not reflected]:** The label unification in our paper means mapping the labels between different datasets to a unified label space to tackle the label inconsistency among datasets, which requires a lot of manual definition and alignment. However, just as indicated in Ln 134, our method just concatenates the label space of different data sets without any manual integration and alignment. This allows our method to better deal with multi-datasets problems. In fact, we also tried the label rough unification on the basis of MDP at the early stages (only the data preprocessing is slightly different from now) but we did not see any improvement. We analyze that it is the feature-remapping in the fine-tuning stage that automatically realizes the label re-unification.

| Method | Pretrained Dataset | Label Unification | VOC mIoU |
| :----- | :----------------: | :---------------: | :------: |
| MDP    |   VOC and ADE20K   |        No         |  71.93   |
| MDP    |   VOC and ADE20K   |        Yes        |  71.91   |

**Q5: [The improvement is marginal considering the improvement of simply using COCO pre-trained weight]:** 

The reviewer may misunderstand our settings. We think your description is inappropriate and the comparison is unfair. The semantic segmentation pipelines like Deeplab [] can indeed achieve around 2% improvement by using the COCO pre-trained weight, but:

1. The COCO pretraining is based on **a well ImageNet-pretrained model **while our MDP is trained from scratch.  Their improvement is based on the application of ImageNet while our pipeline is not. 
2. The COCO data for pretraining is after careful selection and processing. Only the images containing the classes defined in VOC are selected and the COCO classes not defined in VOC are all treated as background classes.  In contrast, our method does not perform any selection and label processing operation on COCO data.

We also further explored the results using some similar settings: 1) Training on the whole COCO dataset without label processing operation from scratch. 2) Training on the whole COCO dataset without label processing operation based on ImageNet pretraining.  The COCO training lasts 80k iteration and the finetune is conducted on VOC for 40k iteration. Results are shown in the table below. The performance is far from satisfactory compared to our method, and we also notice that: 1) ImageNet pretraining is necessary. Training COCO from scratch and then finetuning on VOC obtain worse performance. 2) Data selection and label processing are important. We notice that the loss of our exploration setting (without label processing) is relatively low at the beginning of the finetune stage, but then it has been fluctuating at a higher value for a long time. We think this is due to the label space gap between COCO and VOC. Compared to the above setting, our method aims at making the features distinguishable, which results in much better performance.

| Method                      | Pretrained Dataset | VOC mIOU |
| :-------------------------- | :----------------: | :------: |
| Supervised training on COCO |         -          |  68.48   |
| Supervised training on COCO |      ImageNet      |  74.51   |

**Q6: [The ablation study in Table 2  and Table 3  does not seem complete and are all worse than ImageNet pretraining]:** Table 2 studies on hyper-parameters and the type of memory bank using only VOC dataset so the results are worse tham ImageNet pretraining. We think that the setting of hyper-parameters can be explored in a single dataset for efficiency since directly adding all segmentation datasets for ablation study is too expensive.  Table 3 studies the cross-dataset mixing strategies. We have reported the results of using VOC and ADE20K in the paper and we believe that the huge improvement in the results can prove the effectiveness of our strategies. From another perspective, the results gap between using a single dataset like VOC and using ImageNet also proves the effectiveness of our MDP.-【这个和Q5不是一个问题吗，可以合并？】

**Q7: [Some terms are quite vague and cannot provide the audience a good motivation and a big picture of the proposed method.]**  Thanks for your detailed comments, we have carefully proofread the paper in the revised version. Some more appropriate expressions have also been updated.



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
