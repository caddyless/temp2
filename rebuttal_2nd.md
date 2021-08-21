### **Response to Reviewer qZDT:**

We appreciate that the reviewer is positive about our MDP setting, and for the main concern about **the effects of using ImageNet pretraining**, we add some comparative experiments and detailed responses are listed below. 

**Q1: only pre-training from scratch using segmentation datasets" may not be very useful as ImageNet pre-trained weights are always available.**

Thanks for your suggestion.  we clarify that: 

1. In MDP, we simply make use of training from scratch setting to **decouple the effect of ImageNet pretraining**. Interestingly, we find that for datasets such as ADE20K that have a larger domain gap with ImageNet (comparing with VOC), simply using VOC+ADE20K (35k images in total) can obtain better results than ImageNet pretraining (last column). While for VOC, the performance of using VOC+ADE20K (74.30%) is below ImageNet pretraining (75.63%), while it can be boosted with more pretrained segmentation datasets(adding COCO, 77.79%).

2. For performance, MDP can be easily combined with ImageNet pretrained weights to further improve the results (last two rows), we would add the results in the revised version.

3. Overall, we think it is an interesting topic to explore the necessity of ImageNet pretraining for segmentation. As we claim, there exists a task gap when using a classification-based pretrained model for segmentation tasks, while **the performance of the downstream tasks relies on multiple factors, such as the domain gap and the scale of the downstream dataset**, it remains to be a future research topic. 

   | Method     |              Pretrained Dataset              | Epoch | VOC mIoU |        ADE20K mIoU        |
   | :--------- | :------------------------------------------: | :---: | :------: | :-----------------------: |
   | Supervised |                   ImageNet                   |   -   |  75.63   |         **39.36**         |
   | MDP        |                  VOC+ADE20K                  |  100  |  73.32   | 40.17 **(0.81$\uparrow$)** |
   | MDP        |                  VOC+ADE20K                  |  200  |  74.30   | 40.83 **(1.47$\uparrow$)** |
   | MDP        |               VOC+ADE20K+COCO                |  200  |  77.79   | 42.69 **(3.33$\uparrow$)** |
   | MDP        | ImageNet(first stage), VOC+ADE20K(2nd stage) |  100  |  75.70   | 41.57 **(2.21$\uparrow$)** |
   | MDP        | ImageNet(first stage), VOC+ADE20K(2nd stage) |  200  |  76.24   | 41.93 **(2.57$\uparrow$)** |

**Q2: it might be better to compare with the model for standard trained weights on segmentation datasets, initialized from ImageNet. Currently, it looks like the performance gain comes from ImageNet pre-training but does not show the advantage of MDP**  

Thanks for your comments. As shown in the Table for Q1, MDP can further improve the results even with ImageNet pretrained weights (75.63->76.24 for VOC, 39.36->41.93 for ADE20K, 200 epochs). The performance gain of ADE20K is larger than VOC, possibly due to the larger domain gap of ImageNet and ADE20K. We would add these results in the revised version.  

To further demonstrate the effectiveness of MDP, we add another setting, i.e.,  training on the whole COCO dataset based on ImageNet pretrained model  (without label preprocessing explained in Q5 for first rebuttal, third row in the table below). It can be seen that the use of the COCO dataset will damage the performance (75.63%->74.51%), possibly due to complex label mapping in COCO, while MDP achieves better performance even without using ImageNet (76.85%), and the performance can be further boosted using ImageNet pretrained weights (77.56%). These results demonstrate the effectiveness of MDP. 

| Method                                     | Pretrained Dataset |          VOC mIoU           |
| ------------------------------------------ | ------------------ | :-------------------------: |
| Supervised (customized cross entropy loss) | COCO               |            68.48            |
| Supervised                                 | ImageNet           |          **75.63**          |
| Supervised (customized cross entropy loss) | ImageNet -> COCO   | 74.51 **(1.12$\downarrow$)** |
| MDP                                        | COCO               |  76.85 **(1.22$\uparrow$)**  |
| MDP                                        | ImageNet-> COCO    |  77.56 **(1.93$\uparrow$)**  |

**Q3: [Report full results for ablation study]:**  

Thanks for point out this.  We follow your suggestion and pre-train the model for 200 epochs to verify the performance. The results below are for the hyper-parameters setting.  We can see that although the performance difference between settings become smaller, the previous conclusions in our paper still hold, that is:

1. The proposed algorithm is relatively robust to memory bank size.
2. Smaller temperature brings better performance under the supervised pixel level contrastive learning setting.
3. Class-prototype can obtain more representative embedding and boost performance.

We further analyze our results and conclude that the reduction in the performance gap mainly comes from the long finetune stage. To validate this, we report the results at the initial stages of finetuning (4k iterations) and find that the performance gap is more obvious, which also reflects that our method has learned more discriminative representation comparing with pixel-to-pixel and region type memory bank type.

| Method         | Pretrained Dataset | Memory bank Type | Memory bank Size | Temperature | VOC mIoU 4k | VOC mIoU 40k |
| :------------- | :----------------: | :--------------: | :--------------: | :---------: | :---------: | :----------: |
| pixel-to-pixel |        VOC         |        -         |        -         |    0.07     |    46.70    |    70.03     |
| MDP            |        VOC         |      Region      |       4096       |    0.07     |    58.94    |    70.11     |
| MDP            |        VOC         |      Class       |       4096       |    0.07     |    60.02    |    70.56     |
| MDP            |        VOC         |      Class       |       1024       |    0.07     |    59.72    |    70.48     |
| MDP            |        VOC         |      Class       |      10560       |    0.07     |    61.61    |    70.91     |
| MDP            |        VOC         |      Class       |       4096       |     0.3     |    56.71    |    70.09     |
| MDP            |        VOC         |      Class       |       4096       |     0.5     |    51.95    |    69.89     |

The table below shows the effectiveness of cross-image mixing. It reflects that the mixing is still effective for longer pretraining epochs.

| Method | Pretrained Dataset | Cross-image Mixing | Epoch | VOC mIoU |
| :----- | :----------------: | :----------------: | :---: | :------: |
| MDP    |     VOC+ADE20K     |         No         |  100  |  71.98   |
| MDP    |     VOC+ADE20K     |         No         |  200  |  73.49   |
| MDP    |     VOC+ADE20K     |        Yes         |  100  |  73.32   |
| MDP    |     VOC+ADE20K     |        Yes         |  200  |  74.30   |

We would update the results in the revised version.
