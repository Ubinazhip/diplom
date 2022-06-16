# Generation of consistent segmentation for MLO and CC projections of the breast

# Goal
Solve segmentation task and improve consistency between predicted masks of MLO and CC views of breast by **modifying the loss** function and by using **transformer blocks as an encoder**. </br>
The baseline models are our segmentation models that had been trained without any transformers and without any auxiliary loss. 
# Datasets
Popular InBreast [1] and CBIS-DDSM [2] datasets
# Models
- Models - UNet, UNet++, Feature Pyramid Network(FPN), UNetr from [3]
- Backbones - Resnet34, Resnet50, Densenet121, Efficientnet-b3 
- Main_Loss = w1 * BCE + w2 * Focal + w3 * DICE (weighted sum of binary cross entropy loss, focal loss and dice loss)
# Evaluation
- Segmentation metric - **Dice score**
- Consistency metric - **MSE(vec(pred_mask_MLO), vec(pred_mask_CC))**; where vec(mask) - sum along y-axis, since MLO and CC has comman x-axis.
# Proposed methods
- Transformer as an encoder - send patches of MLO and CC to the transformer. Transformer will find the relation between the patches of MLO and CC.
- Modify Loss - Loss = main_loss + weight * **aux_loss**; **aux_loss** = MSE(vec(pred_MLO), vec(pred_CC))
# Reference
[1] I. C. Moreira, I. Amaral, I. Domingues, A. Cardoso, M. J. Cardoso, and J. S. Cardoso, “Inbreast: toward a full-field digital mammographic database,”
Academic radiology, vol. 19, no. 2, pp. 236–248, 2012 </br>
[2] R. S. Lee, F. Gimenez, A. Hoogi, K. K. Miyake, M. Gorovoy, and D. L. Rubin, “A curated mammography data set for use in computer-aided detection
and diagnosis research,” Scientific data, vol. 4, no. 1, pp. 1–9, 2017. </br>
[3] A. Hatamizadeh, Y. Tang, V. Nath, D. Yang, A. Myronenko, B. Landman, H. R. Roth, and D. Xu, “Unetr: Transformers for 3d medical image
segmentation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022, pp. 574–584. </br>
# Author 
Aslan Ubingazhibov - HSE Moscow - aubingazhibov@edu.hse.ru
