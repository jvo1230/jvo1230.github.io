---
name: Multimodal Attention-based CNNs for the Prediction of Alzheimer's Disease
tools: [Pytorch, Python, Medical Image Processing, Computer Vision]
image: https://3acf3052-cdn.agilitycms.cloud/images/service/BRAIN.jpg
description: This project investigates the use of attention-based mechanisms in the fusion of MRI and PET images to predict the progression of Alzheimer's disease.
---

# Multimodal Attention-based CNNs for the Prediction of Alzheimer's Disease
<br>
<i><b>Publication: To be written</b></i>
<br><i>[Github](https://github.com/JamieVo890/Multimodal-Attention-based-Neural-Networks-for-the-Prediction-of-Cognitive-Decline)</i>


The brain is amazingly complex. It's made up of more than 86 billion nerve cells and it's what makes us, well.. us! But when problems occur in the brain and how it functions, debilitating conditions which can take away our ability to do simple things arise. One of these conditions is known as Alzheimer's Disease.

![Firstimage](https://media.istockphoto.com/id/1358833655/vector/vector-illustration-of-confused-man-with-mess-in-his-head.jpg?s=612x612&w=0&k=20&c=8sJzusexsxa5wKxwezZgOS7HQA7PJ6HOk9T5CqbjjgE=)

Alzheimer’s disease (AD) is a neurodegenerative condition characterised by memory
impairment and cognitive decline. It is one of the most prevalent neurodegenerative diseases,
typically affecting people over the age of 65. Unfortunately, there's 
no current cure for AD, leaving treatment processes to revolve around delaying the onset of symptoms. So, it's pretty important to detect the presence of the AD as soon possible to ensure early treatment. But this is hard. The underlying factors which cause the disease are complex and still not very well understood. So this brings up the question, is there a way we can effectively detect the presence of AD in individuals when we don't fully know the defining patterns or features relating to brains with AD? The answer:

<b>Convolutional Neural Networks!</b>

In this project, we present a <b>M</b>ultimodal <b>N</b>euroimaging <b>A</b>ttention-based convolutional neural network (CNN), <i>MNA-net</i>. Instead of simply classifying whether someone has AD, we focus on predicting the progression of the disease, that is, predict whether a cognitively normal individual will develop AD or some form on mild cognitive impairment (MCI) in the future. To learn the complex features relating to MCI and AD, MNA-net combines both Magnetic Resonance Imaging (MRI) and Positron Emission Tomography (PET) using attention-based mechanisms.

<br>
## Dataset

For this project, we use the OASIS-3 dataset obtained from the Open Access Series of Imaging Studies (OASIS). OASIS was launched in 2007 with the primary goal of making neuroimaging data publicly available for study and analysis. OASIS-3 is a longitudinal dataset released as a part of OASIS in 2018. It is a compilation of clinical data and MRI and PET images of multiple subjects at various stages of cognitive decline collected over the course of 30 years. Subject cognitive states in OASIS-3 are defined by clinical dementia rating (CDR) scores. A total of 1378 participants entered the study, 755 of which were cognitively normal (CDR = 0), and 622 who were at progressing stages of cognitive decline (CDR ≥ 0.5). For our study, we will utilise the MRI and PIB PET images provided in OASIS-3.

<br>

#### Subject Selection

To predict the progression of cognitive impairment in individuals within OASIS-3,
we focus on two groups of subjects: subjects who remained cognitively normal (CN), and subjects transitioned from CN to MCI or AD over the course of the study in OASIS-3. For
this scope of this work, we consider a timeframe 10 years. One important point to keep in mind
is the temporal alignment of data. It's important that subject scans are taken within close proximity of their initial diagnosis so that scans are representative of their cognition at the time of their baseline. Taking these factors into consideration, our subject selection criteria for the OASIS-3 dataset are as follows:
1. Subjects were diagnosed as CN at baseline.
2. Subjects have taken MRI and PET scans that are within a year from their
baseline diagnosis.
3. Of CN subjects who developed cognitive impairment over the course of the
study, only those who were diagnosed with MCI or AD within 10 years of
their baseline diagnosis were considered.
4. Of subjects who remained CN over the course of the study, only those who
received a diagnosis of CN at least 10 years after their baseline diagnosis were
considered.

<br>

#### Image Data

Post processed Freesurfer files for the MRI images are provided by OASIS-3. These
files contain the subject-specific 3D MRI images which have undergone skull stripping. The PIB PET images, however, are provided as 4D Nifti files. These images
are acquired in multiple frames over different time intervals and as such,
we apply temporal averaging of the 4D PET images to average the frames into
static 3D images. Noise and skull is then need to be removed from the PET images using Brain
Extration Tool (BET) and Synthstrip. Figure 3 presents an example of
PET noise and skull removal. 

![skullstrip](https://github.com/JamieVo890/Multimodal-Attention-based-Neural-Networks-for-the-Prediction-of-Cognitive-Decline/assets/70950884/4bf65e67-9768-4a9b-9aa5-35931e17106d)
<i> ‎‎‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎Skull and noise removal of PET image</i>

Finally, both MRI and PET images are standardised and aligned to a common anatomical template by normalising voxel intensities and registering them to Montreal Neurological Institute (MNI) space using FMRIB’s Linear Image Registration Tool (FLIRT) 

Data augmentation is performed on the training set to increase the dataset size. To
simulate different positions and size of the patient within the scanner, and anatomical variations present in the images, random affine transformations and elastic deformations were applied to the images. The figure below examples of elastic deformations and affine transforms applied to an MRI image.

![aug](https://github.com/JamieVo890/Multimodal-Attention-based-Neural-Networks-for-the-Prediction-of-Cognitive-Decline/assets/70950884/2b733000-e4a6-4e97-825e-abe46670d127)
<i>‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏ ‎‏‏‎ ‎‏ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎‏‏‎ ‎ ‎‏‏‎‏‏‎ ‎‏‏‎‏‏‎ ‎‏‏‎‎‏‏‎ ‎‎‏‏‎ ‎‏ ‎‏‏ ‎‏‏From left to right: Control MRI, Elastic Deformation, Affine Transformation</i>
<br>

## MNA-net
To harness the strengths of both MRI and PET in CN to MCI and AD classification,
we propose MNA-net, a multimodal neuroimaging attention-based CNN. We define
three stages in the classification process in MNA-net as shown in the figure below: patch
feature extraction, multimodal attention, and patch fusion.

![MNANET](https://user-images.githubusercontent.com/70950884/277098743-8658a856-293d-479b-b859-1a847c5c58fe.png)

In the first stage, we adopt a patch-based technique. MRI and PET images are
both divided into 27 uniform patches of size 44 x 54 x 44 with 50% overlap. Each
patch is then fed into a 3D ResNet-10 model to extract the local features of each
image. In the second stage of the classification process, we introduce an attentionbased ensemble architecture to facilitate the fusion of the different neuroimaging
modalities. For every patch in corresponding positions between the MRI and PET
patches, we extract the learnt features from the ResNet-10 models and pass them
through an attention-based model. This model utilises self-attention mechanisms to
enable the model to create shared representations of the MRI and PET features. In
the final stage, we consolidate the features extracted from the patch-level models.
The attention weighted multimodal features for each patch are extracted from the
attention models and flattened, concatenated, and passed through a dense with
sigmoid activation for the final classification.
Due to the complexity and wideness of the architecture, training MNA-net as a
single model is computationally intensive. Instead, we train the individual models
for each classification stage separately. Features are extracted from each model and
used as inputs for the subsequent classification stage.

<br>

#### Patch-based Feature Extraction
To extract the patch-based features, we adopt a 3D ResNet architecture as the
backbone model. First proposed by Kaiming et al [37] in 2015, ResNet is family of
CNN architectures which introduce the concept residual connections. ResNet aims
to overcome the issue of exploding and vanishing gradients seen in deep networks.

The major limitation of many CNN architectures applied in the prediction of
cognitive decline is use of 2D kernels. To accommodate PET and MRI scans within
the framework of 2D CNNs, the 3D brain images are often divided into multiple
2D slices. However, this results in a loss of spatial information. To this end, we
instead utilise a 3D ResNet architecture using 3D convolutions adapted from Hara
et al [38]. A brief illustration of the model is shown in Figure 6. 

![patch](https://user-images.githubusercontent.com/70950884/277109747-de9d45c0-1998-4e86-9f44-b512247b95da.png)

The patch images of size 44x54x44 are first passed through a 7x7x7 convolutional layer with stride 2
and padding 3, followed by max pooling, batch normalisation, and a ReLu. We then
introduce the residual connections through four sequential conv blocks. Each conv
block consists of two 3x3x3 convolutional layers, each followed by batch normalisation and Relu. A residual connection is included between the beginning of the block
and the layer preceding the final ReLu. Strides of 2 are used in the convolutional
layers of conv block 2, conv block 3, and conv block 4 to perform down sampling.
The output feature maps of conv block 4 are then finally subjected to an average
pooling layer, flattened, and subsequently passed through a fully connected layer
for final classification. The features prior to the final dense and sigmoid layers are
extracted and used as inputs for the multimodal attention classification stage.

<br>

#### Attention-based Multimodal Feature Fusion
To combine the learned patch features of MRI and PET, we introduce the concept of
self-attention into our fusion pipeline. Figure 7 shows the architecture of the attention model trained to fuse the patch features. Multiple approaches in the fusion of
PET and MRI for MCI and AD classification seen in literature have simply involved
the concatenation learned features. This, however, is flawed due to the lack of cross
modal interactions. Representations of MRI and PET features which take into account information from each other may be more informative than considering each
feature independently. Attention mechanisms aim to mimic the cognitive process of
attention, enabling neural networks to create shared representations which consider
all parts of the input data based on attention scores.

![attention](https://user-images.githubusercontent.com/70950884/277109752-0e5a9a65-8c52-4129-8086-e601ed9a533a.png)

For every patch in corresponding positions between the MRI and PET patches,
we extract and vertically stack the features prior to the last layer from the previously
trained patch-based feature extraction models. We then pass the stacked features
through a multi-head attention layer with 4 attention heads. Finally, the vertically
stacked attention weighted outputs for the PET and MRI features are flattened and
passed through a fully connected layer for final classification. The final flattened
features are then used as inputs to the final model shown in Figure 5.

<br>

## Results
