**Group name - Techno Turtles**

**Group Members** - Pavan Kumar Jonnadula (pj1098), Ramprasad Kokkula (rk1668), Aakanksha Padmanabhan(al7275), Vaishnavi Patil (vp2156)

**Project Title - Gesture Recognition and Background Manipulation in Live Camera.**

**Abstract / Overview**

This project delves into the realm of computer vision with a focus on image processing, feature extraction, and object recognition, centering on the understanding of gestures and background changes in images. Using existing libraries like the cvzone handTracking module, key gesture points will be extracted. The exploration extends to various machine learning frameworks for testing classifier models, with the selection of the most effective one. OpenCV is employed for image processing tasks, including background manipulation and segmentation, while PyTorch and existing CNN models are fine-tuned for segmentation. The project's overarching goal is to combine gesture recognition with background changes, offering users an interactive experience.

**Learning Objectives**

In our project, we want to explore the concepts of computer vision, focusing on image processing, feature extraction, and recognizing objects. Our main focus is on understanding gestures and changing backgrounds in images. We'll dive into Convolutional Neural Networks (CNNs), to become skilled at and using models for accurate gesture recognition.

Using existing libraries such as the cvzone handTracking module, we will extract key points of gestures. We will learn various machine learning frameworks and test different classifier models to compare and select the most effective one. We will use PyTorch and existing CNN models for segmentation and fine-tune them to meet our requirements.

**Discussion of ethics**

We'll make sure to consider ethical concerns at every stage of our project. The following key ethical considerations will guide our approach:

**1. Privacy Protection:**

Our dataset is collected from Kaggle website, where privacy protection is considered all ready.

**2. Bias Mitigation:**

This has been taken care of in the pre-processing stage, as we will be dealing with points and not the image.

**3. Misuse of the model**

There might be a misuse of the gesture recognition model to train certain gestures which are not appropriate.

**What Exists and how we'll use it**

**OpenCV:** We will use OpenCV for various image processing tasks, including background manipulation and segmentation. We are using existing libraries like cvzone handTracking module to extract the key points of the gesture.

**Machine Learning Frameworks:** We will be trying some existing classifier models to train and test, comparing the results of each classifier. Then we will choose the classifier with the best result.

**PyTorch:** We are using existing PyTorch CNN models to implement segmentation and fine-tune them to our specific requirements which will expedite our progress.

**Reach Goal**

This project aims to smoothly combine recognizing gestures with changing backgrounds, providing users with an interactive experience.

**Minimum Goal**

Our project currently has two minimal goals both of which can be achieved separately.

1. **Gesture Detection**

The system aims to recognize and interpret at least two distinct gestures (thumbs up and thumbs down) from users interacting in static images.

1. **Segmentation**

The system aims to segment people and apply backgrounds (solid colors and static images).

**Milestones and internal deadlines**

**Gestures:**

| **Milestone** | **Date** | **Topic Completed** |
| --- | --- | --- |
| 1 | Nov 21 | Download 2 classes(thumbs up and thumbs down) and pre-process the datasets. |
| 2 | Nov 28 | Build a simple model and test the model |
| 3 | Dec 5 | Try different models and do live testing |

**Segmentation:**

| **Milestone** | **Date** | **Topic Completed** |
| --- | --- | --- |
| 1 | Nov 21 | Run existing models and compare speed and qualitative goodness |
| 2 | Nov 28 | Apply static backgrounds(solid colors, static images) |
| 3 | Dec 5 | Integrate gesture, such that backgrounds change on the basis of the gesture. |

**Organization of the team**

| **Topic** | **Worked By** |
| --- | --- |
| Project Idea Brainstorming, Documentation and Code quality | Everyone |
| Understanding existing hand gesture models and segmentation models | Everyone |
| Gesture Detection | Ramprasad Kokkula, Pavan Kumar Jonnadula
 |
| Segmentation of background | Aakanksha Padmanabhan, Vaishnavi Patil |