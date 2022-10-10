#  Image Object Relation Classification

CE7454 Project: DEEP LEARNING FOR DATA SCIENCE

## Challenge Description

Most of the current computer vision tasks focus on the objects in the image. For example, image classification asks models to identify the object classes in the image. Object detection and image segmentation further require the model to identify objects in the image and find their locations (with different granularities). However, the information of **relations** are, by far, ignored by the community. Therefore, in this assignment, the PSG Classification Challenge, we would like to invite our students to explore how to train a computer vision model to well identify the relations in the images.

![elephant](elephant.jpeg)

To be more specific, when the model sees the photo above, rather than providing the information of the objects (e.g., person, elephant), it should output **3 salient relations** such as *feeding*, *leaning-on*, and *looking-at*. There are 56 classes of relation in total. We notice that the first 6 relations (in gray) are too vague for PSG classification task, so we remove them in this project, so that the label space only contains 50 relations (in orange). In other word, the PSG classification task is a 50-class multi-label classification problem.

![relations](relations.jpeg)

## Data

4500 training data, 500 validation data, and 500 test data. All the images and panoptic segmentation are from COCO dataset. 
