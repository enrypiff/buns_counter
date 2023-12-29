# Buns counter
In this project I've started from an isssue saw during my as automation engineer. It's difficult in automated industrial production line to use just sensors when the enviroment is very dynamic, so I've decided to use the power of YOLO in order to fix this problem.

I manually annotated 12 photos I took during the commissionig of the production line and augmented them with Roboflow to have some some samples for my dataset. 

![alt text](https://github.com/enrypiff/buns_counter/blob/main/instance%20segmentation.png)

Then I trained YOLO v8 for instance segmentation (I noticed that is more faster and accurate for very small datasets and in the application I used just the bounding boxes.

To handle the tracking problem for the counter I've used the SORT tracking algorithm.

This is the final result, a very promising use of YOLO in industrial field.

https://github.com/enrypiff/buns_counter/assets/139701172/3e040906-b8a1-43c7-b456-d441afb10d76

