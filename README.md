# activity-recognition

Pipfile contains dependencies

## To install Yolov3

- Install from source [https://github.com/madhawav/YOLO3-4-Py](https://github.com/madhawav/YOLO3-4-Py)
- Clone repo 
- Use download_models.sh to get weights and place them in /data/weights folder
- Change the cfg/coco.data to point to the right names path. Sample is: 
  ```bash
  classes= 80
  # train  = /home/pjreddie/data/coco/trainvalno5k.txt
  valid  = coco_testdev
  # valid = data/coco_val_5k.list
  names = /data/networks/yolov3/data/coco.names
  # backup = /home/pjreddie/backup/
  eval=coco
  ```
## Alternative of yolo(simple)
- pipenv install yolo34py-gpu 
- Follow [](https://github.com/ethereum-mining/ethminer/issues/731) for troubleshoot gcc version