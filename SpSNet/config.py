
val_reps=1 # Number of test views, 1 or more
CLASS_LABELS_DICT = {
    'single': ['unlabeled','car','bicycle','motorcycle','truck','other-vehicle','person','bicyclist','motorcyclist','road','parking','sidewalk','other-ground','building','fence','vegetation','trunk','terrain','pole','traffic-sign'],
    'muti':["unlabeled", "car","bicycle","motorcycle","truck","other-vehicle","person","bicyclist","motorcyclist","road","parking","sidewalk","other-ground","building","fence",
                "vegetation","trunk","terrain","pole","traffic-sign","moving-car","moving-bicyclist","moving-person","moving-motorcyclist","moving-other-vehicle","moving-truck"]
}
CLASS_LABELS = CLASS_LABELS_DICT['muti']
UNKNOWN_ID = 0
N_CLASSES = len(CLASS_LABELS)


