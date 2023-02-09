import pandas as pd
import cv2 
import os
from tqdm import tqdm
root = "./ISIC_2019"

raw_data_meta_df = pd.read_csv(os.path.join(root, "ISIC_2019_Training_GroundTruth.csv"))
X = raw_data_meta_df.iloc[:]['image'] # only image names, not actual images


for ind  in tqdm(range(len(X))):
    img_name = X.iloc[ind]
    image = cv2.imread(os.path.join(root, "ISIC_2019_Training_Input", img_name+".jpg"), cv2.IMREAD_COLOR) 
    image = cv2.resize(image, (512, 512), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(root, "ISIC_2019_Training_Input", img_name+".jpg"), img=image)