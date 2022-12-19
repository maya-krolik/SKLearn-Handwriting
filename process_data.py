"""
    Maya Krolik
    Advanced Honors CS
    November 2022
    Data: https://www.kaggle.com/datasets/corrphilip/numeral-gestures?select=stroke.csv
"""
import pandas as pd

# ------------------------------------------------------------------------------
def main():
    glyph = pd.read_csv('glyph.csv')
    # glyph.drop(['Z_ENT', 'ZINDEX', 'ZDEVICE'])
    
    subject = pd.read_csv('subject.csv')
    # subject.drop(['Z_ENT', 'ZINDEX', 'ZNATIVELANGUAGE'])
    
    total_dataset = pd.merge(subject, glyph, left_on='Z_PK', right_on='ZSUBJECT')
    # print(total_dataset)
    unnecessary_columns = ['Z_PK_x', 'Z_PK_y', 'ZNATIVELANGUAGE', 'Z_ENT_x', 'Z_ENT_y', 'ZDEVICE', 'ZAGE']
    for i in unnecessary_columns:
        total_dataset.pop(i)

    rows, columns = total_dataset.shape

    # change the words into numbers
    
    # 0 --> LEFT
    # 1 --> RIGHT
    num_hand_list = []
    for j in range(rows):
        if str(total_dataset.at[j, "ZHANDEDNESS"]) == "left":
            num_hand_list.append(0)
        else:
            num_hand_list.append(1)
    numerical_handedness = pd.DataFrame({"Handedness": num_hand_list})

    # 0 --> male
    # 1 --> female
    num_gend_list = []
    for k in range(rows):
        if str(total_dataset.at[k, "ZSEX"]) == "male":
            num_gend_list.append(0)
        else:
            num_gend_list.append(1)
    numerical_gender = pd.DataFrame({"Gender": num_gend_list})

    # 0 --> index
    # 1 --> thumb
    num_finger_list = []
    for k in range(rows):
        if str(total_dataset.at[k, "ZFINGER"]) == "index":
            num_finger_list.append(0)
        else:
            num_finger_list.append(1)
    numerical_finger = pd.DataFrame({"Finger": num_finger_list})

    # remove old columns from dataset
    total_dataset.pop("ZHANDEDNESS")
    total_dataset.pop("ZSEX")
    total_dataset.pop("ZFINGER")

    # merge new columns into dataset
    total_dataset = pd.concat([total_dataset, numerical_gender], axis = 1)
    total_dataset = pd.concat([total_dataset, numerical_handedness], axis = 1)
    total_dataset = pd.concat([total_dataset, numerical_finger], axis = 1)

    total_dataset.to_csv('data.csv')

main()
