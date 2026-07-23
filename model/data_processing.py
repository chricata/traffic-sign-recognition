import pandas as pd


classes_description_file = "traffic-sign-recognition/dataset/signs/Classes_Description.xlsx"
scenes_train_dir = "traffic-sign-recognition/dataset/scenes/train"
scenes_test_dir = "traffic-sign-recognition/dataset/scenes/test"

def get_classes_description(path):
    classes_description_df = pd.read_excel(path, header=1)
    classes_description_df = classes_description_df[['Id', 'Name', 'Description']]
    classes_description_df.rename(columns={"Name": "Category"}, inplace=True)

    return classes_description_df


def load_data(scenes_path, meta_path, classes):
    meta_data = pd.read_csv(meta_path)

    combined_data = pd.merge(meta_data, classes, left_on='class_id', right_on='Id', how='left')
    combined_data.drop(columns=['Id'], inplace=True)
    combined_data['image_path'] = scenes_path + '/' + combined_data['image_id'].astype(str) + '.jpg'

    return combined_data


def main():
    classes = get_classes_description(classes_description_file)

    train_data = load_data(scenes_train_dir + "/imgs", scenes_train_dir + "/meta_train.csv", classes)
    test_data = load_data(scenes_test_dir + "/imgs", scenes_test_dir + "/meta_test.csv", classes)

    print(train_data.head())
    print(test_data.head())


if __name__ == '__main__':
    main()