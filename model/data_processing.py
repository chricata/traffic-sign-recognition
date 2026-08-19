import pandas as pd
import os

from PIL import Image

classes_description_file = "traffic-sign-recognition/dataset/signs/Classes_Description.xlsx"
scenes_train_dir = "traffic-sign-recognition/dataset/scenes/train"
scenes_test_dir = "traffic-sign-recognition/dataset/scenes/test"
dataset_dir = "traffic-sign-recognition/dataset"


def rename_images_folder(path):
    """Renames the training scene images directory for YOLO training.

    Parameters
    ----------
    path : str
        The path to the training scene directory.
    """    
    if os.path.exists(path + "imgs") and not os.path.exists(path + "/images"):
        os.rename(path + "imgs", path + "/images")


def get_classes_description(path):
    """Loads and returns the sign class descriptions from an Excel file.

    Renames the 'Name' column to 'Category' for better readability.

    Parameters
    ----------
    path : str
        The path to the Excel file containing the class descriptions.

    Returns
    -------
    pandas dataframe
        A dataframe containing the class descriptions with columns 'Id', 'Category', and 'Description'.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Classes description file not found at {path}")

    classes_description_df = pd.read_excel(path, header=1)
    classes_description_df = classes_description_df[['Id', 'Name', 'Description']]
    classes_description_df.rename(columns={"Name": "Category"}, inplace=True)

    return classes_description_df


def load_data(scenes_path, meta_path, classes):
    """Loads the traffic scene meta data and combines it with the class descriptions.

    Parameters
    ----------
    scenes_path : str
        The path to the directory containing the traffic scene images.
    meta_path : str
        The path to the CSV file containing the metadata.
    classes : pandas dataframe
        A dataframe containing the class descriptions.

    Returns
    -------
    pandas dataframe
        A dataframe containing all the scene meta data combined with class descriptions.
            
    Raises
    ------
    FileNotFoundError
        If the specified file or directory does not exist at the given path.
    """
    if not os.path.exists(scenes_path):
        raise FileNotFoundError(f"Traffic scenes directory not found at {scenes_path}")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta data file not found at {meta_path}")

    meta_data = pd.read_csv(meta_path)

    combined_data = pd.merge(meta_data, classes, left_on='class_id', right_on='Id', how='left')
    combined_data.drop(columns=['Id'], inplace=True)
    combined_data['image_path'] = scenes_path + '/' + combined_data['image_id'].astype(str) + '.jpg'

    return combined_data


def clean_data(df):
    """Cleans the combined data by removing rows with missing descriptions.

    Parameters
    ----------
    df : pandas dataframe
        A dataframe containing the combined scene meta data and class descriptions.

    Returns
    -------
    pandas dataframe
        A dataframe containing the data without rows that have missing descriptions.
    """
    df = df[df["Description"].notna()]
    return df


def create_class_mapping(data):
    """Creates a mapping from class descriptions to numbered labels.

    Parameters
    ----------
    data : pandas dataframe
        A dataframe containing the training data required for model training.

    Returns
    -------
    dictionary
        A dictionary mapping each unique class description to an integer label
    """
    unique_classes = sorted(data['Description'].unique())
    mapping = {}
    for i, description in enumerate(unique_classes):
        mapping[description] = i
    return mapping


def create_labels(data, class_mapping, label_dir):
    """Creates label files for each scene in the dataset for YOLO training.

    Parameters
    ----------
    data : pandas dataframe
        A dataframe containing the scene image data.
    
    class_mapping : dictionary
        A dictionary that maps all class descriptions to an integer.

    label_dir : str
        The directory where the label files will be saved.
    """
    os.makedirs(label_dir, exist_ok=True)

    for image_id, group in data.groupby('image_id'):
        label_path = os.path.join(label_dir, str(image_id) + '.txt')
        with Image.open(group['image_path'].iloc[0]) as img:
            width, height = img.size
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                class_id = class_mapping[row['Description']]

                x1, y1 = row['xtl'], row['ytl']
                x2, y2 = row['xbr'], row['ybr']

                xc = (x1 + x2) / 2 / width
                yc = (y1 + y2) / 2 / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height

                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def create_yaml_file(class_mapping, output_path):
    """Creates a YAML file required for YOLO training.

    Parameters
    ----------
    class_mapping : dictionary
        A dictionary that maps all class descriptions to an integer.

    output_path : str
        The path where the YAML file will be saved.
    """
    yaml_content = f"path: {os.path.abspath(dataset_dir)}/scenes\n"
    yaml_content += f"train: train/images\n"
    yaml_content += f"val: test/images\n"
    yaml_content += f"names:\n"
    for description, id in class_mapping.items():
        yaml_content += f" {id}: {description}\n"

    with open(output_path, 'w') as f:
        f.write(yaml_content)


def main():
    rename_images_folder(scenes_train_dir)

    classes = get_classes_description(classes_description_file)

    train_data = load_data(scenes_train_dir + "/images", scenes_train_dir + "/meta_train.csv", classes)
    test_data = load_data(scenes_test_dir + "/images", scenes_test_dir + "/meta_test.csv", classes)

    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    class_mapping = create_class_mapping(train_data)

    create_labels(train_data, class_mapping, scenes_train_dir + "/labels")
    create_labels(test_data, class_mapping, scenes_test_dir + "/labels")

    create_yaml_file(class_mapping, f"{dataset_dir}/data.yaml")


if __name__ == '__main__':
    main()