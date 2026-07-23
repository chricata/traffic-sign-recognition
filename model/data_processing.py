import pandas as pd
import os

classes_description_file = "traffic-sign-recognition/dataset/signs/Classes_Description.xlsx"
scenes_train_dir = "traffic-sign-recognition/dataset/scenes/train"
scenes_test_dir = "traffic-sign-recognition/dataset/scenes/test"

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


def main():
    classes = get_classes_description(classes_description_file)

    train_data = load_data(scenes_train_dir + "/imgs", scenes_train_dir + "/meta_train.csv", classes)
    test_data = load_data(scenes_test_dir + "/imgs", scenes_test_dir + "/meta_test.csv", classes)

    train_data = clean_data(train_data)
    test_data = clean_data(test_data)


if __name__ == '__main__':
    main()