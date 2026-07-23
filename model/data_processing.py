import pandas as pd


classes_description_file = "traffic-sign-recognition/dataset/signs/Classes_Description.xlsx"


def get_classes_description(path):
    classes_description_df = pd.read_excel(path, header=1)
    classes_description_df = classes_description_df[['Id', 'Name', 'Description']]
    classes_description_df.rename(columns={"Name": "Category"}, inplace=True)

    return classes_description_df


def main():
    classes = get_classes_description(classes_description_file)


if __name__ == '__main__':
    main()