import pandas as pd
import matplotlib.pyplot as plt

def open_file(fileName):
    data = pd.read_csv(fileName)
    return data

# def delete_outliners(data, column, min_range, max_range):
#     data_withoutOutliners = data[column].between(min_range, max_range, inclusive=True)
#     show_data_info(data_withoutOutliners)



if __name__ == '__main__':
    data = open_file("/Users/krystelbaca/Documents/Mineria_datos/proyecto-final/housing-market-datamining/train.csv")
    # Crea la grafica donde se ven los outliers
    #create_whisker_plot1(data)


    # delete_outliners(data, "OverallCond", 4, 7)

    # show_data_info(data)
    #data = get_feature_subset(data, "Survived", "Pclass", "Sex", "Embarked")
    #data = delete_colum(data, "PassengerId")

    #data = delete_missing_values(data, 'instance')
    #data = replace_missing_values_with_constant(data, 'Age', -1)
    #replace_missing_values_with_mean(data, 'Age')

    # print(data)