import pandas as pd
import matplotlib.pyplot as plt


def open_file(fileName):
    data = pd.read_csv(fileName)
    return data


# def delete_outliners(data, column, min_range, max_range):
#     data_withoutOutliners = data[column].between(min_range, max_range, inclusive=True)
#     show_data_info(data_withoutOutliners)

def create_whisker_plot1(data):
    print(data['full_sq'].size)
    data['full_sq'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()


def create_whisker_plot2(data):
    print(data['life_sq'].size)
    data['life_sq'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()


def create_whisker_plot3(data):
    print(data['floor'].size)
    data['floor'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot4(data):
    print(data['max_floor'].size)
    data['max_floor'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot5(data):
    print(data['build_year'].size)
    data['build_year'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()


if __name__ == '__main__':
    data = open_file("/Users/krystelbaca/Documents/Mineria_datos/proyecto-final/housing-market-datamining/train.csv")
    # Crea la grafica donde se ven los outliers
    # create_whisker_plot1(data)
    #create_whisker_plot2(data)
    #create_whisker_plot3(data)
    create_whisker_plot4(data)
    create_whisker_plot5(data)