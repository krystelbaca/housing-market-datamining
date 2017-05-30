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

def create_whisker_plot6(data):
    print(data['kitch_sq'].size)
    data['kitch_sq'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot7(data):
    print(data['school_km'].size)
    data['school_km'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot8(data):
    print(data['state'].size)
    data['state'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot9(data):
    print(data['num_room'].size)
    data['num_room'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot10(data):
    print(data['material'].size)
    data['material'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot11(data):
    print(data['metro_min_avto'].size)
    data['metro_min_avto'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot12(data):
    print(data['industrial_km'].size)
    data['industrial_km'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot13(data):
    print(data['green_zone_km'].size)
    data['green_zone_km'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()

def create_whisker_plot14(data):
    print(data['hospital_beds_raion'].size)
    data['hospital_beds_raion'].plot(kind='box', subplots=True, layout=(2, 14), sharex=False, sharey=False)
    plt.show()



if __name__ == '__main__':
    data = open_file("/Users/krystelbaca/Documents/Mineria_datos/proyecto-final/housing-market-datamining/train.csv")
    # Crea la grafica donde se ven los outliers
    # create_whisker_plot1(data)
    #create_whisker_plot2(data)
    #create_whisker_plot3(data)
    # create_whisker_plot4(data)
    # create_whisker_plot5(data)
    # create_whisker_plot6(data)
    #create_whisker_plot7(data)
    #create_whisker_plot8(data)
    #create_whisker_plot9(data)
    #create_whisker_plot10(data)
    #create_whisker_plot11(data)
    #create_whisker_plot12(data)
    # create_whisker_plot13(data)
    create_whisker_plot14(data)