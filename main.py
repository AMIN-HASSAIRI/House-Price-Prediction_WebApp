import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import csv
from flask import Flask, request, render_template
from flask_cors import cross_origin
from csv import writer

app = Flask(__name__)


@app.route("/")
@cross_origin()
def page():
    return render_template("home.html")


housePoland_file_path = 'Houses.csv'
house_data = pd.read_csv(housePoland_file_path, encoding='latin-1')

y = house_data.price

house_Poland_features = ['floor', 'latitude', 'longitude', 'rooms', 'sq', 'year']
X = house_data[house_Poland_features]
X.describe()
X.head()

# Define model. Specify a number for random_state to ensure same results each run
poland_model = DecisionTreeRegressor(random_state=1)

# Fit model
poland_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())

fields = ['floor', 'latitude', 'longitude', 'rooms', 'sq', 'year']
rows = [[6, 50.456, 17.456, 4, 36, 2020],
        [7, 52.563, 16.666, 3, 60, 2019],
        [8, 51.996, 19.124, 2, 92, 2011],
        [5, 49.466, 18.452, 3, 23, 2021]]
# name of csv file
filename = "New_Houses.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, lineterminator='\n')

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)

new_houses_path = 'New_Houses.csv'
new_house_data = pd.read_csv(new_houses_path)

print("The predictions are")
print(poland_model.predict(new_house_data))


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        floor_num = int(request.form["Floor"])
        latitude = float(request.form["Latitude"])
        longitude = float(request.form["Longitude"])
        room_num = int(request.form["Room"])
        square_meters = float(request.form["Square"])
        year = int(request.form["Year"])

        # List
        house_list = [floor_num, latitude, longitude, room_num, square_meters, year]

        # Open our existing CSV file in append mode
        # Create a file object for this file
        with open('New_Houses.csv', 'a') as f_object:
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object, lineterminator='\n')

            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(house_list)

            # Close the file object
            f_object.close()

        new_house_data2 = pd.read_csv(new_houses_path)

        return render_template('home.html',
                               prediction_text="House Prediction Price Is: {}".format(
                                   poland_model.predict(new_house_data2)))

    return render_template("home.html")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

