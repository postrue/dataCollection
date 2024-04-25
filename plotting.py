import csv
import matplotlib.pyplot as plt

user = 'asiyah'
date = '0226'

# Step 1: Read the CSV file
for t in range(0, 61, 10):
    file_path = f"./Data/{user}_{date}/{user}_{t}.csv"

    # Step 2: Parse the file and extract every third value starting from the first
    y_values = [[] for _ in range(9)]
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        ondetect = False
        for row in reader:
            row_values = []
            for value in row:
                try:
                    float_value = float(value)
                    row_values.append(float_value)
                except ValueError:
                    pass  # Skip non-numeric values
                
            # row_values = [float(value) for value in row]
            # extracted_values = row_values[2::3]  # Extract every third value starting from the first
            if len(row_values) == 10:
                for _ in range(9):
                    y_values[_].append(row_values[_])
                # extracted_values = row_values[0]  # Extract every third value starting from the first
                # y_values.append(extracted_values)

    # Step 3: Plot the extracted values
    coords = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    titles = ['X3','Y3','Z3','X2','Y2','Z2','X1','Y1','Z1']
    fig, axs = plt.subplots(3, 3)
    for i in range(len(coords)):
        ax = axs[coords[i][0], coords[i][1]]
        ax.plot(y_values[i])
        ax.set_xlabel('Index')
        ax.set_ylabel('Signal')
        ax.set_title(f"{titles[i]} - {t} min")
    #axs[0,0].grid(True)
    plt.tight_layout()
    plt.show()


# try:
#     float_value = float(value)
#     row_values.append(float_value)
# except ValueError:
#     pass  # Skip non-numeric values