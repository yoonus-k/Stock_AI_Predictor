import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_pips(data: np.array, n_pips: int, dist_measure: int):
    # dist_measure
    # 1 = Euclidean Distance
    # 2 = Perpindicular Distance
    # 3 = Vertical Distance

    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]] # Price

    for curr_point in range(2, n_pips):

        md = 0.0 # Max distance
        md_i = -1 # Max distance index
        insert_index = -1

        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj] # Time difference
            price_diff = pips_y[right_adj] - pips_y[left_adj] # Price difference
            slope = price_diff / time_diff # Slope
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope; # y = mx + c

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                # Euclidean Distance:
                # Use when you need the shortest path between two points.
                # Example: Calculating the distance between two cities on a map.
                
                # Perpendicular Distance:
                # Use when you need the shortest distance from a point to a line.
                # Example: Finding the closest point on a road to a given location.
                
                # Vertical Distance:
                # Use when you care only about the vertical difference between a point and a line.
                # Example: Measuring the error between observed and predicted values in regression analysis.
                
                d = 0.0 # Distance
                if dist_measure == 1: # Euclidean distance
                    d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5 # Left distance formula : sqrt((x1 - x2)^2 + (y1 - y2)^2)
                    d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5 # Right distance formula : sqrt((x1 - x2)^2 + (y1 - y2)^2)
                elif dist_measure == 2: # Perpindicular distance
                    d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5 # Perpindicular distance formula : |Ax + By + C| / (A^2 + B^2)^0.5
                else: # Vertical distance    
                    d = abs( (slope * i + intercept) - data[i] ) # Vertical distance formula : |Ax + By + C| 

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y


if __name__ == "__main__":
    data = pd.read_csv('C:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/XAUUSD60.csv')
    # create the datetime index from the Date and the Time columns
    data['Date'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data['Date'] = data['Date'].astype('datetime64[s]')
    data = data.set_index('Date')

# drop the time column
    data = data.drop(columns=['Time'])
    i = 1198
    x = data['Close'].iloc[i-40:i].to_numpy()
    
    pips_x, pips_y = find_pips(x, 5, 2)
    
    # convert pips_y to numpy array
    pips_y = np.array(pips_y)
    print(pips_y.shape)

    pd.Series(x).plot()
    for i in range(5):
        plt.plot(pips_x[i], pips_y[i], marker='o', color='red')

    plt.show()



