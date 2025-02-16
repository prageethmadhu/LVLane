import numpy as np
import matplotlib.pyplot as plt
import json

def fileInitialFormatter(in_filename,out_fileName,rawName):
    # Opening JSON file
    f = open(in_filename)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    #data

    # create tusimple format labels from via labels using spline
    from scipy.interpolate import CubicSpline


    h_samples = []
    for i in range(160,720,10):
        h_samples.append(i)

    with open(out_fileName, "w") as outfile:
        for each_image in data:
            label = data[each_image]
            filename = label["filename"]
            lanes = np.array([], dtype=int)
            count_lanes = 0
            data_samples = []
            for each_lane in label["regions"]:           
                label_coordinates = each_lane["shape_attributes"]
                x = label_coordinates["all_points_x"]
                y = label_coordinates["all_points_y"]
                data_samples = [(slp, cls) for slp, cls in zip(x, y)]
                data_samples.sort(key = lambda x: x[1]) 
                x_cor = [] 
                y_cor = []
                for sample in data_samples:
                    x_cor.append(sample[0])
                    y_cor.append(sample[1])
                    
                y_values = [i//10 for i in y_cor]
                cs = CubicSpline(y_cor, x_cor)
                y_min = min(y_values)*10+10

                tusimple_y = []
                tusimple_x = []
                for i in range(y_min, max(y_values)*10, 10):
                    tusimple_y.append(i)
                    tusimple_x.append(int(cs(i)))
                print(filename)
                y_min = tusimple_y[0]/10
                y_max = tusimple_y[-1]/10
                front = int(y_min - 16)
                back = int(71- y_max)
                front_arr = np.ones(front, dtype=int)*(-2)
                back_arr = np.ones(back, dtype=int)*(-2)
                tusimple_x = np.array([np.hstack([front_arr, tusimple_x, back_arr])])
                
                count_lanes += 1
                if count_lanes >1:
                    lanes = np.vstack([lanes, tusimple_x])
                else:
                    lanes = tusimple_x
                #print(lanes)

            dictionary = {"lanes": lanes.tolist(),
                    "h_samples": h_samples,
                    "raw_file": "clips/"+rawName+filename
            }
            
            # Serializing json
            json_object = json.dumps(dictionary)
            outfile.write(json_object)
            outfile.write('\n')


for raw in range(40,60,20):
    #raw=i*10
    in_f=f"C:\\Users\\Pragiya\\MSC_Research\\LVLane\\my_anno_all\\my_combined_key_frames\\{raw}\\refined_{raw}-{raw+20}.json"
    out_f=f"C:\\Users\\Pragiya\\MSC_Research\\LVLane\\my_anno_all\\my_combined_key_frames\\{raw}\\prerefined_{raw}-{raw+20}.json"
    print(raw,"\n",in_f,"\n",out_f)
    try:
        fileInitialFormatter(in_f,out_f,f"raw")
    except Exception as e:
        print(f"Error caught: {e}, but continuing the loop.")
        print("except::",raw,"\n",in_f,"\n",out_f)