import numpy as np
import matplotlib.pyplot as plt
import json

# order lanes from left to right using slope, also delete any lanes with less than 4 coordinates
def coodinateCaliberate(in_f,out_f,raw):

    with open(out_f, 'w') as outfile:
        json_path = in_f
        with open(json_path) as f:
            json_lines = f.readlines()
            line_index = 0
            while line_index < len(json_lines):
                line = json_lines[line_index]
                label = json.loads(line)

                # ---------- clean and sort lanes -------------
                lanes = []
                _lanes = []
                slope = [] # identify 0th, 1st, 2nd, 3rd, 4th, 5th lane through slope
                for i in range(len(label['lanes'])):
                    l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples']) if x >= 0]
                    if (len(l)>1):
                        _lanes.append(l)
                        slope.append(np.arctan2(l[-1][1]-l[0][1], l[0][0]-l[-1][0]) / np.pi * 180)
                _lanes = [_lanes[i] for i in np.argsort(slope)]   # arrange lanes based on slope
                data = [(ind, slp) for ind, slp in enumerate(slope)]
                data.sort(key = lambda x: x[1])                   # arrange (slope, class_list) based on slope
                slope = [slope[i] for i in np.argsort(slope)]     # arrange slope low to high

                ind = [c[0] for c in data]
                label['lanes'] = [label['lanes'][i] for i in ind]
                
                x = []
                for i in range(len(label['lanes'])):
                    l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples']) if x >= 0]
                    if len(l) < 4:
                        x.append(i)
                for index in sorted(x, reverse=True):
                    del label['lanes'][index]
                if len(label['lanes']) > 0:
                    json_object = json.dumps(label)
                    outfile.write(json_object)
                    outfile.write('\n')

                line_index += 1


for raw in range(120,140,20):
    #raw=i*10
    in_f=f"C:\\Users\\Pragiya\\MSC_Research\\LVLane\\my_anno_all\\my_combined_key_frames\\{raw}\\prerefined_{raw}-{raw+20}.json"
    out_f=f"C:\\Users\\Pragiya\\MSC_Research\\LVLane\\my_anno_all\\my_combined_key_frames\\{raw}\\orefined_{raw}-{raw+20}.json"
    print(raw,"\n",in_f,"\n",out_f)
    try:
        coodinateCaliberate(in_f,out_f,f"raw")
    except Exception as e:
        print(f"Error caught: {e}, but continuing the loop.")
        print("except::",raw,"\n",in_f,"\n",out_f)