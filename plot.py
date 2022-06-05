import json
import os
from plotting_functions import periodic_table_heatmap

with open('plotter.json', 'r') as fid:
    m = json.load(fid)

save_location = 'figures'

#Figure 1

periodic_table_heatmap(m['figure1']['elem_dict'],show_plot=False,
                           blank_color='lightgrey',edge_color='white', value_format='%.0f',
                            cmap='coolwarm', save_name=os.path.join(save_location,'figure1.pdf'))

#Figure 2

periodic_table_heatmap(m['figure2']['elem_dict'],show_plot=False,
                           blank_color='lightgrey',edge_color='white', value_format='%.0f',
                            cmap='coolwarm', save_name=os.path.join(save_location,'figure2.pdf'))


