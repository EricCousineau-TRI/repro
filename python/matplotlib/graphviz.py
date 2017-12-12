#!/usr/bin/python

import tempfile
from StringIO import StringIO
# import networkx as nx
import pydotplus as pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# sudo apt install python-networkx python-pygraphviz # DOES NOT WORK
# pip install networkx pygraphviz pydotplus

dot = """
digraph _46955600 {
rankdir=LR
subgraph cluster46955600diagram {
color=black
concentrate=true
label="test_diagram";
subgraph cluster46955600inputports {
rank=same
color=lightgrey
style=filled
label="input ports"
_46955600_u0[color=blue, label="u0"];
_46955600_u1[color=blue, label="u1"];
_46955600_u2[color=blue, label="u2"];
}
subgraph cluster46955600outputports {
rank=same
color=lightgrey
style=filled
label="output ports"
_46955600_y0[color=green, label="y0"];
_46955600_y1[color=green, label="y1"];
_46955600_y2[color=green, label="y2"];
}
subgraph cluster46955600subsystems {
color=white
label=""
42291536 [shape=record, label="adder0|{{<u0>u0|<u1>u1} | {<y0>y0}}"];
46915616 [shape=record, label="adder1|{{<u0>u0|<u1>u1} | {<y0>y0}}"];
46913232 [shape=record, label="adder2|{{<u0>u0|<u1>u1} | {<y0>y0}}"];
46944592 [shape=record, label="integrator0|{{<u0>u0} | {<y0>y0}}"];
46948240 [shape=record, label="integrator1|{{<u0>u0} | {<y0>y0}}"];
42291536:y0 -> 46913232:u0;
46915616:y0 -> 46913232:u1;
42291536:y0 -> 46915616:u0;
42291536:y0 -> 46944592:u0;
46944592:y0 -> 46948240:u0;
_46955600_u0 -> 42291536:u0 [color=blue];
_46955600_u1 -> 42291536:u1 [color=blue];
_46955600_u2 -> 46915616:u1 [color=blue];
46915616:y0 -> _46955600_y0 [color=green];
46913232:y0 -> _46955600_y1 [color=green];
46948240:y0 -> _46955600_y2 [color=green];
}
}
}
"""

# filename = 'tmp.dot'
# with open(filename, 'w') as f:
#     f.write(dot)

# # https://stackoverflow.com/questions/4596962/display-graph-without-saving-using-pydot

# g = pydot.graph_from_dot_file(filename)
# png = filename + '.png'
# g.write_png(png)

# plt.axis('off')
# plt.imshow(plt.imread(png), aspect="equal")

def plot_dot(dot_text):
    """Renders a DOT graph in matplotlib. """
    g = pydot.graph_from_dot_data(dot_text)
    s = StringIO()
    g.write_png(s)
    s.seek(0)
    plt.axis('off')
    plt.imshow(plt.imread(s), aspect="equal")

plot_dot(dot)
plt.show()
