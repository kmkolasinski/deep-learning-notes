import plotly.offline as py
import plotly.graph_objs as go
from matplotlib import collections  as mc
import matplotlib.pyplot as plt
import numpy as np

_targets = ['mammal.n.01', 'beagle.n.01', 'canine.n.02', 'german_shepherd.n.01',
           'collie.n.01', 'border_collie.n.01',
           'carnivore.n.01', 'tiger.n.02', 'tiger_cat.n.01', 'domestic_cat.n.01',
           'squirrel.n.01', 'finback.n.01', 'rodent.n.01', 'elk.n.01',
           'homo_sapiens.n.01', 'orangutan.n.01', 'bison.n.01', 'antelope.n.01',
           'even-toed_ungulate.n.01', 'ungulate.n.01', 'elephant.n.01', 'rhinoceros.n.01',
           'odd-toed_ungulate.n.01', 'mustang.n.01', 'liger.n.01', 'lion.n.01', 'cat.n.01', 'dog.n.01']


def set_targets(targets):
    global _targets
    _targets = targets    
    

def transitive_isometry(t1, t0):
    u'''
    computing isometry which move t1 to t0
    '''

    (x1, y1), (x0,y0) = t1, t0

    def to_h(z):
        return (1 + z)/(1 - z) * complex(0,1)

    def from_h(h):
        return (h - complex(0,1)) / (h + complex(0,1))

    z1 = complex(x1, y1)
    z0 = complex(x0, y0)

    h1 = to_h(z1)
    h0 = to_h(z0)

    def f(h):
        return h0.imag/h1.imag * (h - h1.real) + h0.real

    def ret(z):
        z = complex(z[0], z[1])
        h = to_h(z)
        h = f(h)
        z = from_h(h)
        return z.real, z.imag

    return ret


def plot_embedings2d(embeddings, edges_df, center_mammal, center_node='mammal.n.01'):

    targets = list(set([x for x in _targets]))    
    print(len(targets), ' targets found')
    
    # load embeddings        
    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0,0), 1., color='black', fill=False)
    ax.add_artist(circle)

    max_radius = (embeddings.x1**2 + embeddings.x2**2).map(np.sqrt).max()
    embeddings = embeddings/max_radius/1.01
    
    z = embeddings.loc[center_node]
    
    def apply_center(row):        
        x, y = isom((row[0], row[1]))
        return [x, y]
    
    if center_mammal:
        isom = transitive_isometry((z[0], z[1]), (0, 0))
        embeddings = embeddings.apply(apply_center, 1)
    
    lines = []
    for edge in edges_df.iterrows(): 
        from_z = embeddings.loc[edge[1]['from']]
        to_z = embeddings.loc[edge[1]['to']]
        lines.append([(from_z[0], from_z[1]), (to_z[0], to_z[1])])


    lc = mc.LineCollection(lines, linewidths=0.5, antialiaseds=True)
    ax.add_collection(lc)
    
    ax.plot(embeddings.values[:, 0], embeddings.values[:, 1], 'o', color='k', markersize=0.5)
    
    for n in targets:
        z = embeddings.loc[n]    
        x, y = z[0], z[1]
    
        if n == center_node:
            ax.plot(x, y, 'o', color='g', markersize=5.5)
            ax.text(x+0.001, y+0.001, n, color='r', alpha=0.6)
        else:
            ax.plot(x, y, 'o', color='k', markersize=5.5)
            ax.text(x+0.001, y+0.001, n, color='b', alpha=0.6)
    
    plt.show()


def plotly_embedings2d(embeddings, edges_df, center_mammal=True, name='blank', center_node='mammal.n.01'):

    def apply_center(row):        
            x, y = isom((row[0], row[1]))
            return [x, y]

    if center_mammal:
        z = embeddings.loc[center_node]
        isom = transitive_isometry((z[0], z[1]), (0, 0))
        embeddings = embeddings.apply(apply_center, 1)

    # Create a trace
    nodes = go.Scattergl(
        x = embeddings.x1,
        y = embeddings.x2,
        mode = 'markers',
        marker = dict(
            size = 5,
            color = 'rgba(0, 0, 0, .5)',
            line = dict(
                width = 0.1,
                color = 'rgb(0, 0, 0, 0.8)'
            )
        ),
        text = embeddings.index,
        name = 'nodes'
    )

    edges = go.Scattergl(
        x=[], 
        y=[], 
        mode='lines',
        name='connections',
        line = dict(
            width = 0.5,
            color='rgba(150, 150, 150, 50)'
        ),

    )
    for edge in edges_df.iterrows(): 
        from_z = embeddings.loc[edge[1]['from']]
        to_z = embeddings.loc[edge[1]['to']]
        edges['x'] += [from_z[0], to_z[0], None]
        edges['y'] += [from_z[1], to_z[1], None]


    x = []
    y = []
    names = []
    for target in _targets:
        z = embeddings.loc[target]    
        x.append(z[0]) 
        y.append(z[1]) 
        names.append(target.split('.')[0])


    target_nodes = go.Scattergl(
        x = x,
        y = y,
        mode = 'markers',
        marker = dict(
            size = 5,
            color = 'rgba(150, 71, 96, 100)',
        ),
        text = names,
        name = 'targets'
    )

    data = [edges, nodes, target_nodes]

    annotations = [
            dict(
                x=x_val,
                y=y_val,
                xref='x',
                yref='y',
                text=text,
                showarrow=True,
                font=dict(
                    family='Courier New, monospace',
                    size=8,
                    color='#000000'
                ),
                align='center',
                arrowhead=0,
                arrowsize=2,
                arrowwidth=1,
                arrowcolor='#000000',
                ax=0,
                ay=-10,
                bordercolor='rgba(6, 6, 6, 100)',
                borderwidth=1,
                borderpad=1,
                bgcolor='rgba(250, 250, 250, 100)',
                opacity=0.9
            )
        for x_val, y_val, text in zip(x, y, names)
    ]


    layout = {
        'xaxis': {
            'range': [-1.1, 1.1],        
        },
        'yaxis': {
            'range': [-1.1, 1.1]
        },
        'width': 800,
        'height': 800,
        'shapes': [
            # unfilled circle
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'x0': -1,
                'y0': -1,
                'x1': 1,
                'y1': 1,
                'line': {
                    'color': 'rgba(50, 171, 96, 1)',
                },
            },    
        ],
        'annotations': annotations
    }


    py.init_notebook_mode(connected=True)

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=name)


def plotly_embedings3d(embeddings, edges_df):
    # Create a trace
    nodes = go.Scatter3d(
        x = embeddings.x1,
        y = embeddings.x2,
        z = embeddings.x3,
        mode = 'markers',
        marker = dict(
            size = 2,
            color = 'rgba(0, 0, 0, .5)',
            line = dict(
                width = 0.1,
                color = 'rgb(0, 0, 0, 0.8)'
            )
        ),
        text = embeddings.index,
        name = 'nodes'
    )

    edges = go.Scatter3d(
        x=[], 
        y=[],
        z=[],
        mode='lines',
        name='connections',
        line = dict(
            width = 1.0,
            color='rgba(150, 150, 150, 50)'
        ),

    )
    for edge in edges_df.iterrows(): 
        from_z = embeddings.loc[edge[1]['from']]
        to_z = embeddings.loc[edge[1]['to']]
        edges['x'] += [from_z[0], to_z[0], None]
        edges['y'] += [from_z[1], to_z[1], None]
        edges['z'] += [from_z[2], to_z[2], None]


    x = []
    y = []
    z = []
    names = []
    for target in _targets:
        xyz = embeddings.loc[target]    
        x.append(xyz[0]) 
        y.append(xyz[1]) 
        z.append(xyz[2]) 
        names.append(target.split('.')[0])


    target_nodes = go.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba(150, 71, 96, 100)',
        ),
        text = names,
        name = 'targets'
    )

    data = [edges, nodes, target_nodes]

    annotations = [
            dict(
                x=x_val,
                y=y_val,
                z=z_val,
                text=text,
                showarrow=False,
                font=dict(
                    family='Courier New, monospace',
                    size=8,
                    color='#000000'
                ),
                align='center',                
                ax=0,
                ay=-30,
                bordercolor='rgba(6, 6, 6, 100)',
                borderwidth=1,
                borderpad=1,
                bgcolor='rgba(250, 250, 250, 100)',
                opacity=0.9
            )
        for x_val, y_val, z_val, text in zip(x, y, z, names)
    ]


    layout = go.Layout(
       width = 1000,
       height = 1000,    
       scene = dict(
        aspectratio = dict(
          x = 1,
          y = 1,
          z = 1
        ),
        camera = dict(
          center = dict(
            x = 0,
            y = 0,
            z = 0
          ),
          eye = dict(
            x = 1.96903462608,
            y = -1.09022831971,
            z = 0.405345349304
          ),
          up = dict(
            x = 0,
            y = 0,
            z = 1
          )
        ),
        dragmode = "turntable",
        xaxis = dict(
          title = "x",
        ),
        yaxis = dict(
          title = "y",
        ),
        zaxis = dict(
          title = "z",
        ),
        annotations = annotations,
      ),
      xaxis = dict(title = "x"),
      yaxis = dict(title = "y")
    )


    py.init_notebook_mode(connected=True)

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
