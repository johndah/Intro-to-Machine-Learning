<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true},
jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
TeX: {
extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
equationNumbers: {
autoNumber: "AMS"
}
}
});
</script>


```python
dark_theme = False # Requires dark theme editor or browser, e.g. Dark Reader extension on Chrome
```


```python
from IPython.display import Latex, HTML, Markdown

def hide_input_code():   
    from IPython import display
    import binascii
    import os
    uid = binascii.hexlify(os.urandom(8)).decode()    
    html = """<div id="%s"></div>
    <script type="text/javascript">
        $(function(){
            var p = $("#%s");
            if (p.length==0) return;
            while (!p.hasClass("cell")) {
                p=p.parent();
                if (p.prop("tagName") =="body") return;
            }
            var cell = p;
            cell.find(".input").addClass("hide-in-slideshow")
        });
    </script>""" % (uid, uid)
    display.display_html(html, raw=True)


```


```python
%%html
<style>
 .container.slides .celltoolbar, .container.slides .hide-in-slideshow {
    display: None ! important;
}
</style>
```


<style>
 .container.slides .celltoolbar, .container.slides .hide-in-slideshow {
    display: None ! important;
}
</style>



# Artificial Intelligence
  * ## Robotics
      * ### <span style="color:blue">Path Planning (e.g. A*)</span>
      * #### Computer Vision
      ...
 
  * ## Machine Learning
    * ### <span style="color:blue">*Esemble Learning*</span>  
    * #### Bayesian Inference
    * #### Reinforcement Learning  
    * #### Generic Algorithms 
      ...  
    * ### Artificial Neural Networks
        * #### <span style="color:green">*The Perceptron*</span>
        * #### <span style="color:green">*Feed forward Neural Networks*</span>
        * ##### <span style="color:orange">Deep Learning</span>
            * Convolutional Neural Networks
            * Recurrent Neural Networks
            
              ...
              
          ...
            
<span style="color:blue">■</span> Show quick example  
<span style="color:green">■</span> Main focus  
<span style="color:orange">■</span> Maybe future topic

Logical AND Problem


```python
from IPython.display import Latex, HTML, Markdown
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
%matplotlib inline

def get_figure(size=(8, 8)):
    fig = plt.figure(figsize=size)
    ax = fig.gca()
    ax.set_axisbelow(True)
    plt.grid(linewidth=.2 if dark_theme else .7)
    
    if dark_theme:
        ax.patch.set_alpha(0.)
        label_color = 'white'
        ax.set_title(ax.get_title(), color=label_color)
        plt.grid(linewidth=.2)

        [ax.spines[spine].set_color(label_color) for spine in ax.spines]
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
        ax.tick_params(axis='x', colors=label_color)
        ax.tick_params(axis='y', colors=label_color)

    return ax
    

def plot_problem(ax, data, targets, title=''):
    for sample_index in range(len(data)):
        sample_input = data[sample_index].astype(int)
        ground_truth = targets[sample_index]

        if ground_truth: # If output is True
            sample_marker = 'P'
            sample_color = 'limegreen'
        else:
            sample_marker = 'X'
            sample_color = 'r'
            
        plt.plot(sample_input[0],
                sample_input[1],
                sample_marker,
                color=sample_color,
                markersize=13,
                label=ground_truth)
    plt.axis('equal')
    limits = [-.25, 1.25, -.25, 1.25]
   
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    handles, labels = ax.get_legend_handles_labels()
    legends = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*legends))
    plt.title(title)

    ticks = np.arange(limits[0], 2*limits[1], .25)
    tick_labels = [str(tick) for tick in ticks]
    tick_labels[np.where(ticks == 0)[0][0]] += '\n(False)'
    tick_labels[np.where(ticks == 1)[0][0]] += '\n(True)'
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    plt.axis(limits)
    
    return legends

```


```python
ax = get_figure()

data = [np.array([False, False]),
        np.array([True,  False]),
        np.array([False, True]),
        np.array([True,  True])]

targets = [False,
           False,
           False,
           True]

legends = plot_problem(ax, data, targets)
plt.show()
```


![png](output_6_0.png)



```python

def plot_classifier(legends='',
                    threshold=.5,
                    feature_index=0,
                    color='orange',
                    text_color='orange',
                    fill_alpha=.4,
                    linewidth=3,
                    fontsize=16):    
    
    limits = [-.25, 1.6]
    
    if feature_index == 0:
        plt.fill_betweenx(limits,
                          [threshold]*2,
                          [max(limits)]*2,
                          alpha=fill_alpha,
                          color='limegreen')
        classification_plot = plt.plot([threshold]*2,
                                       limits,
                                       linewidth=linewidth, 
                                       color=color)
        plt.text(threshold+.05, .4, r'threshold $t_1 = %s$'%np.round(threshold, 2), fontsize=fontsize, color=text_color)
        legends.append((classification_plot[0], r'Classification line for feature $x_1$'))

    else:
        plt.fill_between(limits,
                         [threshold]*2,
                         [max(limits)]*2,
                         alpha=fill_alpha,
                         color='limegreen')
        classification_plot = plt.plot(limits,
                                       [threshold]*2,
                                       linewidth=linewidth, 
                                       color=color)
        plt.text(-.1, threshold+.05, r'threshold $t_2 = %s$'%np.round(threshold, 2), fontsize=fontsize, color=text_color)
        legends.append((classification_plot[0], r'Classification line for feature $x_2$'))

    plt.legend(*zip(*legends))
    plt.axis(limits*2)

```


```python
import pandas as pd
import networkx as nx

def plot_decision_tree(features, thresholds):
    plt.figure(figsize=(14, 2+4*len(features)))

    if ('1' in features[0]):
        color = 'orange'
        bias = thresholds[0]
    else:
        color = 'dodgerblue'
        bias = thresholds[-1]
        
    feature1_node = [features[0] + r'$\geq %s$' % np.round(bias, 2)]*2
    output_nodes = [features[0] + '_yes',
                    features[0] + '_no']

    pos = {feature1_node[0]: (1.5, 2),
           output_nodes[0]: (1, 1),
           output_nodes[1]: (2, 1)}

    edge_labels = {(feature1_node[0], output_nodes[0]): 'yes',
                   (feature1_node[0], output_nodes[1]): 'no'}

    if len(features) == 2:
        feature1_node.extend([features[1] + r'$\geq %s$' % np.round(thresholds[0], 2)]*2)
        feature2_node = [features[1] + r'$\geq %s$' % np.round(thresholds[1], 2)]*4
        output_nodes.extend([features[0] + '_yes' + features[1] + '_no',
                             features[0] + '_no' + features[1] + '_no'])

        pos.update({feature1_node[-1]: (3.5, 2),
                    output_nodes[2]: (3, 1),
                    output_nodes[3]: (4, 1)})
        edge_labels.update({(feature1_node[2], output_nodes[2]): 'yes',
                            (feature1_node[2], output_nodes[3]): 'no'})

        pos2 = {feature2_node[0]: (2.5, 3),
                feature1_node[0]: (1.5, 2),
                feature1_node[-1]: (3.5, 2)}
        data_frame2 = {'feature2_node': feature2_node,
                       'feature1_node': feature1_node}
        network2 = nx.from_pandas_edgelist(pd.DataFrame(data_frame2),
                                           'feature2_node',
                                           'feature1_node',
                                           create_using=nx.DiGraph())

        label2 = {feature2_node[0]: feature2_node[0]}
        nodes2 = nx.draw_networkx_nodes(network2, pos2,
                                        nodelist=network2.nodes,
                                        node_size=1.e4,
                                        node_shape='o',
                                        node_color='dodgerblue',
                                        alpha=1.)

        
        nx.draw_networkx_edges(network2, pos2, node_size=1.e4,
                                       arrowstyle='->',
                                       edge_color='dodgerblue',
                                       width=4)
        nx.draw_networkx_labels(network2, pos2, label2, font_size=18)

        edge_labels2 = {(feature2_node[0], feature1_node[0]): 'yes',
                        (feature2_node[1], feature1_node[2]): 'no'}

        nx.draw_networkx_edge_labels(network2, pos2, bbox=dict(facecolor='none', edgecolor='none'),
                                     font_size=18,
                                     font_color='dodgerblue',
                                     #label_pos=.6,
                                     verticalalignment='bottom',
                                     edge_labels=edge_labels2)

    data_frame = {'feature1_node': feature1_node,
                  'output_nodes': output_nodes}

    network = nx.from_pandas_edgelist(pd.DataFrame(data_frame),
                                      'feature1_node',
                                      'output_nodes',
                                      create_using=nx.DiGraph())

    nodes = nx.draw_networkx_nodes(network, pos,
                                   nodelist=network.nodes,
                                   node_size=1.e4,
                                   node_shape='o',
                                   node_color=color,
                                   alpha=1.)

    labels = dict(zip(pos.keys(), pos.keys()))
    labels[feature1_node[-1]] = feature1_node[0]
    labels[output_nodes[0]] = 'True'
    for i in range(len(output_nodes)):
        labels[output_nodes[i]] = 'True' if i == 0 else 'False'

    nx.draw_networkx_labels(network, pos, labels, font_size=18)
    nx.draw_networkx_edges(network, pos, node_size=1.e4,
                                   arrowstyle='->',
                                   edge_color=color,
                                   width=4)
    nx.draw_networkx_edge_labels(network, pos, bbox=dict(facecolor='none', edgecolor='none'),
                                 font_size=18,
                                 font_color=color,
                                 verticalalignment='baseline',
                                 edge_labels=edge_labels)

    plt.xlim(0, 1+2*len(features))
    plt.ylim(0, 2+len(features))

    plt.axis('off')
    plt.show()


```


```python
hide_input_code()
Latex('''
Introduce weak classifier, defining $true$ for
\\begin{align*}
    x_1 \geq t_1,
\\end{align*}
for some threshold $t_1$.
Correctly classifies 3 out of 4.
''')
```


<div id="20b9d88c468482c7"></div>
    <script type="text/javascript">
        $(function(){
            var p = $("#20b9d88c468482c7");
            if (p.length==0) return;
            while (!p.hasClass("cell")) {
                p=p.parent();
                if (p.prop("tagName") =="body") return;
            }
            var cell = p;
            cell.find(".input").addClass("hide-in-slideshow")
        });
    </script>






Introduce weak classifier, defining $true$ for
$$
\begin{aligned}
    x_1 \geq t_1,
\end{aligned}
$$
for some threshold $t_1$.
Correctly classifies 3 out of 4.





```python
np.random.seed(0)
t1 = np.random.rand(1).item()

plot_decision_tree([r'$x_1$'], [t1])

ax = get_figure()
legends = plot_problem(ax, data, targets)
plot_classifier(legends, threshold=t1, feature_index=0)
```


![png](output_10_0.png)



![png](output_10_1.png)



```python
hide_input_code()
Latex('''
Introduce another weak classifier, defining $true$ for
\\begin{align*}
    x_2 \geq t_2,
\\end{align*}
for some threshold $t_2$.
Also correctly classifies 3 out of 4.
''')
```


<div id="f47e8fe4766e72e1"></div>
    <script type="text/javascript">
        $(function(){
            var p = $("#f47e8fe4766e72e1");
            if (p.length==0) return;
            while (!p.hasClass("cell")) {
                p=p.parent();
                if (p.prop("tagName") =="body") return;
            }
            var cell = p;
            cell.find(".input").addClass("hide-in-slideshow")
        });
    </script>






Introduce another weak classifier, defining $true$ for
$$
\begin{aligned}
    x_2 \geq t_2,
\end{aligned}
$$
for some threshold $t_2$.
Also correctly classifies 3 out of 4.





```python
t2 = np.random.rand(1).item()

plot_decision_tree([r'$x_2$'], [t2])

ax = get_figure()
legends = plot_problem(ax, data, targets)

plot_classifier(legends, threshold=t2, feature_index=1, color='mediumslateblue', text_color='navy')
```


![png](output_12_0.png)



![png](output_12_1.png)



```python
hide_input_code()

Latex('''
Boost classifier by voting, i.e. combining weak classifiers
\\begin{align*}
\\begin{cases}
    x_1 \geq t_1, \\\ 
    x_2 \geq t_2.
\\end{cases}
\\end{align*}
Will correctly classify all samples.
''')

```


<div id="715564790d36b0b1"></div>
    <script type="text/javascript">
        $(function(){
            var p = $("#715564790d36b0b1");
            if (p.length==0) return;
            while (!p.hasClass("cell")) {
                p=p.parent();
                if (p.prop("tagName") =="body") return;
            }
            var cell = p;
            cell.find(".input").addClass("hide-in-slideshow")
        });
    </script>






Boost classifier by voting, i.e. combining weak classifiers
$$
\begin{aligned}
\begin{cases}
    x_1 \geq t_1, \\ 
    x_2 \geq t_2.
\end{cases}
\end{aligned}
$$
Will correctly classify all samples.





```python
np.random.seed(0)
t1 = np.random.rand(1).item()
t2 = np.random.rand(1).item()

plot_decision_tree([r'$x_1$', r'$x_2$'], [t1, t2])

ax = get_figure()
legends = plot_problem(ax, data, targets)
plot_classifier(legends, threshold=t1, feature_index=0)
plot_classifier(legends, threshold=t2, feature_index=1, color='mediumslateblue', text_color='navy')
```


![png](output_14_0.png)



![png](output_14_1.png)

