import matplotlib.pyplot as plt
import networkx as nx
# Function to draw a neural network with formatted labels and indexing starting from 1
def draw_neural_network_with_labels_starting_from_1(input_neurons, hidden_neurons, output_neurons):
    G = nx.DiGraph()

    # Adding nodes for the input layer
    input_layer_positions = [(0, i) for i in range(input_neurons)]
    for i, pos in enumerate(input_layer_positions, 1):  # Start indexing from 1
        G.add_node(f'a$_{{{i}}}^1$', pos=pos)  # Layer 1: Input layer with subscript as neuron number

    # Adding nodes for the hidden layer
    hidden_layer_positions = [(1, i + (input_neurons - hidden_neurons) // 2) for i in range(hidden_neurons)]
    for i, pos in enumerate(hidden_layer_positions, 1):  # Start indexing from 1
        G.add_node(f'a$_{{{i}}}^2$', pos=pos)  # Layer 2: Hidden layer with subscript as neuron number

    # Adding nodes for the output layer
    output_layer_positions = [(2, (input_neurons - output_neurons) // 2)]
    for i, pos in enumerate(output_layer_positions, 1):  # Start indexing from 1
        G.add_node(f'a$_{{{i}}}^3$', pos=pos)  # Layer 3: Output layer with subscript as neuron number

    # Adding edges from input layer to hidden layer
    for i in range(1, input_neurons + 1):
        for h in range(1, hidden_neurons + 1):
            G.add_edge(f'a$_{{{i}}}^1$', f'a$_{{{h}}}^2$')

    # Adding edges from hidden layer to output layer
    for h in range(1, hidden_neurons + 1):
        for o in range(1, output_neurons + 1):
            G.add_edge(f'a$_{{{h}}}^2$', f'a$_{{{o}}}^3$')

    # Drawing the graph
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title('Neural Network with Layer and Neuron Labels (Indexing from 1)')
    plt.show()

# Draw the neural network with formatted labels and indexing starting from 1
draw_neural_network_with_labels_starting_from_1(20, 3, 1)
