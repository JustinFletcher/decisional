

import argparse
import uuid

import requests
from random import shuffle
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def get_random_dag(num_verticies=10, edge_probability=0.33):
    G = nx.gnp_random_graph(n=num_verticies,
                            p=edge_probability,
                            directed=True)

    print("Number of nodes input: %d" % (G.number_of_nodes()))

    digraph_edges = list()
    for (u, v) in G.edges():
        if u < v:
            edge = (u, v, {'weight': np.random.uniform(0.0, 1.0)})
            digraph_edges.append(edge)
    dag = nx.DiGraph(digraph_edges)

    print("Number of nodes output: %d" % (dag.number_of_nodes()))
    return(dag)

def assign_at_random(a, b):
    """
    Pairs one element of a with at least one element of b at random.

    :param a: a list, each element of which will be paired with at least one
              element of b
    :param b: a list, each element of which will be paired with an element of
              a
    :return: a shuffled list of (a_element, b_element) tuples.
    """

    assignments_list = list()

    # Shuffle b to remove any order dependency.
    shuffle(b)

    # Iterate over each element of b.
    for b_element_index, b_element in enumerate(b):

        # Pair the first (len(a) - 1) elements of a to b.
        if b_element_index < (len(a) - 1):
            assignments_list.append((a[b_element_index], b_element))

        # Then pair the element of b with a random element of a.
        else:
            assignments_list.append((np.random.choice(a), b_element))

    return(assignments_list)

def generate_datatype(observed=False):

    datatype = dict()

    # ...beginning with UUID.
    datatype["datatype_uuid"] = uuid.uuid4()

    datatype_size = np.random.randint(2 ** 6, 2 ** 10)
    datatype["datatype_size"] = datatype_size

    if observed:
        datatype_size = np.random.uniform(0.001, 1000)
        datatype["datatype_generation_hz"] = datatype_size

    return (datatype)

def assign_random_edge_datatypes(application_dag):

    # This index will be incremented as datatypes are defined.
    datatype_index = 0

    # Iterate over each node, each of which represent a dataflow operator...
    for operator_index in application_dag.nodes:

        # ...and build and assign a list of datatypes.
        operator_datatypes = list()

        # Choose a random number of datatypes, <= the outdegree of this node.
        operator_outdegree = application_dag.out_degree(operator_index)
        if operator_outdegree > 1:
            num_operator_output_types = np.random.randint(1, operator_outdegree)
        else:
            num_operator_output_types = operator_outdegree

        # For each of these datatypes, make random datatype attributes.

        for operator_output_type in range(num_operator_output_types):

            datatype = generate_datatype()
            operator_datatypes.append(datatype)

        # So long as this operator has at least one datatype..
        if operator_datatypes:

            # ...assign operator datatypes at random to operator outedges...
            outedges = list(application_dag.out_edges(operator_index))
            datatype_edge_pairs = assign_at_random(operator_datatypes,
                                                   outedges)

            # ...and then apply this attribute to the edges.
            for (datatype, outedge) in datatype_edge_pairs:

                attrs = {outedge: {"datatype": datatype}}
                nx.set_edge_attributes(application_dag, attrs)

    return

def get_sample_dataflow_application_graph(num_operators=8):

    application_dag = get_random_dag(num_verticies=num_operators)
    assign_random_edge_datatypes(application_dag)
    # print(application_dag[0][1]["datatype"])

    # Add metadata for each operator.
    application_dag_attributes = dict()

    for operator_index in application_dag.nodes:

        operator_attributes = dict()

        # Randomly assign a compute requirement for each operator.
        operator_compute = np.random.randint(2**6, 2**10)
        operator_attributes["operator_compute"] = operator_compute

        # Randomly assign a scalability order to each operator.
        scalability_order = np.random.randint(1, 2)
        operator_attributes["scalability_order"] = scalability_order

        # Select some in-degree zero operators to be perceptors.
        in_degree = 0
        for edge in application_dag.edges:

            if operator_index == edge[1]:
                in_degree += 1

        if in_degree == 0:
            operator_attributes['perceptor'] = True

            observation_datatype = generate_datatype(observed=True)
            operator_attributes['observation_datatype'] = observation_datatype

        application_dag_attributes[operator_index] = operator_attributes


        # Select some out-degree zero operators to be deciders.
        if len(application_dag[operator_index]) == 0:

            operator_attributes['decider'] = True

            decision_datatype = generate_datatype()
            operator_attributes['decision_datatype'] = decision_datatype


    nx.set_node_attributes(application_dag, application_dag_attributes)


    return(application_dag)


def hill_eq(x, alpha, beta):

    if x <= 0.0:

        return 0.0

    else:

        return 1 / (1 + ((alpha/x)**beta))


def one_minus_hill_eq(x, alpha, beta):

    return (1 - hill_eq(x, alpha, beta))


def weighted_hill_surface(x, y, alpha, beta):

    # z = x + y
    z = x + np.log(y)
    return z


def timeliness(x, alpha, beta):

    return(hill_eq(x, alpha, beta))


def informedness(x, y):

    # return np.random.randn() * x + np.random.randn() * y
    # Recency alpha = number of seconds after which half of the computable information is computed
    # Recency beta = rate at which computation improves informativeness.
    # compute_value = hill_eq(x, alpha=60.0, beta=8.0)
    # # Recency alpha = number of seconds after which a predicate is half as infomative
    # # Recency beta = rate at which informativenss decays
    # recency_value = one_minus_hill_eq(y, alpha=30.0, beta=2.0)

    compute_value = hill_eq(x, alpha=10.0, beta=1.0/8.0)
    # Recency alpha = number of seconds after which a predicate is half as infomative
    # Recency beta = rate at which informativenss decays
    recency_value = one_minus_hill_eq(y, alpha=3.0, beta=1.0/4.0)

    informedness_value = compute_value * recency_value
    # informedness_value = (compute_value + recency_value) / 2

    return(informedness_value)



class WorldModel(object):

    def __init__(self, dataflow_application_model):

        self.datatypes = list()

        self.build_world_model_from_application_graph(dataflow_application_model.application_dag)


    def build_world_model_from_application_graph(self, application_dag):

        """
        This function parses a NetworkX DAG into a world model, which is represented and
        returned as a list of datatpye dictionaries. Some DAG nodes should have 'perceptor'
        and 'decider' attributes, as these will be used to construct the world model.
        :param application_dag: a NetworkX DAG representing a dataflow applications.
        :return: a list of datatype dictionaries representing a world model.
        """

        # First, catalog the datatypes built into the application graph.
        # TODO: refactor to add objects that solve this.
        datatypes = list()

        # Check each out-edge of each node...
        for node in application_dag.nodes:

            for out_edge in application_dag[node].values():

                # ...and if it hasn't been stored, store it.
                out_edge_datatype_stored = False

                for datatype in datatypes:

                    if out_edge['datatype']['datatype_uuid'] == datatype['datatype_uuid']:

                        out_edge_datatype_stored = True

                if not(out_edge_datatype_stored):

                    datatypes.append(out_edge['datatype'])

        # TODO: Generate datatypes at node construction time. Parse them here.
        # Now, for each perceptor node, generate and catalog a datatype.
        for observation_datatype in nx.get_node_attributes(application_dag, "observation_datatype").values():

            datatypes.append(observation_datatype)

        # Finally, for each decider node, generate and catalog a datatype.
        for decision_datatype in nx.get_node_attributes(application_dag, "decision_datatype").values():

            datatypes.append(decision_datatype)

        for datatype in datatypes:

            self.datatypes.append(datatype)

class DataflowApplicationModel(object):

    def __init__(self):

        self.application_dag = get_sample_dataflow_application_graph()

    def plot(self):
        # print(nx.get_node_attributes(application_dag, "scalability_order"))
        # print(application_dag.number_of_nodes())
        # nx.draw(application_dag, pos=pos)

        pos = self.decisional_layout(self.application_dag)
        connectionstyle = "arc3,rad=0.4"
        nx.draw_networkx_nodes(self.application_dag, pos)
        nx.draw_networkx_edges(self.application_dag,
                               pos,
                               connectionstyle=connectionstyle)
        plt.savefig('decisional_application_graph.png')
        plt.close()

    def decisional_layout(self, G):

        max_path_lens = list()

        for target_node in G.nodes:

            paths_to_target_node = list()

            for source_node in G.nodes:

                for path in nx.all_simple_paths(G,
                                                source=source_node,
                                                target=target_node):
                    paths_to_target_node.append(path)

            path_lengths_to_target_node = list()

            for path_to_target_node in paths_to_target_node:
                path_lengths_to_target_node.append(len(path_to_target_node))

            if path_lengths_to_target_node:

                max_path_length_to_target_node = max(path_lengths_to_target_node)

            else:

                max_path_length_to_target_node = 0

            max_path_lens.append(max_path_length_to_target_node)

            G.nodes[target_node]['max_path_len'] = max_path_length_to_target_node

            G.nodes[target_node]['subset'] = max_path_length_to_target_node

        pos = dict()

        x_coord = 0.0
        y_coord = 0.0

        for node in sorted(G.nodes(data=True), key=lambda x: x[1]['max_path_len']):
            x_coord += 1.0
            y_coord += 4.0

            pos[node[0]] = np.array([x_coord, y_coord])

        # for index, (node_key, position_array) in enumerate(pos.items()):
        #
        #     modified_position_array = position_array
        #
        #     modified_position_array[0] =  G.nodes[node_key]['max_path_len']
        #     modified_position_array[1] =  G.nodes[node_key]['max_path_len']
        #
        #     pos[node_key] = modified_position_array

        return (pos)


class NetworkModel(object):

    def __init__(self, num_verticies=128, edge_probability=0.01):

        # TODO: Replace with graph generation that models real networks.
        # Generate a sparse random graph.
        G = nx.gnp_random_graph(n=num_verticies,
                                p=edge_probability,
                                directed=False)

        # Prune all but the largest connected component.
        nx.connected_components(G)
        connected_components = list(nx.connected_components(G))
        component_sizes = [len(c) for c in connected_components]
        largest_component = connected_components[np.argmax(component_sizes)]
        G = nx.subgraph(G, largest_component)


        # Parse and apply bandwidth, latency, volatility, etc.
        annotated_graph = list()
        centralities = [nx.closeness_centrality(G, u=u) for u in G.nodes()]

        # TODO: scale bandwidth, volatility, and latency with edge centrality.
        for (u, v) in G.edges():
            u_centrality = nx.closeness_centrality(G, u=u)
            v_centrality = nx.closeness_centrality(G, u=v)
            edge = (u, v, {'bandwidth': np.random.uniform(0.0, 1.0),
                           'latency': np.random.uniform(0.0, 1.0),
                           'volatility': np.random.uniform(0.0, 1.0)})
            annotated_graph.append(edge)

        G = nx.Graph(annotated_graph)


        self.system_graph = G

    def plot(self):

        # pos = nx.kamada_kawai_layout(self.system_graph)
        pos = nx.spring_layout(self.system_graph)
        # pos = nx.circular_layout(self.system_graph)
        nx.draw_networkx_nodes(self.system_graph, pos=pos, node_size=10)
        nx.draw_networkx_edges(self.system_graph, pos=pos)
        plt.savefig('decisional_network_graph.png')

        return



def simulate_decisional_model(input):
    """

    :param input:
    :return:
    """
    # TODO: Dale check this out:
    # https://simgrid.org/publications.html

    # Generate a model of a dataflow application; it comprises operators and typed edges.
    dataflow_application_model = DataflowApplicationModel()
    dataflow_application_model.plot()

    # Generate a world model; it comprises Beliefs, some of which are Observations.
    world_model = WorldModel(dataflow_application_model)
    print(world_model.datatypes)

    # TODO: Generate network graph; it comprises Hosts and Links.
    network_model = NetworkModel()
    network_model.plot()


    # TODO: Generate world events.
    # TODO: Generate command generations.
    # TODO: Simulate model.


def cli_main(flags):

    input = flags

    simulate_decisional_model(input)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide arguments.')

    # Set arguments and their default values
    parser.add_argument('--service_name',
                        type=str,
                        default="http://localhost:5000",
                        help='A string encoding the target service.')

    parser.add_argument('--service_endpoint',
                        type=str,
                        default='test/endpoint',
                        help='The target endpoint.')

    parsed_flags, _ = parser.parse_known_args()

    cli_main(flags=parsed_flags)