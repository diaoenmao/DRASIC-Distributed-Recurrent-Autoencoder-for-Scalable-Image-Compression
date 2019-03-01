import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NodeLSTMCell(nn.Module):
    def __init__(self, node_info):
        super(NodeLSTMCell, self).__init__()
        self.node_info = node_info
        self.iw,self.hw = self.make_node(node_info)
        
    def make_node(self, node_info):
        self.name = node_info['name']
        self.edge_names = node_info['edge_names']
        self.input_channels = node_info['input_channels']
        self.output_channels = node_info['output_channels']
        self.bias = node_info['bias']
        iw = nn.Linear(self.input_channels,4*self.output_channels,self.bias)
        hw = nn.ModuleDict({})
        for i in range(len(self.edge_names)):
            hw[self.edge_names[i]] = nn.Linear(self.output_channels,4*self.output_channels,self.bias)
        return iw, hw
        
    def forward(self, input, hidden):
        wi = self.iw(input).chunk(4, 1)
        wh = {}
        wh_sum_in = 0
        wh_sum_cell = 0
        wh_sum_out = 0
        for k in self.edge_names:
            wh[k] = self.hw[k](hidden[k][0]).chunk(4, 1)
            wh_sum_in = wh_sum_in + wh[k][0]
            wh_sum_cell = wh_sum_cell + wh[k][2]
            wh_sum_out = wh_sum_out + wh[k][3]
        in_gate = torch.sigmoid(wi[0] + wh_sum_in)        
        forget_gate = {}
        for k in self.edge_names:
            forget_gate[k] = torch.sigmoid(wi[1] + wh[k][1])
        cell_in = torch.tanh(wi[2] + wh_sum_cell)
        out_gate = torch.sigmoid(wi[3] + wh_sum_out)    
        cell_state =  (in_gate * cell_in)
        for k in self.edge_names:
            cell_state = cell_state + forget_gate[k]*hidden[k][1]
        hidden_state = out_gate * torch.tanh(cell_state)
        return hidden_state, cell_state

class GraphLSTMCell(nn.Module):
    def __init__(self, graph_info):
        super(GraphLSTMCell, self).__init__()
        self.graph_info = graph_info
        self.nodes = self.make_graph(self.graph_info)

    def make_graph(self,graph_info):
        nodes = nn.ModuleDict({})
        for k in graph_info:
            nodes[k] = NodeLSTMCell(graph_info[k]['node_info'])
        return nodes
        
    def forward(self, input, hidden):
        output = {}
        new_hidden = {}
        for k in input:
            new_hidden[k] = self.nodes[k](input[k],hidden)
            output[k] = new_hidden[k][0]
        return output, new_hidden
            
            
class GraphLSTM(nn.Module):
    def __init__(self,graph_info,dropout=0):
        super(GraphLSTM, self).__init__()
        self.dropout = dropout
        self.graph_info = graph_info
        self.graph = self.make_graph(self.graph_info)

    def make_graph(self,graph_info):
        self.num_layers = len(self.graph_info)
        graph = nn.ModuleList([])
        for g in range(self.num_layers):
            graph.append(GraphLSTMCell(graph_info[g]))
        return graph
        
    def forward(self, input, hidden, protocol):
        output = {k:[] for k in input}
        num_seq = protocol['num_seq']
        for i in range(num_seq):
            x = {k:input[k][:,i,] for k in input}
            for g in range(self.num_layers):
                x, hidden[g] = self.graph[g](x, hidden[g])
                if(self.dropout != 0 and g < (len(self.graph)-1)):
                    for k in x:
                        x[k] = F.dropout(x[k],p=self.dropout)
            for k in x:
                output[k].append(x[k])
        for k in output:
            output[k] = torch.stack(output[k],1)
        return output, hidden            
            
            