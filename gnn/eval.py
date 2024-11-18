import dgl
import torch as th
import torch.nn.functional as F
import numpy as np

from graph.graph import GraphLoader
from gnn.wtagnn import WTAGNN

### print out the performance related numbers
def performance(pred, labels, acc=None):
    print('\n************model performance on {:d} test edges************'.format(len(pred)))
    tp = 0; fp = 0; tn = 0; fn = 0
    for i in range(len(labels)):
        tp += 1 if pred[i] == 1 and labels[i] == 1 else 0
        fp += 1 if pred[i] == 1 and labels[i] == 0 else 0
        tn += 1 if pred[i] == 0 and labels[i] == 0 else 0
        fn += 1 if pred[i] == 0 and labels[i] == 1 else 0
    print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)

    acc = float((tp + tn) / (tp + tn + fp + fn)) if acc == None else acc
    precision = float(tp / (tp + fp))
    recall = float(tp / (tp + fn))
    tnr = float(tn / (tn + fp))
    tpr = float(tp / (tp + fn))
    f1 = 2 * float(precision *  recall/ (precision + recall))

    print("accuracy {:.4f}".format(acc))
    print("precision {:.4f}".format(precision))
    print("recall {:.4f}".format(recall))
    print("tnr {:.4f}".format(tnr))
    print("tpr {:.4f}".format(tpr))
    print("f1 {:.4f}".format(f1))
    print('acc/pre/rec: ', str("{:.2f}".format(acc* 100) ) + '%/' + str("{:.2f}".format(precision* 100) ) + '%/' +
        str("{:.2f}".format(recall* 100) ) + '%')
    return precision, recall, tnr, tpr, f1


def evaluate(model, g, nf, ef, e_label, edge_val_mask):
    model.eval()
    with th.no_grad():
        # Pass g, nf, and ef to the model
        n_logits, e_logits = model(g, nf, ef)

        # Edge validation predictions and labels
        e_predictions = e_logits[edge_val_mask].argmax(dim=1).cpu()
        e_labels = e_label[edge_val_mask].cpu()

    # Compute accuracy for edge predictions
    acc = (e_predictions == e_labels).float().mean().item()
    return acc, e_predictions, e_labels


### load the pretrained and predict some of edges in the training graph
### for testing purpose only
def eval_saved_model(args):
    gloader = GraphLoader()
    if args.g_to_merge is not None: #load two graph and merge together
        g, nf, ef, e_label, train_mask, test_mask, val_mask = gloader.load_and_merge_graph(args)  
    else: # eval the model on the same graph used for training
        g, nf, ef, e_label, train_mask, test_mask, val_mask = gloader.load_graph(args)  

    n_classes = 2
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    # load the pre-trained model
    best_model = WTAGNN(g, input_node_feat_size, input_edge_feat_size, args.n_hidden, n_classes, args.n_layers,  F.relu, args.dropout)
    best_model.load_state_dict(th.load( './output/best.model.' + args.model_name))
    print('model load from: ./output/best.model.' + args.model_name )

    acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)
    precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)
    

### in the inductive evaluation, we first load the pre-trained model and the corresponding training graph
### then for each a test site, we find each edge's neighbor in the training graph
### we next apply the trained model to the testing graph and predict those edges that never seen before during training
def eval_model_inductive(args):
    gloader = GraphLoader()
    ### load the training graph
    train_g, train_g_nf, train_g_ef, train_g_e_label, _, _, _ = gloader.load_graph(args)  
    train_g_id_node_map, train_g_id_edge_map = gloader.load_node_edge_map(args)
    train_g.ndata['nf'], train_g.edata['ef'], train_g.edata['e_label'] = train_g_nf, train_g_ef, train_g_e_label

    ### load the full graph, where we have all testing edges from all testing sites
    ### note: you can indeen build a graph for each testing site and load it separately
    ### here I am loading the full graph for convinence
    ### since later I will extract edges/nodes for each testing site
    ### it works the same if you load edges/nodes separately for each of your testing site
    args.graph_name = 'full' #set it to full so that we can load the full graph; it was set to the train_g orginally
    full_g, full_g_nf, full_g_ef, full_g_e_label, _, _, _ = gloader.load_graph(args)  
    full_g_id_node_map, full_g_id_edge_map = gloader.load_node_edge_map(args)
    full_g.ndata['nf'], full_g.edata['ef'], full_g.edata['e_label'] = full_g_nf, full_g_ef, full_g_e_label

    train_g_node_id_map = {v: k for k, v in train_g_id_node_map.items()}
    full_g_node_id_map  = {v: k for k, v in full_g_id_node_map.items()}
    train_g_dst_eids_map = {}; train_g_src_eids_map = {} # key is node str, values are edge ids point to this node
    for e in train_g_id_edge_map:
        train_g_src_eids_map.setdefault(e['src'], []).append(e['id'])
        train_g_dst_eids_map.setdefault(e['dst'], []).append(e['id'])

    # parameters
    n_classes = 2
    input_node_feat_size = train_g.ndata['nf'].shape[1]
    input_edge_feat_size = train_g.edata['ef'].shape[1]

    # load testing edges and corresponding testing sites
    test_sites_map = {} # key is the id of testing site, value is a list of edges of this testing site
    train_g_eid_set = set(e['id'] for e in train_g_id_edge_map)
    # those edges not exist in train_g are our testing edges in the inductive setting
    raw_test_eids = [e['id'] for e in full_g_id_edge_map if e['id'] not in train_g_eid_set] 
    for raw_test_eid in raw_test_eids:
        site_id = full_g_id_edge_map[raw_test_eid]['site_id'] # get the site id of this unseen (testing) edge
        test_sites_map.setdefault(site_id, []).append(raw_test_eid)

    print('edges # in general', len(full_g_id_edge_map), 'sites in general', len(set(e['site_id'] for e in full_g_id_edge_map)))
    print('edges # used for training', len(train_g_id_edge_map), 'sites for training', len(set(e['site_id'] for e in train_g_id_edge_map)))
    print('edges # for testing', len(raw_test_eids), 'sites for testing', len(test_sites_map), '\n')

    results = []
    ct = 1
    for site_id in test_sites_map:
        print('evaluating on testing site:',site_id, 
              '  total testing edge in this site:', len(test_sites_map[site_id]))

        node_id_map = {}; idx = 0; node_set = set()
        src_ids = []; dst_ids = []; edge_feats = []; edge_labels = []
        # here we extract the testing site's edges and nodes from the full graph
        for eid in test_sites_map[site_id]:
            src = full_g_id_edge_map[eid]['src']
            dst = full_g_id_edge_map[eid]['dst']
            node_set.add(src); node_set.add(dst)
            if src not in node_id_map:
                node_id_map[src] = idx
                idx += 1
            if dst not in node_id_map:
                node_id_map[dst] = idx   
                idx += 1

            # prepare edge and its feature/label
            src_ids.append(node_id_map[src])
            dst_ids.append(node_id_map[dst])
            edge_feats.append(full_g.edata['ef'][eid])
            edge_labels.append(full_g.edata['e_label'][eid])

        #extract node feat
        node_feats = []
        for node in node_id_map:
            nid_in_full_g = full_g_node_id_map[node]
            node_feats.append(full_g.ndata['nf'][nid_in_full_g])
       
        g = dgl.DGLGraph()
        g.add_nodes(len(node_id_map))
        g.add_edges(src_ids, dst_ids)

        # assign nf, ef, and label
        g.ndata['nf'] = th.stack(node_feats)
        g.edata['ef'] = th.stack(edge_feats)
        g.edata['e_label'] = th.stack(edge_labels)

        # now we completed the extraction and construction of the testing site
        # again, you can pre-build your each of your testing site and load it directly (without above extraction) 
        # as noted before; currently all the edges of this g are unseen during training, and we want to predict them
        print('the graph of the testing site: ', g)

        # we are now have the graph of testing site in hand
        # we then find neighbors in train_g, and add to current testing graph
        nodes_list_in_cur_g = list(node_id_map.keys()) 
        new_edge_src_nids = []; new_edge_dst_nids = []; cur_nid = g.number_of_nodes(); 
        num_of_node_before_adding = g.number_of_nodes(); num_of_edge_before_adding = g.number_of_edges()
        new_node_feats = []; new_edge_feats = []; new_edge_labels = []
        for node in nodes_list_in_cur_g: # find neighbors for each node in g
            if node in train_g_dst_eids_map: # this node exist in train_g, so we have neighbors for this node, so we add edges
                ### adding nodes/edges once
                for eid in train_g_dst_eids_map[node]: #option 1: choose all neighbor
                # for eid in train_g_dst_eids_map[node][:10]: #option2: choose 10 neighbor
                    src_in_train_g = train_g_id_edge_map[eid]['src']
                    if src_in_train_g not in node_id_map: # new node, so add; Or already added
                        node_id_map[src_in_train_g] = cur_nid ### allocate this nid to the newly added (thou not happen yet)
                        new_node_feats.append( train_g.ndata['nf'][train_g_node_id_map[src_in_train_g]].clone().detach() )
                        cur_nid += 1
                    
                    new_edge_src_nids.append(node_id_map[src_in_train_g])
                    new_edge_dst_nids.append(node_id_map[node]) # the dst nid will always be the id of current node

                    new_edge_feats.append( train_g.edata['ef'][eid].clone().detach() )
                    new_edge_labels.append( train_g.edata['e_label'][eid] )
  
        if len(new_node_feats) > 0: # we have neighbor found/added
            g.add_nodes(len(new_node_feats))
            g.ndata['nf'][num_of_node_before_adding : g.number_of_nodes()] = th.stack(new_node_feats)

        if len(new_edge_src_nids) > 0: # we have neighbor edge found/added
            g.add_edges(new_edge_src_nids, new_edge_dst_nids)
            g.edata['ef'][num_of_edge_before_adding : g.number_of_edges()] = th.stack(new_edge_feats)
            g.edata['e_label'][num_of_edge_before_adding : g.number_of_edges()] = th.stack(new_edge_labels)

        ## with above code, we added neighbor nodes/edges (if any) to the testing graph
        ## however, we only want to predict the edges of current testing site, not those neighbors 
        ## So, below we use the test_mask to achieve this goal
        print('After adding neighbors, the graph: ', g, '\n')

        nf = g.ndata.pop('nf')
        ef = g.edata.pop('ef')
        e_label = g.edata['e_label']
        test_mask = np.zeros(e_label.shape[0]) 
        test_mask[0 : num_of_edge_before_adding] = 1  # our target are the edges of testing site only
        test_mask = th.BoolTensor(test_mask)

        # load the best model
        best_model = WTAGNN(g, input_node_feat_size, input_edge_feat_size, args.n_hidden, n_classes, 
                            args.n_layers,  F.relu, args.dropout)
        best_model.load_state_dict(th.load('./output/best.model.' + args.model_name))
        acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)

        ## here i am not printing the performace immediately as there might be ZeroDivisionError
        ## instead, i put predictions of each testing site into a list
        results += [(eid, edge_labels[idx].item(), predictions[idx].item()) for idx, eid in enumerate(test_sites_map[site_id])]
        # print('\n')
        # if ct == 15: break
        # ct+=1
        
    ### we completed predict edges in each testing site;
    ### meanwhile, we have raw predictions saved in the results list
    ### now let print the performance out
    labels      = [r[1] for r in results]
    predictions = [r[2] for r in results]
    precision, recall, tnr, tpr, f1 = performance(predictions, labels)

    