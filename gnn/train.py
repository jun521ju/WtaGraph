import time
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from gnn.eval import performance, evaluate
from gnn.wtagnn import WTAGNN  # Ensure this imports your modified WTAGNN
from graph.graph import GraphLoader

def start_train(args):
    th.autograd.set_detect_anomaly(True)
    gloader = GraphLoader()
    g, nf, ef, e_label, edge_train_mask, edge_test_mask, edge_val_mask = gloader.load_graph(args)

    # Create node masks
    num_nodes = g.num_nodes()
    node_train_mask = th.zeros(num_nodes, dtype=th.bool)
    node_val_mask = th.zeros(num_nodes, dtype=th.bool)
    node_test_mask = th.zeros(num_nodes, dtype=th.bool)

    # For simplicity, use all nodes for training
    node_train_mask[:] = True  # Example: All nodes are part of the training set
    node_val_mask[:] = True
    node_test_mask[:] = True

    # Verify feature sizes
    print('Node feature size:', nf.shape)
    print('Edge feature size:', ef.shape)

    n_classes = 2
    input_node_feat_size = nf.shape[1]
    input_edge_feat_size = ef.shape[1]

    print('\n************initialize model************')
    # Create the model with multi-head attention for both nodes and edges
    model = WTAGNN(g, input_node_feat_size, input_edge_feat_size,
                   args.n_hidden, n_classes, args.n_layers, args.n_heads, F.relu, args.dropout)
    print(model)

    # Apply CUDA if GPU is specified
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        print('using gpu:', args.gpu)
        th.cuda.empty_cache()
        th.cuda.set_device(args.gpu)
        nf, ef, e_label = nf.cuda(), ef.cuda(), e_label.cuda()
        edge_train_mask, edge_val_mask, edge_test_mask = edge_train_mask.cuda(), edge_val_mask.cuda(), edge_test_mask.cuda()
        node_train_mask, node_val_mask, node_test_mask = node_train_mask.cuda(), node_val_mask.cuda(), node_test_mask.cuda()
        model.cuda()

    # Loss function and optimizer
    loss_fcn = th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Start training
    print('\n************start training************')
    dur, max_acc = [], -1
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        # Forward pass with both node and edge features
        n_logits, e_logits = model(g, nf, ef)

        # Compute the loss for nodes and edges
        node_loss = loss_fcn(n_logits[node_train_mask],
                             th.zeros_like(n_logits[node_train_mask][:, 0].long()))  # Example: Dummy labels
        edge_loss = loss_fcn(e_logits[edge_train_mask], e_label[edge_train_mask])
        loss = node_loss + edge_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # Safely calculate average duration
        avg_dur = np.mean(dur) if dur else 0.0

        # Evaluate the model on validation set
        acc, predictions, labels = evaluate(model, g, nf, ef, e_label, edge_val_mask)

        # Save the best model
        if acc > max_acc:
            max_acc = acc
            th.save(model.state_dict(), './output/best.model.' + args.model_name)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, avg_dur, loss.item(),
                                            acc, g.number_of_edges() / avg_dur / 1000 if avg_dur > 0 else 0))

    # Load the best model for testing
    best_model = WTAGNN(g, input_node_feat_size, input_edge_feat_size, args.n_hidden, n_classes,
                        args.n_layers, args.n_heads, F.relu, args.dropout)
    if cuda:
        best_model.cuda()
    best_model.load_state_dict(th.load('./output/best.model.' + args.model_name))

    # Test the model on the test set
    acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, edge_test_mask)
    precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)


def start_train_cv(args):
    gloader = GraphLoader()
    g, nf, ef, e_label, _, _, _ = gloader.load_graph(args)

    n_classes = 2
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        print('using gpu:', args.gpu)
        th.cuda.empty_cache()
        th.cuda.set_device(args.gpu)
        nf, ef = nf.cuda(), ef.cuda()

    print('\n************start training for {:d} folds************'.format(args.fold))
    kf = StratifiedKFold(n_splits=args.fold, shuffle=True)
    kf.get_n_splits()
    print(kf)
    fold = 0
    total_precision = total_acc = total_recall = 0
    for train_index, test_index in kf.split(e_label, e_label):
        fold += 1
        print('\nfold #: ', str(fold))
        train_mask = np.zeros(e_label.shape[0])
        train_mask[train_index] = 1
        train_mask = th.BoolTensor(train_mask)

        test_mask = np.zeros(e_label.shape[0])
        test_mask[test_index] = 1
        test_mask = th.BoolTensor(test_mask)

        # Create model with multi-head attention
        model = WTAGNN(g, input_node_feat_size, input_edge_feat_size, args.n_hidden,
                       n_classes, args.n_layers, args.n_heads, F.relu, args.dropout)
        print(model)

        # Apply CUDA if GPU is used
        if cuda:
            e_label = e_label.cuda()
            train_mask = train_mask.cuda()
            test_mask = test_mask.cuda()
            model.cuda()

        loss_fcn = th.nn.CrossEntropyLoss()
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        dur = [];
        max_acc = -1
        for epoch in range(args.n_epochs):
            model.train()
            if epoch >= 3: t0 = time.time()

            # Forward pass with multi-head attention for both nodes and edges
            n_logits, e_logits = model(g, nf, ef)

            # Compute the loss on the training edges and nodes
            node_loss = loss_fcn(n_logits[train_mask], e_label[train_mask])
            edge_loss = loss_fcn(e_logits[train_mask], e_label[train_mask]) if e_logits is not None else 0
            loss = node_loss + edge_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc, predictions, labels = evaluate(model, g, nf, ef, e_label, test_mask)
            # Save the best model
            if acc > max_acc:
                max_acc = acc
                th.save(model.state_dict(), './output/best.model.' + args.model_name + '.fold.' + str(fold))

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                acc, g.number_of_edges() / np.mean(dur) / 1000))

        # Load the best model for testing on this fold
        best_model = WTAGNN(g, input_node_feat_size, input_edge_feat_size, args.n_hidden, n_classes,
                            args.n_layers, args.n_heads, F.relu, args.dropout)
        if cuda:
            best_model.cuda()
        best_model.load_state_dict(th.load('./output/best.model.' + args.model_name + '.fold.' + str(fold)))

        acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)

        precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)

        total_precision += precision
        total_acc += acc
        total_recall += recall

    print('\n************training done! Averaged model performance************')
    print('acc/pre/rec: ', str("{:.2f}".format(total_acc / args.fold * 100)) + '%/ '
          + str("{:.2f}".format(total_precision / args.fold * 100)) + '%/ ' +
          str("{:.2f}".format(total_recall / args.fold * 100)) + '%')
