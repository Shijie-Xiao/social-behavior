'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle
import random
import time

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import DataLoader
from utils_mice import MouseMotionData
from st_graph import ST_GRAPH
from model import SRNN
from criterion import Gaussian2DLikelihood


def main():
    parser = argparse.ArgumentParser()

    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=2,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=5,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=20,
                        help='Sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for the optimizer')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')

    # Distributed training arguments
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')

    # Mouse dataset
    parser.add_argument('--use_mouse_data', action='store_true',
                        help='Use mouse dataset instead of human trajectory dataset')
    parser.add_argument('--mouse_data_path', type=str, default='../data/mice/user_train_r1.npy',
                        help='Path to mouse data (only if --use_mouse_data)')
    parser.add_argument('--frame_subsample', type=int, default=4,
                        help='Take every Nth frame for mouse data; 1=no subsampling')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Fraction of sequences for validation (mouse data)')
    parser.add_argument('--exp_tag', type=str, default='',
                        help='Experiment tag for unique save path; auto from obs_pred_fs if empty')

    args = parser.parse_args()

    # Initialize distributed training
    # Priority: environment variables (set by torchrun/torch.distributed.launch) > command line argument
    # Check for LOCAL_RANK in environment (set by both torchrun and torch.distributed.launch)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        # If RANK and WORLD_SIZE are set (torchrun), use them
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            # torch.distributed.launch: assume single node, local_rank = rank
            rank = local_rank
            world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    elif args.local_rank >= 0:
        # Manual specification via command line
        local_rank = args.local_rank
        rank = local_rank
        world_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        # Single GPU mode
        rank = -1
        world_size = 1
        local_rank = -1

    if rank >= 0:
        # Initialize distributed process group
        # torch.distributed.launch/torchrun will set MASTER_ADDR and MASTER_PORT if needed
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        args.local_rank = local_rank
        args.rank = rank
        args.world_size = world_size
        # Print distributed training info for verification
        print(f'[Rank {rank}] Distributed training initialized - World Size: {world_size}, Local Rank: {local_rank}, GPU: cuda:{local_rank}')
    else:
        args.local_rank = -1
        args.rank = 0
        args.world_size = 1
        print('[Rank 0] Single GPU training mode')

    train(args)


def train(args):
    if args.use_mouse_data:
        if not os.path.isabs(args.mouse_data_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mouse_data_path = os.path.join(script_dir, args.mouse_data_path)
        else:
            mouse_data_path = args.mouse_data_path
        if args.rank <= 0:
            print("Using mouse dataset from: {}".format(mouse_data_path))
            print("5 keypoints per mouse -> 15 nodes; frame_subsample={}".format(args.frame_subsample))
        dataloader = MouseMotionData(
            fpath=mouse_data_path,
            batch_size=args.batch_size,
            seq_length=args.seq_length + 1,
            forcePreProcess=True,
            infer=False,
            frame_subsample=args.frame_subsample,
            val_fraction=args.val_fraction
        )
        obs_len = max(args.seq_length - args.pred_length, 0)
        exp_tag = args.exp_tag or "obs{}_pred{}_fs{}".format(obs_len, args.pred_length, args.frame_subsample)
        log_directory = os.path.join('log', 'mice', exp_tag, 'log_attention')
        save_directory = os.path.join('save', 'mice', exp_tag, 'save_attention')
    else:
        datasets = [i for i in range(5)]
        datasets.remove(args.leaveDataset)
        dataloader = DataLoader(args.batch_size, args.seq_length + 1, datasets, forcePreProcess=True)
        log_directory = os.path.join('log', str(args.leaveDataset), 'log_attention')
        save_directory = os.path.join('save', str(args.leaveDataset), 'save_attention')

    stgraph = ST_GRAPH(1, args.seq_length + 1)

    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(save_directory, exist_ok=True)
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Open the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar')

    # Initialize net
    net = SRNN(args)
    if args.local_rank >= 0:
        net = net.cuda(args.local_rank)
        net = DDP(net, device_ids=[args.local_rank], find_unused_parameters=True)
        print(f'[Rank {args.rank}] Model moved to GPU {args.local_rank}, wrapped with DDP')
    else:
        net.cuda()
        print('[Rank 0] Model moved to GPU 0 (single GPU mode)')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)

    learning_rate = args.learning_rate
    if args.rank == 0:
        print('Training begin')
        print(f'Distributed training: {args.world_size} GPUs, Using DDP: {args.local_rank >= 0}')
    else:
        print(f'[Rank {args.rank}] Process started on GPU {args.local_rank}')
    best_val_loss = 100
    best_epoch = 0

    # Training
    for epoch in range(args.num_epochs):
        # Set random seed for data synchronization across processes
        if args.world_size > 1:
            torch.manual_seed(epoch)
            random.seed(epoch)
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            # Skip batches not assigned to this process
            if batch % args.world_size != args.rank:
                # Still need to advance the batch pointer to keep data in sync
                dataloader.next_batch(randomUpdate=True)
                continue

            start = time.time()

            # Get batch data
            x, _, _, d = dataloader.next_batch(randomUpdate=True)
            
            # Debug: Print which batch each rank processes (only first few batches)
            if epoch == 0 and batch < 5 and args.world_size > 1:
                print(f'[Rank {args.rank}] Processing batch {batch} on GPU {args.local_rank}')

            # Loss for this batch
            loss_batch = 0

            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                # Construct the graph for the current sequence
                stgraph.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                # Convert to cuda variables
                device = f'cuda:{args.local_rank}' if args.local_rank >= 0 else 'cuda'
                nodes = Variable(torch.from_numpy(nodes).float()).to(device)
                edges = Variable(torch.from_numpy(edges).float()).to(device)

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).to(device)
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).to(device)

                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).to(device)
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).to(device)

                # Zero out the gradients
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                loss_batch += loss.item()

                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

                # Reset the stgraph
                stgraph.reset()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            if args.rank == 0:
                print(
                    '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                        args.num_epochs * dataloader.num_batches,
                                                                                        epoch,
                                                                                        loss_batch, end - start))

        # Compute loss for the entire epoch and count actual batches processed
        num_batches_processed = 0
        for batch in range(dataloader.num_batches):
            if batch % args.world_size == args.rank:
                num_batches_processed += 1
        
        if num_batches_processed > 0:
            loss_epoch /= num_batches_processed

        # Aggregate loss across all processes
        if args.world_size > 1:
            device = f'cuda:{args.local_rank}' if args.local_rank >= 0 else 'cuda'
            loss_sum_tensor = torch.tensor(loss_epoch * num_batches_processed, device=device)
            count_tensor = torch.tensor(num_batches_processed, device=device, dtype=torch.float)
            
            # Sum both loss sum and batch count across all processes
            dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            
            # Divide by total number of batches processed across all processes
            total_batches = count_tensor.item()
            if total_batches > 0:
                loss_epoch = loss_sum_tensor.item() / total_batches
            else:
                loss_epoch = 0
            
            # Print batch distribution info for first epoch (only rank 0)
            if epoch == 0 and args.rank == 0:
                print(f'Training: Total batches={dataloader.num_batches}, Processed per rank ~{total_batches//args.world_size}, Total processed={int(total_batches)}')
        # Log it (only rank 0)
        if args.rank == 0:
            log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')

        # Validation
        if args.world_size > 1:
            torch.manual_seed(epoch + 10000)  # Different seed for validation
            random.seed(epoch + 10000)
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch = 0

        # Process validation batches using round-robin assignment
        # Note: When valid_num_batches < world_size, some ranks may not process any batches
        # This is normal and handled correctly in the aggregation step
        for batch in range(dataloader.valid_num_batches):
            # Round-robin assignment: batch % world_size determines which rank processes it
            assigned_rank = batch % args.world_size
            
            # Skip batches not assigned to this process
            if assigned_rank != args.rank:
                # Still need to advance the batch pointer to keep data in sync
                dataloader.next_valid_batch(randomUpdate=False)
                continue
            
            # Get batch data for batches assigned to this rank
            x, _, d = dataloader.next_valid_batch(randomUpdate=False)

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                stgraph.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                # Convert to cuda variables
                device = f'cuda:{args.local_rank}' if args.local_rank >= 0 else 'cuda'
                nodes = Variable(torch.from_numpy(nodes).float()).to(device)
                edges = Variable(torch.from_numpy(edges).float()).to(device)

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).to(device)
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).to(device)
                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).to(device)
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).to(device)

                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                                             hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                             cell_states_node_RNNs, cell_states_edge_RNNs)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)

                loss_batch += loss.item()

                # Reset the stgraph
                stgraph.reset()

            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

        # Compute loss for the entire epoch and count actual batches processed
        # Count batches this rank actually processed (using same round-robin logic)
        num_batches_processed = 0
        for batch in range(dataloader.valid_num_batches):
            assigned_rank = batch % args.world_size
            if assigned_rank == args.rank:
                num_batches_processed += 1
        
        if num_batches_processed > 0:
            loss_epoch /= num_batches_processed
        else:
            # If this rank processed no batches, set loss to 0 for aggregation
            loss_epoch = 0.0

        # Aggregate validation loss across all processes
        if args.world_size > 1:
            device = f'cuda:{args.local_rank}' if args.local_rank >= 0 else 'cuda'
            # Send sum of losses (loss_epoch * num_batches_processed) and count separately
            loss_sum_tensor = torch.tensor(loss_epoch * num_batches_processed, device=device)
            count_tensor = torch.tensor(num_batches_processed, device=device, dtype=torch.float)
            
            # Sum both loss sum and batch count across all processes
            dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            
            # Divide by total number of batches processed across all processes
            total_batches = count_tensor.item()
            if total_batches > 0:
                loss_epoch = loss_sum_tensor.item() / total_batches
            else:
                loss_epoch = 0.0
            
            # Print validation batch distribution info for first epoch (only rank 0)
            if epoch == 0 and args.rank == 0:
                print(f'Validation: Total batches={dataloader.valid_num_batches}, Total processed={int(total_batches)} '
                      f'(Expected: {dataloader.valid_num_batches})')
                if dataloader.valid_num_batches < args.world_size:
                    print(f'Warning: valid_num_batches({dataloader.valid_num_batches}) < world_size({args.world_size}), '
                          f'some ranks may not process validation batches (this is normal)')

        # Update best validation loss until now (synchronized across all processes)
        if args.rank == 0:
            if loss_epoch < best_val_loss:
                best_val_loss = loss_epoch
                best_epoch = epoch

            # Record best epoch and best validation loss
            print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
            print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
            # Log it
            log_file_curve.write(str(loss_epoch)+'\n')

            # Save the model after each epoch
            print('Saving model')
            model_state_dict = net.module.state_dict() if args.local_rank >= 0 else net.state_dict()
            torch.save({
                'epoch': epoch,
                'state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch))
        else:
            # Non-rank 0 processes still need to update best_val_loss for consistency
            if loss_epoch < best_val_loss:
                best_val_loss = loss_epoch
                best_epoch = epoch

    # Record the best epoch and best validation loss overall (only rank 0)
    if args.rank == 0:
        print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
        # Log it
        log_file.write(str(best_epoch)+','+str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()

    # Clean up distributed training
    if args.local_rank >= 0:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
