import argparse
import uuid

parser = argparse.ArgumentParser(description='Trainer Template of Graph Neural Network with Pooling')
parser.add_argument('--model', type=str, default='MVRPool', help='Model name:MVRPool')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    help='Dataset name, PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES /PTC_MR/PTC_FM/IMDB-BINARY/IMDB-MULTI'
                         'MUTAG / COLLAB / PTC_FR / PTC_MM')

parser.add_argument('--run_name', dest="name", type=str, default='test_' + str(uuid.uuid4())[:8],
                    help='Name of the run')
parser.add_argument('--notes', type=str, default='None', help='Notes/Other Params')

parser.add_argument('--gpu', type=str, default='0', help='Cuda devices')
parser.add_argument('--folds', type=int, default=10, help='Cross validation folds')
parser.add_argument('--restore', action='store_true', help='Model restoring')
parser.add_argument('--replication', action='store_false', help='Replication with different random seeds')
parser.add_argument('--seed', type=int, default=3653, help='Random seed')
parser.add_argument('--replication_num', type=int, default=1, help='Repeat Times (Number of Random seeds)')
parser.add_argument('--channel_fusion', type=str, default="Cat", help='Aggregation of dual channels, Sum/Cat')

parser.add_argument('--epochs', dest="max_epochs", type=int, default=300, help='Max epochs')
parser.add_argument('--patience', type=int, default=100, help='Early stopping')
parser.add_argument('--num_layers', type=int, default=3, help='Number of GCN-Pooling blocks')

parser.add_argument('--dropout_ratio', type=float, default=0.4, help='Dropout ratio')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--hid_dim', type=int, default=64, help='Hidden embedding size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='Pooling ratio')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--jump_connection', type=str, default="Cat", help='Aggregation of jump connections, '
                                                                       'None/Sum/Cat/Mean')
parser.add_argument('--data_normalization', type=str, default="Mean", help='None/MaxMin/Mean')
parser.add_argument('--noise', type=str, default="gaussian_noise",
                    help='none/gumble/gaussian_noise/random_zero/sign_flip')

parser.add_argument('--l2', type=float, default=5e-4, help='L2 regularization')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--lr_decay_step', type=int, default=50, help='Decay step of lr scheduler')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Decay factor of lr scheduler')

parser.add_argument('--normalization', type=str, default="Layer", help='Layer/Graph')

parser.add_argument('--perturb', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--feature_fusion', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--score_fusion', type=lambda x: (str(x).lower() == 'true'), default=True)

parser.add_argument('--original', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Include original graph view')

parser.add_argument('--fully_connected', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Include fully connected graph view')

parser.add_argument('--spec_loss_coef', type=float, default=1)
parser.add_argument('--gg_loss_coef', type=float, default=-1)
parser.add_argument('--gl_loss_coef', type=float, default=-1)

parser.add_argument('--order', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--order_num', type=int, default=5)
parser.add_argument('--order_norm', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--order_threshold', type=float, default=0.3)
parser.add_argument('--include_self_loops', type=lambda x: (str(x).lower() == 'true'), default=False)

parser.add_argument('--acyclic', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Include acyclic view (spanning tree)')
parser.add_argument('--acyclic_method', type=str, default='spanning_tree',
                    choices=['spanning_tree', 'dfs_removal'],
                    help='Method to build acyclic graph')

parser.add_argument('--spectral_embedding', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--spectral_embedding_k', type=int, default=1)
parser.add_argument('--spectral_embedding_include_original', type=lambda x: (str(x).lower() == 'true'), default=True)

parser.add_argument('--curvature_based', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Include curvature_based view')
parser.add_argument('--curvature_type', type=str, default='balanced_forman',
                    help='Method to build acyclic graph')
parser.add_argument('--dynamic_update_fraction', type=float, default=0.5)
parser.add_argument('--use_long_range', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--use_curvature_max', type=lambda x: (str(x).lower() == 'true'), default=True)

args, unknown = parser.parse_known_args()
