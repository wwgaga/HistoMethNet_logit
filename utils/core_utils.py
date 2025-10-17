import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, CLAM_MT
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from torchmetrics.regression import R2Score
from sklearn.metrics import r2_score
from models.rrt import RRTMIL
from models.rrt_logit import RRTMILLogit
from models.logit_only import LogitOnly
from utils.losses import (
    ConfidentialIntervalLoss,
    PiModelLoss,
    ProportionLoss,
    VATLoss,
    SupConLoss,
    VATLoss_CLAM,
    ConfidentialIntervalLoss_CLAM,
    consistency_loss_function,
)
from scipy.stats import pearsonr

# Add wandb import
import wandb

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    # Initialize wandb instead of SummaryWriter
    if args.log_data:
        wandb.init(project=f"Experiment_{args.exp_code}_Seed_{args.seed}", config=args)
        wandb.run.name = f"Fold_{cur}"
        writer = wandb  # Assign wandb to writer variable for consistency
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes).to(device)
        if device.type == 'cuda':
            loss_fn = loss_fn.to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)
    print('Done!')
    
    print('\nInit Model...', end=' ')
    if args.model_type in ['rrt', 'rrt_logit', 'logit_only']:
        model_dict = {
            'input_dim': args.embed_dim,
            'n_classes': args.n_classes,
            'dropout': args.drop_out,
            'act': args.act,
            'region_num': args.region_num,
            'pos': args.pos,
            'pos_pos': args.pos_pos,
            'pool': args.pool,
            'peg_k': args.peg_k,
            'drop_path': args.drop_path,
            'n_layers': args.n_trans_layers,
            'n_heads': args.n_heads,
            'attn': args.attn,
            'da_act': args.da_act,
            'trans_dropout': args.trans_drop_out,
            'ffn': args.ffn,
            'mlp_ratio': args.mlp_ratio,
            'trans_dim': args.trans_dim,
            'epeg': args.epeg,
            'min_region_num': args.min_region_num,
            'qkv_bias': args.qkv_bias,
            'epeg_k': args.epeg_k,
            'cell_property': args.cell_property,
            'epeg_2d': args.epeg_2d,
            'epeg_bias': args.epeg_bias,
            'epeg_type': args.epeg_type,
            'region_attn': args.region_attn,
            'peg_1d': args.peg_1d,
            'cr_msa': args.cr_msa,
            'crmsa_k': args.crmsa_k,
            'all_shortcut': args.all_shortcut,
            'crmsa_mlp':args.crmsa_mlp,
            'crmsa_heads':args.crmsa_heads,
         }
    else:
        model_dict = {"dropout": args.drop_out, 
                    'n_classes': args.n_classes, 
                    "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type not in ['mil', 'rrt', 'rrt_logit', 'logit_only']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb', 'clam_mt']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2).to(device)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.to(device)
        else:
            instance_loss_fn = nn.CrossEntropyLoss().to(device)
        
        # Define the loss functions
        stem_immune_criterion = nn.MSELoss().to(device)
        cell_type_criterion = nn.KLDivLoss(reduction='batchmean').to(device)
        # cell_type_criterion = nn.MSELoss()

        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mt':
            model = CLAM_MT(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    elif args.model_type == 'rrt':
        model = RRTMIL(**model_dict)
        instance_loss_fn = nn.CrossEntropyLoss().to(device)
        prop_criterion = VATLoss(xi=10.0, eps=1.0, ip=1)
        conf_interval_criterion = ConfidentialIntervalLoss()
        supcon_criterion = SupConLoss(temperature=0.07)
    elif args.model_type == 'rrt_logit':
        model = RRTMILLogit(**model_dict)
        instance_loss_fn = nn.CrossEntropyLoss().to(device)
        prop_criterion = VATLoss(xi=10.0, eps=1.0, ip=1)
        conf_interval_criterion = ConfidentialIntervalLoss()
        supcon_criterion = SupConLoss(temperature=0.07)
    elif args.model_type == 'logit_only':
        # Direct multinomial logistic regression over patch features
        model = LogitOnly(input_dim=args.embed_dim, n_classes=None, dropout=args.drop_out, cell_property=args.cell_property)
        instance_loss_fn = nn.CrossEntropyLoss().to(device)
        prop_criterion = VATLoss(xi=10.0, eps=1.0, ip=1)
        conf_interval_criterion = ConfidentialIntervalLoss()
        supcon_criterion = SupConLoss(temperature=0.07)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model = model.to(device)
    if args.model_type in ['clam_sb', 'clam_mb', 'clam_mt']:
        prop_criterion = VATLoss_CLAM(xi=10.0, eps=1.0, ip=1)
        conf_interval_criterion = ConfidentialIntervalLoss_CLAM()
    print('Done!')
    
    if args.model_type in ['rrt', 'rrt_logit', 'logit_only']:
        print('rrt is ready')
    else:
        print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)
    
    else:
        early_stopping = None
    best_auc = -1
    best_r_value = -1
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb', 'clam_mt'] and not args.no_inst_cluster:
             
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, prop_criterion, conf_interval_criterion)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        elif args.model_type in ['rrt', 'rrt_logit', 'logit_only']:
            
            train_loop_rrt(epoch, model, train_loader, optimizer, args.n_classes, \
                           args.bag_weight, writer, loss_fn, prop_criterion, \
                            conf_interval_criterion, supcon_criterion, cell_property=args.cell_property)
            stop = validate_rrt(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, \
                    cell_property=args.cell_property)
            
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if args.model_type in ['rrt', 'rrt_logit', 'logit_only']:
            _, val_error, _, _, val_r_value= summary_rrt(model, val_loader, args.n_classes, cell_property=args.cell_property)
        else:
            _, val_error, val_auc, _, _, _= summary(model, val_loader, args.n_classes)
        
        # if (val_auc+val_r_value)>= best_auc:  
        #     best_auc = (val_auc+val_r_value)
        #     torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_best.pt".format(cur)))
        
        
        if val_r_value >= best_r_value:  
            best_r_value = val_r_value
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_best_r.pt".format(cur)))
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint_best_r.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_best_r.pt".format(cur)))

    # _, val_error, val_auc, _, mean_r2_score_stem, mean_r2_score_immune, mean_r2_score_cell_type = summary(model, val_loader, args.n_classes)
    if args.model_type in ['rrt', 'rrt_logit', 'logit_only']:
        _, val_error, _, mae_value, r_value = summary_rrt(model, val_loader, args.n_classes, cell_property=args.cell_property)
    else:
        _, val_error, val_auc, _, mae_value, r_value = summary(model, val_loader, args.n_classes)

    print('Val error: {:.4f}, Pearson r score cell_type: {:.4f}, MAE score cell type: {:.4f}'.format(val_error, r_value, mae_value))

    if args.model_type in ['rrt', 'rrt_logit', 'logit_only']:
        results_dict, test_error, acc_logger, mae_value, r_value = summary_rrt(model, test_loader, args.n_classes, cell_property=args.cell_property)
    else:
        results_dict, test_error, test_auc, acc_logger, mae_value, r_value = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, Pearson r score cell_type: {:.4f}, MAE score cell type: {:.4f}'.format(test_error, r_value, mae_value))

    # for i in range(args.n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    #     if writer:
    #         # Replace SummaryWriter.add_scalar with wandb.log
    #         wandb.log({f'final/test_class_{i}_acc': acc})
    
    if writer:
        # Replace SummaryWriter.add_scalar with wandb.log
        wandb.log({
            'final/val_error': val_error,
            'final/val_auc': 0.0,
            'final/test_error': test_error,
            'final/test_auc': 0.0,
            'final/pearson_r_cell_type': r_value,
            'final/mae_cell_type': mae_value,
        })
        writer.finish()  # Ensure wandb run is properly closed
        test_auc = 0.0 
        val_auc = 0.0 
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, r_value, mae_value


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, prop_criterion = None, conf_interval_criterion = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    train_consistency_loss = 0.
    train_conf_interval_loss = 0.

    print('\n')
    for batch_idx, data in enumerate(loader):
        
        data, label, stem, immune, cell_type_percentages, min_max_dict = data[0].to(device), \
        data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]    
        # print(data.shape)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        # print(stem_like)        
        # print(immune)
        
        # stem_immune_prob = instance_dict['stem_immune_prob'] 
        cell_type_prob = instance_dict['cell_type_prob']
       
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        
        cell_type_percentages_min = min_max_dict[0]['cell_type_percentages_min']
        cell_type_percentages_max = min_max_dict[0]['cell_type_percentages_max']
        if prop_criterion and conf_interval_criterion:
            consistency_cell_type_loss = consistency_loss_function(
                prop_criterion, 
                model, 
                loader, 
                data, 
                epoch, 
                task_name='cell_type'
            )
        
        consistency_loss_value = consistency_cell_type_loss.item()
        loss_value = loss.item()
        conf_interval_loss = conf_interval_criterion(
            cell_type_prob,
            cell_type_percentages,
            cell_type_percentages_min.to(device),
            cell_type_percentages_max.to(device),
            )
        
        conf_interval_loss_value = conf_interval_loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        train_consistency_loss += consistency_loss_value
        train_conf_interval_loss += conf_interval_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss + (1-bag_weight) * conf_interval_loss + (1-bag_weight) * consistency_loss_value

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += (loss_value + consistency_loss_value + conf_interval_loss_value)
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_consistency_loss /= len(loader)
    train_conf_interval_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss: \
           {:.4f}, train_conf_interval_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,  train_inst_loss,  train_conf_interval_loss, train_consistency_loss, train_error))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            # Replace SummaryWriter.add_scalar with wandb.log
            wandb.log({f'train/class_{i}_acc': acc})

    if writer:
        # Replace SummaryWriter.add_scalar with wandb.log
        wandb.log({
            'train/loss': train_loss,
            'train/error': train_error,
            'train/clustering_loss': train_inst_loss,
            'train/conf_interval_loss': train_conf_interval_loss,
            'train/consistency_loss': train_consistency_loss,
        })



def calculate_prop(output, nb, bs):
    output = F.softmax(output, dim=1)
    output = output.reshape(nb, bs, -1)
    lp_pred = output.mean(dim=1)
    return lp_pred


def train_loop_rrt(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, prop_criterion = None, conf_interval_criterion=None, supcon_criterion=None, cell_property='cell_type'):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    train_consistency_loss = 0.
    train_conf_interval_loss = 0.
    train_contrastive_loss = 0.
    print('\n')
    for batch_idx, data in enumerate(loader):
        
        data, label, stem, immune, eight_cell_type, min_max_dict = data[0].to(device), \
        data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]
        
        if cell_property == 'cell_type': 
            cell_type_percentages = eight_cell_type
        elif cell_property == 'stem': 
            cell_type_percentages = stem
        elif cell_property == 'immune': 
            cell_type_percentages = immune

        # print(cell_type_percentages.shape)
        
        instance_dict = model(data)
        # stem_prob = instance_dict['stem_prob'] 
        # immune_prob = instance_dict['immune_prob']
        cell_type_prob = instance_dict['cell_type_prob']
        # print(cell_type_prob.shape)
        # uncomment if doing multiple regression tasks        
        # stem_min = min_max_dict['stem_min']
        # stem_max = min_max_dict['stem_max']
        # immune_min = min_max_dict['immune_min']
        # immune_max = min_max_dict['immune_max']
        features = instance_dict['features']
        cell_type_logits = instance_dict['cell_type_logits']

        cell_type_percentages_min = min_max_dict[0]['{}_min'.format(cell_property)]
        cell_type_percentages_max = min_max_dict[0]['{}_max'.format(cell_property)]

    
    
    
        # uncomment if doing multiple regression tasks        
        # consistency_stem_loss = consistency_loss_function (
        #     prop_criterion, 
        #     loader, 
        #     data, 
        #     epoch, 
        #     task_name='stem'
        # )
        # consistency_immune_loss = consistency_loss_function (
        #     prop_criterion, 
        #     loader, 
        #     data, 
        #     epoch, 
        #     task_name='immune'
        # )
        consistency_cell_type_loss = consistency_loss_function(
            prop_criterion, 
            model, 
            loader, 
            data, 
            epoch, 
            task_name=cell_property
        )
        # print(cell_type_prob.shape)
        # print(cell_type_percentages.shape)
        # print(cell_type_percentages_min.shape)
        conf_interval_loss = conf_interval_criterion(
            cell_type_prob,
            cell_type_percentages,
            cell_type_percentages_min.to(device),
            cell_type_percentages_max.to(device),
            )
        consistency_loss_value = consistency_cell_type_loss.item()
        contrastive_loss = supcon_criterion(features, cell_type_logits)

        conf_interval_loss_value = conf_interval_loss.item()

        contrastive_value = contrastive_loss.item()
        # contrastive_value = 0.0

        train_consistency_loss += consistency_loss_value
        train_conf_interval_loss += conf_interval_loss_value
        train_contrastive_loss += contrastive_value

        train_loss += (consistency_loss_value + conf_interval_loss_value + contrastive_value)

        total_loss =  50 * consistency_cell_type_loss + 5 * conf_interval_loss + 2 * contrastive_loss
        
        
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

     # calculate loss and error for epoch
    train_loss /= len(loader)
    train_consistency_loss /= len(loader)
    train_conf_interval_loss /= len(loader)
    train_contrastive_loss /= len(loader)
    train_error /= len(loader)
    
    print('Epoch: {}, train_loss: {:.4f}, train_conf_interval_loss: {:.4f}, train_consistency_loss: {:.4f},  train_error: {:.4f}'.format(epoch, train_loss,  train_conf_interval_loss, train_consistency_loss, train_error))
    
    if writer:
        # Replace SummaryWriter.add_scalar with wandb.log
        wandb.log({
            'train/loss': train_loss,
            'train/error': train_error,
            'train/conf_interval_loss': train_conf_interval_loss,
            'train/consistency_loss': train_consistency_loss,
            'train/contrastive_loss': train_contrastive_loss,
        })


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            # Replace SummaryWriter.add_scalar with wandb.log
            wandb.log({f'train/class_{i}_acc': acc})

    if writer:
        # Replace SummaryWriter.add_scalar with wandb.log
        wandb.log({
            'train/loss': train_loss,
            'train/error': train_error,
        })

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        # Replace SummaryWriter.add_scalar with wandb.log
        wandb.log({
            'val/loss': val_loss,
            'val/auc': auc,
            'val/error': val_error,
        })

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


# Function to compute Pearson's r for each output
def compute_pearsonr(preds, targets, labels):
    pearsonr_values = []
    for i, label in enumerate(labels):
        r_value, _ = pearsonr(preds[:, i], targets[:, i])
        pearsonr_values.append(r_value)
        print(f"Pearson's r for {label}: {r_value:.4f}")
    return pearsonr_values

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    r2_score_stem_immune_list = []
    real_stem_ratios = []
    pred_stem_ratios = []

    real_immune_ratios = []
    pred_immune_ratios = []

    real_cell_ratios = []
    pred_cell_ratios = []

    r2_score_cell_type_list = []
    r2 = R2Score().to(device)
    with torch.inference_mode():
        for batch_idx, data in enumerate(loader):
            # data, label = data[0].to(device), data[1].to(device)   
            data, label, stem, immune, cell_type_percentages, min_max_dict = data[0].to(device), \
            data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]    
            # print(data.shape)

            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            # stem_immune_prob = instance_dict['stem_immune_prob'] 
            cell_type_prob = instance_dict['cell_type_prob']
            
            # stem_prob = stem_immune_prob[0][0].detach().cpu().numpy()
            # pred_stem_ratios.append(stem_prob)
            # print(stem_prob)
            # print(stem_immune)
            # stem_ratio = stem_immune[0].detach().cpu().numpy()
            # real_stem_ratios.append(stem_ratio)
            
            # immune_prob = stem_immune_prob[0][1].detach().cpu().numpy()
            # immune_label = stem_immune[1].detach().cpu().numpy()

            # pred_immune_ratios.append(immune_prob)
            # real_immune_ratios.append(immune_label)

            pred_cell_type_prob = cell_type_prob.detach().cpu().numpy()
            
            pred_cell_ratios.append(pred_cell_type_prob)
            real_cell_ratios.append(cell_type_percentages.squeeze().detach().cpu().numpy())
            
            # print(stem_immune_prob)
            # print(stem_immune)

            # r2_score_stem_immune = r2(stem_immune_prob.squeeze(), stem_immune.squeeze())
            # r2_score_stem_immune_list.append(r2_score_stem_immune.detach().cpu().numpy())
            
            # r2_score_cell_type = r2(cell_type_prob.squeeze(), cell_type_percentages.squeeze())
            # r2_score_cell_type_list.append(r2_score_cell_type.detach().cpu().numpy())
            # print('r2 score for stem and immune is : {}'.format(r2_score_stem_immune))
            # print('r2 score for cell types is : {}'.format(r2_score_cell_type))


            loss = loss_fn(logits, label)
           
            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    
    # mean_r2_score_stem = r2_score(np.array(real_stem_ratios), np.array(pred_stem_ratios))
 
    # mean_r2_score_immune = r2_score(np.array(real_immune_ratios), np.array(pred_immune_ratios))

    mean_r2_score_stem = 0.0
 
    mean_r2_score_immune = 0.0
    
    # Compute MAE for other cell types
    real_cell_ratios = np.vstack(real_cell_ratios).flatten()
    pred_cell_ratios = np.vstack(pred_cell_ratios).flatten()
   
    mae_value = torch.abs(torch.tensor(real_cell_ratios) - torch.tensor(pred_cell_ratios)).mean(dim=0)
    print("MAE for cell types propotions:", mae_value)
    
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute Pearson's r
    r_value, p_value = pearsonr(pred_cell_ratios, real_cell_ratios)
    
    print(f"Overall Pearson's r across all cell types and samples: {r_value:.4f}")


    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, pearson r: {:.4f}, MAE Score: {:.4f}'.format(val_loss, val_error, auc, r_value, mae_value))
    

    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        # Replace SummaryWriter.add_scalar with wandb.log
        wandb.log({
            'val/loss': val_loss,
            'val/auc': auc,
            'val/error': val_error,
            'val/inst_loss': val_inst_loss,
            'val/mae_value': mae_value, #MAE for cell types propotions
            'val/pearson_r': r_value, #Pearson's r across all cell types and samples    
        })


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            # Replace SummaryWriter.add_scalar with wandb.log
            wandb.log({f'val/class_{i}_acc': acc})
     
    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_rrt(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None, cell_property='cell_type'):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
# sample_size = model.k_sample
    r2_score_stem_immune_list = []
    real_stem_ratios = []
    pred_stem_ratios = []

    real_immune_ratios = []
    pred_immune_ratios = []

    real_cell_ratios = []
    pred_cell_ratios = []

    r2_score_cell_type_list = []
    r2 = R2Score().to(device)
    with torch.inference_mode():
        for batch_idx, data in enumerate(loader):
            # data, label = data[0].to(device), data[1].to(device)   
            data, label, stem, immune, eight_cell_type = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)       
            instance_dict = model(data) 


            if cell_property == 'cell_type': 
                cell_type_percentages = eight_cell_type
            elif cell_property == 'stem': 
                cell_type_percentages = stem
            elif cell_property == 'immune': 
                cell_type_percentages = immune


            cell_type_prob = instance_dict['cell_type_prob']
            
            # stem_prob = stem_prob.detach().cpu().numpy()
            # pred_stem_ratios.append(stem_prob)
            # # print(stem_prob)
            # # print(stem_immune)
            # stem_label = stem.detach().cpu().numpy()
            # real_stem_ratios.append(stem_label)
            
            # immune_prob = immune_prob.detach().cpu().numpy()
            # immune_label = immune.detach().cpu().numpy()

            # pred_immune_ratios.append(immune_prob)
            # real_immune_ratios.append(immune_label)

            pred_cell_type_prob = cell_type_prob.detach().cpu().numpy()
            
            pred_cell_ratios.append(pred_cell_type_prob)
            real_cell_ratios.append(cell_type_percentages.squeeze().detach().cpu().numpy())
            # print(pred_cell_type_prob)
            # print(cell_type_percentages.detach().cpu().numpy())
            # print(stem_immune_prob)
            # print(stem_immune)
            

            # r2_score_stem_immune = r2(stem_immune_prob.squeeze(), stem_immune.squeeze())
            # r2_score_stem_immune_list.append(r2_score_stem_immune.detach().cpu().numpy())
            
            # r2_score_cell_type = r2(cell_type_prob.squeeze(), cell_type_percentages.squeeze())
            # r2_score_cell_type_list.append(r2_score_cell_type.detach().cpu().numpy())
            # print('r2 score for stem and immune is : {}'.format(r2_score_stem_immune))
            # print('r2 score for cell types is : {}'.format(r2_score_cell_type))


           
            

            # instance_loss = instance_dict['instance_loss']
            
            # inst_count+=1
            # instance_loss_value = instance_loss.item()
            # val_inst_loss += instance_loss_value

            # inst_preds = instance_dict['inst_preds']
            # inst_labels = instance_dict['inst_labels']
            # inst_logger.log_batch(inst_preds, inst_labels)

          
            
      

    val_error /= len(loader)
    val_loss /= len(loader)

    
    
    # mean_r2_score_stem = r2_score(np.array(real_stem_ratios), np.array(pred_stem_ratios))
 
    # mean_r2_score_immune = r2_score(np.array(real_immune_ratios), np.array(pred_immune_ratios))
    mean_r2_score_stem = 0.0
    mean_r2_score_immune = 0.0
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute MAE for other cell types
    real_cell_ratios = np.vstack(real_cell_ratios).flatten()
    pred_cell_ratios = np.vstack(pred_cell_ratios).flatten()
    
    mae_value = torch.abs(torch.tensor(real_cell_ratios) - torch.tensor(pred_cell_ratios)).mean(dim=0).item()
    print("MAE for cell types propotions:", mae_value)
    
    # Compute Pearson's r
    r_value, p_value = pearsonr(pred_cell_ratios, real_cell_ratios)
    
    print(f"Overall Pearson's r across all cell types and samples: {r_value:.4f}")

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, pearson r: {:.4f}, MAE Score: {:.4f}'.format(val_loss, val_error, r_value, mae_value))

    if writer:
        # Replace SummaryWriter.add_scalar with wandb.log
        wandb.log({
            'val/loss': val_loss,
            'val/error': val_error,
            # 'val/inst_loss': val_inst_loss,
            'val/mae_value': mae_value, #MAE for cell types propotions
            'val/pearson_r': r_value, #Pearson's r across all cell types and samples    
        })



    # for i in range(n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
    #     if writer and acc is not None:
    #         # Replace SummaryWriter.add_scalar with wandb.log
    #         wandb.log({f'val/class_{i}_acc': acc})
     
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint_best.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    r2_score_stem_immune_list = []
    r2_score_cell_type_list = []

    real_stem_ratios = []
    pred_stem_ratios = []

    real_immune_ratios = []
    pred_immune_ratios = []

    real_cell_ratios = []
    pred_cell_ratios = []

    r2 = R2Score().to(device)
    for batch_idx, data in enumerate(loader):
        data, label, stem, immune, cell_type_percentages, min_max_dict = data[0].to(device), \
        data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]    
        # print(data.shape)
        if cell_property == 'cell_type': 
            cell_type_percentages = eight_cell_type
        elif cell_property == 'stem': 
            cell_type_percentages = stem
        elif cell_property == 'immune': 
            cell_type_percentages = immune
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            # logits, instance_dict = model(data)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        
        # stem_immune_prob = instance_dict['stem_immune_prob'] 
        cell_type_prob = instance_dict['cell_type_prob']

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        
        # stem_prob = stem_immune_prob[0][0].detach().cpu().numpy()
        # pred_stem_ratios.append(stem_prob)
        # # print(stem_prob)
        # # print(stem_immune)
        # stem_ratio = stem_immune[0].detach().cpu().numpy()
        # real_stem_ratios.append(stem_ratio)
        
        # immune_prob = stem_immune_prob[0][1].detach().cpu().numpy()
        # immune_label = stem_immune[1].detach().cpu().numpy()

        # pred_immune_ratios.append(immune_prob)
        # real_immune_ratios.append(immune_label)

        pred_cell_type_prob = cell_type_prob.detach().cpu().numpy()
        pred_cell_ratios.append(pred_cell_type_prob)
        real_cell_ratios.append(cell_type_percentages.detach().cpu().numpy())
        
        all_probs[batch_idx] =  probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        # error = calculate_error(Y_hat, label)
        # test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    # mean_r2_score_stem_immune = np.mean(np.array(r2_score_stem_immune_list))
    # mean_r2_score_cell_type = np.mean(np.array(r2_score_cell_type_list))
    # print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, r2 score stem immune: {:.4f}, r2 score cell_type: {:.4f}'.format(val_loss, val_error, auc, mean_r2_score_stem_immune, mean_r2_score_cell_type))
    # mean_r2_score_stem = r2_score(np.array(real_stem_ratios), np.array(pred_stem_ratios))
 
    # mean_r2_score_immune = r2_score(np.array(real_immune_ratios), np.array(pred_immune_ratios))
    mean_r2_score_stem = 0.0
    mean_r2_score_immune = 0.0
   
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute MAE for other cell types
    real_cell_ratios = np.vstack(real_cell_ratios).flatten()
    pred_cell_ratios = np.vstack(pred_cell_ratios).flatten()
   
    mae_value = torch.abs(torch.tensor(real_cell_ratios) - torch.tensor(pred_cell_ratios)).mean(dim=0).item()
    print("MAE for cell types propotions:", mae_value)
    
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute Pearson's r
    r_value, p_value = pearsonr(pred_cell_ratios, real_cell_ratios)
    
    # print(f"Overall Pearson's r across all cell types and samples: {r_value:.4f}")
    

    # if writer:
    #     # Replace SummaryWriter.add_scalar with wandb.log
    #     wandb.log({
    #         'test_error': test_error,
    #         'test_auc': auc,
    #         'test_mae': mae_value.tolist(),
    #         'test_pearson_r': r_value,
    #         'test_r2_cell_type': mean_r2_score_cell_type
    #     })

    return patient_results, test_error, auc, acc_logger, mae_value, r_value


def summary_rrt(model, loader, n_classes, cell_property='cell_type'):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    r2_score_stem_immune_list = []
    r2_score_cell_type_list = []

    real_stem_ratios = []
    pred_stem_ratios = []

    real_immune_ratios = []
    pred_immune_ratios = []

    real_cell_ratios = []
    pred_cell_ratios = []

    r2 = R2Score().to(device)
    # print('length of loader')
    # print(len(loader))
    for batch_idx, data in enumerate(loader):
        # print(batch_idx)
        data, label, stem, immune, eight_cell_type, min_max_dict = data[0].to(device), \
        data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]    
        
        if cell_property == 'cell_type': 
            cell_type_percentages = eight_cell_type
        elif cell_property == 'stem': 
            cell_type_percentages = stem
        elif cell_property == 'immune': 
            cell_type_percentages = immune
        # print(data.shape)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            instance_dict = model(data)
            # logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        
        # stem_immune_prob = instance_dict['stem_immune_prob'] 
        cell_type_prob = instance_dict['cell_type_prob']
        
        
        # stem_prob = stem_immune_prob[0][0].detach().cpu().numpy()
        # pred_stem_ratios.append(stem_prob)
        # # print(stem_prob)
        # # print(stem_immune)
        # stem_ratio = stem_immune[0].detach().cpu().numpy()
        # real_stem_ratios.append(stem_ratio)
        
        # immune_prob = stem_immune_prob[0][1].detach().cpu().numpy()
        # immune_label = stem_immune[1].detach().cpu().numpy()

        # pred_immune_ratios.append(immune_prob)
        # real_immune_ratios.append(immune_label)

        pred_cell_type_prob = cell_type_prob.detach().cpu().numpy()
        pred_cell_ratios.append(pred_cell_type_prob)
        real_cell_ratios.append(cell_type_percentages.detach().cpu().numpy())
        
        all_labels[batch_idx] = label.item()
        probs = 0.0
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

    test_error /= len(loader)

    # mean_r2_score_cell_type = np.mean(np.array(r2_score_cell_type_list))
    # print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, r2 score stem immune: {:.4f}, r2 score cell_type: {:.4f}'.format(val_loss, val_error, auc, mean_r2_score_stem_immune, mean_r2_score_cell_type))
    # mean_r2_score_stem = r2_score(np.array(real_stem_ratios), np.array(pred_stem_ratios))
 
    # mean_r2_score_immune = r2_score(np.array(real_immune_ratios), np.array(pred_immune_ratios))
    mean_r2_score_stem = 0.0
    mean_r2_score_immune = 0.0
   
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute MAE for other cell types
    real_cell_ratios = np.vstack(real_cell_ratios).flatten()
    pred_cell_ratios = np.vstack(pred_cell_ratios).flatten()
   
    mae_value = torch.abs(torch.tensor(real_cell_ratios) - torch.tensor(pred_cell_ratios)).mean(dim=0).item()
    # print("MAE for cell types propotions:", mae_value)
    
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute Pearson's r
    r_value, p_value = pearsonr(pred_cell_ratios, real_cell_ratios)
    
    # print(f"Overall Pearson's r across all cell types and samples: {r_value:.4f}")
    

    # if writer:
    #     # Replace SummaryWriter.add_scalar with wandb.log
    #     wandb.log({
    #         'test_error': test_error,
    #         'test_auc': auc,
    #         'test_mae': mae_value.tolist(),
    #         'test_pearson_r': r_value,
    #         'test_r2_cell_type': mean_r2_score_cell_type
    #     })

    return patient_results, test_error,  acc_logger, mae_value, r_value
