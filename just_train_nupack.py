from comet_ml import Experiment
import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
import math
from sklearn.utils import shuffle
from argparse import ArgumentParser
from utils import *
import time
import warnings
import os
import sys
import tqdm
import torch.optim.lr_scheduler as lr_scheduler


try: # we don't always install these on every platform
    from nupack import *
except:
    pass

backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error

def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action = 'store_true')
    group.add_argument('--no-' + name, dest=name, action = 'store_false')
    parser.set_defaults(**{name:default})

def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser.add_argument("--run_num", type=int, default=0, help="Experiment ID")
    parser.add_argument("--comet_project", default='just train nupack', type=str)
    parser.add_argument("--comet_tags", type=str,default='10k_series')
    parser.add_argument("--model_seed",type=int,default=0,help="if we are using a toy dataset, it may take a specific seed")
    parser.add_argument("--dataset_seed",type=int,default=0,help="if we are using a toy dataset, it may take a specific seed")
    parser.add_argument("--oracle_seed",type=int,default=0,help="if we are using a toy dataset, it may take a specific seed")
    parser.add_argument("--device", default="cuda", type=str, help="'cuda' or 'cpu'")

    # oracle -- only used when building a new dataset
    parser.add_argument("--dataset_size",type=int,default=int(2e4),help="number of items in the initial (toy) dataset")
    parser.add_argument("--dataset_dict_size",type=int,default=4,help="number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4 - with variable length, 0's are added for padding")
    parser.add_argument("--oracle", type=str, default="nupack energy")  # 'linear' 'potts' 'nupack energy' 'nupack pairs' 'nupack pins'
    parser.add_argument("--dataset_type",type=str,default="toy",help="Toy oracle is very fast to sample",)
    add_bool_arg(parser,"dataset_variable_length",default=True)
    parser.add_argument("--min_sample_length", type=int, default=10)
    parser.add_argument("--max_sample_length", type=int, default=60)

    # Proxy model
    parser.add_argument("--proxy_model_type",type=str,default="transformer",  help="type of proxy model - mlp or transformer")
    parser.add_argument("--training_parallelism",action="store_true",default=False,help="fast enough on GPU without paralellism - True doesn't always work on linux")
    parser.add_argument("--proxy_model_ensemble_size",type=int,default=1,help="number of models in the ensemble")
    parser.add_argument("--proxy_model_embedding_width", type=int, default=256) # depth of transformer embedding
    parser.add_argument("--proxy_model_width",type=int,default=256,help="number of neurons per proxy NN layer")
    parser.add_argument("--proxy_model_layers",type=int,default=8,help="number of layers in NN proxy models (transformer encoder layers OR MLP layers)")
    parser.add_argument("--proxy_training_batch_size", type=int, default=100000)
    parser.add_argument("--proxy_max_epochs", type=int, default=2000)
    add_bool_arg(parser, 'proxy_shuffle_dataset', default=True)
    add_bool_arg(parser, 'proxy_clip_max', default=False)
    parser.add_argument("--proxy_dropout_prob", type=float,default=0) #[0,1) dropout probability on fc layers
    parser.add_argument("--proxy_attention_dropout_prob", type=float,default=0) #[0,1) dropout probability on attention layers
    parser.add_argument("--proxy_norm", type=str,default='batch') # None, 'batch'
    parser.add_argument("--proxy_attention_norm", type = str, default = 'layer')
    parser.add_argument("--proxy_aggregation", type=str, default = 'sum')
    parser.add_argument("--proxy_init_lr", type=float, default = 1e-3)
    parser.add_argument("--proxy_history", type=int, default = 500)

    return parser

def process_config(config):
    # Normalize seeds
    config.model_seed = config.model_seed % 10
    config.dataset_seed = config.dataset_seed % 10
    config.toy_oracle_seed = config.oracle_seed % 10

    return config


class nupackModel():
    def __init__(self, config, ensembleIndex, mean, std, comet = None):
        self.mean = mean
        self.std = std
        self.config = config
        self.ensembleIndex = ensembleIndex
        self.config.history = min(config.proxy_history, self.config.proxy_max_epochs) # length of past to check
        self.comet = comet
        self.initModel()
        torch.random.manual_seed(int(config.model_seed + ensembleIndex))

        self.scheduler1 = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=min(self.config.history // 2,20),
            threshold = 1e-3,
            threshold_mode = 'rel',
            cooldown=self.config.history // 2
        )

    def get_training_batch_size(self):
        finished = 0
        training_batch_0 = 1 * self.config.proxy_training_batch_size
        #  test various batch sizes to see what we can store in memory
        datasetBuilder = buildDataset(self.config)
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad=True, lr=self.config.proxy_init_lr)  # optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#

        while (self.config.proxy_training_batch_size > 1) & (finished == 0):
            try:
                print('Trying batch size {}'.format(self.config.proxy_training_batch_size))
                test_dataset = []
                for i in range(len(datasetBuilder)):  # test data is drawn from oldest datapoints
                    test_dataset.append(datasetBuilder[i])
                test_dataloader = data.DataLoader(test_dataset, batch_size=self.config.proxy_training_batch_size, shuffle=False, num_workers=0, pin_memory=True)
                self.model.train(True)
                for i, trainData in enumerate(test_dataloader):
                    proxy_loss = self.getLoss(trainData)

                    self.optimizer.zero_grad()  # run the optimizer
                    proxy_loss.backward()
                    self.optimizer.step()
                    finished = 1
            except RuntimeError:  # if we get an OOM, try again with smaller batch
                self.config.proxy_training_batch_size = int(np.ceil(self.config.proxy_training_batch_size * 0.8)) - 1

        final_batch_size = max(int(self.config.proxy_training_batch_size * 0.8), 1)
        print('Final batch size is {}'.format(final_batch_size))
        return final_batch_size


    def initModel(self):
        '''
        Initialize model and optimizer
        :return:
        '''
        if self.config.proxy_model_type == 'transformer': # switch to variable-length sequence model
            self.model = transformer(self.config)
        elif self.config.proxy_model_type == 'mlp':
            self.model = MLP(self.config)
        else:
            print(self.config.proxy.model_type + ' is not one of the available models')

        if self.config.device == 'cuda':
            self.model = self.model.cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)



    def converge(self, dataset, returnHist = False):
        '''
        train model until test loss converges
        :return:
        '''
        self.config.proxy_training_batch_size = self.get_training_batch_size()
        self.initModel() # reset model

        [self.err_tr_hist, self.err_te_hist] = [[], []] # initialize error records

        tr, te, self.datasetSize = getDataloaders(self.config, self.ensembleIndex, dataset)

        #printRecord(f"Dataset size is: {bcolors.OKCYAN}%d{bcolors.ENDC}" %self.datasetSize)

        self.converged = 0 # convergence flag
        self.epochs = 0

        while (self.converged != 1):
            t0 = time.time()
            if self.epochs > 0: #  this allows us to keep the previous model if it is better than any produced on this run
                self.train_net(tr)
            else:
                self.err_tr_hist.append(torch.zeros(1).to(self.config.device)[0])

            self.test(te) # baseline from any prior training
            tf = time.time()
            # after training at least 10 epochs, check convergence
            if self.epochs >= self.config.history:
                self.checkConvergence()

            if True:#(self.epochs % 10 == 0):
                printRecord("Model {} epoch {} train loss {:.3f} test loss {:.3f} took {} seconds".format(self.ensembleIndex, self.epochs, self.err_tr_hist[-1], self.err_te_hist[-1], int(tf-t0)))

            self.epochs += 1

        if returnHist:
            return torch.stack(self.err_tr_hist).cpu().detach().numpy(), torch.stack(self.err_te_hist).cpu().detach().numpy()


    def checkConvergence(self):
        """
        check if we are converged
        condition: test loss has increased or levelled out over the last several epochs
        :return: convergence flag
        """
        # check if test loss is increasing for at least several consecutive epochs
        eps = 1e-4 # relative measure for constancy

        if all(torch.stack(self.err_te_hist[-self.config.history+1:])  > self.err_te_hist[-self.config.history]): #
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs - test loss increasing at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)

        # check if train loss is unchanging
        lr0 = np.copy(self.optimizer.param_groups[0]['lr'])
        self.scheduler1.step(torch.mean(torch.stack(self.err_tr_hist[1:])))  # plateau scheduler, skip first epoch
        lr1 = np.copy(self.optimizer.param_groups[0]['lr'])
        if lr1 != lr0:
            print('Learning rate reduced on plateau from {} to {}'.format(lr0, lr1))

        if abs(self.err_tr_hist[-self.config.history] - torch.mean(torch.stack(self.err_tr_hist[-self.config.history:])))/self.err_tr_hist[-self.config.history] < eps:
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs - hit train loss convergence criterion at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)

        # check if we run out of epochs
        if self.epochs >= self.config.proxy_max_epochs:
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs- epoch limit was hit with test loss {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)


    def train_net(self, tr):
        '''
        perform one epoch of training
        :param tr: training set dataloader
        :return: n/a
        '''
        err_tr = []
        self.model.train(True)
        for i, trainData in enumerate(tr):
            proxy_loss = self.getLoss(trainData)
            err_tr.append(proxy_loss.data)  # record the loss

            self.optimizer.zero_grad()  # run the optimizer
            proxy_loss.backward()
            self.optimizer.step()

        self.err_tr_hist.append(torch.mean(torch.stack(err_tr)))

        if self.comet:
            self.comet.log_metric('train loss', epoch = self.epochs, value=self.err_tr_hist[-1].cpu().detach().numpy())
            self.comet.log_metric('test loss', epoch = self.epochs, value=self.err_te_hist[-1].cpu().detach().numpy())


    def test(self, te):
        '''
        get the loss over the test dataset
        :param te: test set dataloader
        :return: n/a
        '''
        err_te = []
        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            for i, testData in enumerate(te):
                loss = self.getLoss(testData)
                err_te.append(loss.data)  # record the loss

        self.err_te_hist.append(torch.mean(torch.stack(err_te)))


    def getLoss(self, train_data):
        """
        get the regression loss on a batch of datapoints
        :param train_data: sequences and scores
        :return: model loss over the batch
        """
        inputs = train_data[0]
        targets = train_data[1]
        if self.config.device == 'cuda':
            inputs = inputs.cuda()
            targets = targets.cuda()

        targets = (targets - self.mean)/self.std # standardize the targets during training
        standardized_target_max = torch.amax(targets)

        if self.config.proxy_clip_max:
            output = self.model(inputs.float(), clip = standardized_target_max)
        else:
            output = self.model(inputs.float())

        '''
        compare outputs
        
        ind = np.argsort(targets.cpu().detach().numpy())
        plt.plot(targets.cpu().detach().numpy()[ind],'.')
        plt.plot(output.cpu().detach().numpy()[ind],'.')
        '''

        return F.smooth_l1_loss(output[:,0], targets.float())


    def evaluate(self, Data, output="Average"):
        '''
        evaluate the model
        output types - if "Average" return the average of ensemble predictions
            - if 'Variance' return the variance of ensemble predictions
        # future upgrade - isolate epistemic uncertainty from intrinsic randomness
        :param Data: input data
        :return: model scores
        '''
        if self.config.device == 'cuda':
            Data = torch.Tensor(Data).cuda().float()
        else:
            Data = torch.Tensor(Data).float()

        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            out = self.model(Data).cpu().detach().numpy()
            if output == 'Average':
                return np.average(out,axis=1) * self.std + self.mean
            elif output == 'Variance':
                return np.var(out * self.std + self.mean,axis=1)
            elif output == 'Both':
                return np.average(out,axis=1) * self.std + self.mean, np.var(out * self.std,axis=1)


    def raw(self, Data, output="Average"):
        '''
        evaluate the model
        output types - if "Average" return the average of ensemble predictions
            - if 'Variance' return the variance of ensemble predictions
        # future upgrade - isolate epistemic uncertainty from intrinsic randomness
        :param Data: input data
        :return: model scores
        '''
        if self.config.device == 'cuda':
            Data = torch.Tensor(Data).cuda().float()
        else:
            Data = torch.Tensor(Data).float()

        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            out = self.model(Data).cpu().detach().numpy()
            if output == 'Average':
                return np.average(out,axis=1)
            elif output == 'Variance':
                return np.var(out,axis=1)
            elif output == 'Both':
                return np.average(out,axis=1), np.var(out,axis=1)


class buildDataset():
    '''
    build dataset object
    '''
    def __init__(self, config, dataset = None):
        if dataset is None:
            dataset = np.load('nupack_dataset.npy',allow_pickle=True).item()
        self.samples = dataset['samples']
        self.targets = dataset['scores']

        self.samples, self.targets = shuffle(self.samples, self.targets, random_state=config.dataset_seed)
        self.samples = self.samples[:config.dataset_size]
        self.targets = self.targets[:config.dataset_size]

    def reshuffle(self, seed=None):
        self.samples, self.targets = shuffle(self.samples, self.targets, random_state=seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def getFullDataset(self):
        return self.samples, self.targets

    def getStandardization(self):
        return np.mean(self.targets), np.sqrt(np.var(self.targets))


def getDataloaders(config, ensembleIndex, dataset): # get the dataloaders, to load the dataset in batches
    '''
    creat dataloader objects from the dataset
    :param config:
    :return:
    '''
    training_batch = config.proxy_training_batch_size
    dataset = buildDataset(config, dataset)  # get data
    if config.proxy_shuffle_dataset:
        dataset.reshuffle(seed=ensembleIndex)
    train_size = int(0.5 * len(dataset))  # split data into training and test sets

    test_size = len(dataset) - train_size

    # construct dataloaders for inputs and targets
    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size): # take the training data from the end - we will get the newly appended datapoints this way without ever seeing the test set
        train_dataset.append(dataset[i])
    for i in range(test_size): # test data is drawn from oldest datapoints
        test_dataset.append(dataset[i])

    tr = data.DataLoader(train_dataset, batch_size=training_batch, shuffle=True, num_workers= 0, pin_memory=False)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers= 0, pin_memory=False) # num_workers must be zero or multiprocessing will not work (can't spawn multiprocessing within multiprocessing)

    return tr, te, dataset.__len__()


def getDataSize(dataset):
    samples = dataset['samples']

    return len(samples[0])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class transformer(nn.Module):
    def __init__(self,config):
        super(transformer,self).__init__()

        self.embedDim = config.proxy_model_embedding_width
        self.filters = config.proxy_model_width
        self.encoder_layers = config.proxy_model_layers
        self.decoder_layers = config.proxy_model_layers
        self.maxLen = config.max_sample_length
        self.dictLen = config.dataset_dict_size
        self.classes = int(config.dataset_dict_size + 1)
        self.heads = min([4, max([1,self.embedDim//self.dictLen])])
        act_func = 'gelu'

        self.positionalEncoder = PositionalEncoding(self.embedDim, max_len = self.maxLen, dropout=config.proxy_attention_dropout_prob)
        self.embedding = nn.Embedding(self.dictLen + 1, embedding_dim = self.embedDim)

        factory_kwargs = {'device': None, 'dtype': None}
        #encoder_layer = nn.TransformerEncoderLayer(self.embedDim, nhead = self.heads,dim_feedforward=self.filters, activation='gelu', dropout=0)
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.layers)
        self.decoder_linear = []
        self.encoder_norms1 = []
        self.encoder_norms2 = []
        self.decoder_norms = []
        self.encoder_dropouts = []
        self.decoder_dropouts = []
        self.encoder_linear = []
        self.self_attn_layers = []
        self.aggregation_mode = config.proxy_aggregation
        self.encoder_activations = []
        self.decoder_activations = []

        for i in range(self.encoder_layers):
            self.encoder_linear.append(nn.Linear(self.embedDim,self.embedDim))
            self.self_attn_layers.append(nn.MultiheadAttention(self.embedDim, self.heads, dropout=config.proxy_attention_dropout_prob, batch_first=False, **factory_kwargs))
            self.encoder_activations.append(Activation(act_func, self.filters))

            if config.proxy_dropout_prob != 0:
                self.encoder_dropouts.append(nn.Dropout(config.proxy_dropout_prob))
            else:
                self.encoder_dropouts.append(nn.Identity())

            if config.proxy_attention_norm == 'layer': # work in progress
                self.encoder_norms1.append(nn.LayerNorm(self.embedDim))
                self.encoder_norms2.append(nn.LayerNorm(self.embedDim))

            else:
                self.encoder_norms1.append(nn.Identity())
                self.encoder_norms2.append(nn.Identity())


        for i in range(self.decoder_layers):
            if i == 0:
                self.decoder_linear.append(nn.Linear(self.embedDim, self.filters))
            else:
                self.decoder_linear.append(nn.Linear(self.filters, self.filters))

            self.decoder_activations.append(Activation(act_func,self.filters))
            if config.proxy_dropout_prob != 0:
                self.decoder_dropouts.append(nn.Dropout(config.proxy_dropout_prob))
            else:
                self.decoder_dropouts.append(nn.Identity())

            if config.proxy_norm == 'batch':  # work in progress
                self.decoder_norms.append(nn.BatchNorm1d(self.filters))
            else:
                self.decoder_norms.append(nn.Identity())

        self.decoder_linear = nn.ModuleList(self.decoder_linear)
        self.encoder_linear = nn.ModuleList(self.encoder_linear)
        self.self_attn_layers = nn.ModuleList(self.self_attn_layers)
        self.encoder_norms1 = nn.ModuleList(self.encoder_norms1)
        self.encoder_norms2 = nn.ModuleList(self.encoder_norms2)
        self.decoder_norms = nn.ModuleList(self.decoder_norms)
        self.encoder_dropouts = nn.ModuleList(self.encoder_dropouts)
        self.decoder_dropouts = nn.ModuleList(self.decoder_dropouts)
        self.encoder_activations = nn.ModuleList(self.encoder_activations)
        self.decoder_activations = nn.ModuleList(self.decoder_activations)

        self.output_layer = nn.Linear(self.filters,1,bias=False)

    def forward(self,x, clip = None):
        x_key_padding_mask = (x==0).clone().detach() # zero out the attention of empty sequence elements
        x = self.embedding(x.transpose(1,0).int()) # [seq, batch]
        x = self.positionalEncoder(x)
        #x = self.encoder(x,src_key_padding_mask=x_key_padding_mask)
        #x = x.permute(1,0,2).reshape(x_key_padding_mask.shape[0], int(self.embedDim*self.maxLen))
        for i in range(self.encoder_layers):
            # Self-attention block
            residue = x.clone()
            x = self.encoder_norms1[i](x)
            x = self.self_attn_layers[i](x,x,x,key_padding_mask=x_key_padding_mask)[0]
            x = self.encoder_dropouts[i](x)
            x = x + residue

            # dense block
            residue = x.clone()
            x = self.encoder_norms2[i](x)
            x = self.encoder_linear[i](x)
            x = self.encoder_activations[i](x)
            x = x + residue

        if self.aggregation_mode == 'mean':
            x = x.mean(dim=0) # mean aggregation
        elif self.aggregation_mode == 'sum':
            x = x.sum(dim=0) # sum aggregation
        elif self.aggregation_mode == 'max':
            x = x.max(dim=0) # max aggregation
        else:
            print(self.aggregation_mode + ' is not a valid aggregation mode!')

        for i in range(self.decoder_layers):
            if i != 0:
                residue = x.clone()
            x = self.decoder_linear[i](x)
            x = self.decoder_norms[i](x)
            x = self.decoder_dropouts[i](x)
            x = self.decoder_activations[i](x)
            if i != 0:
                x += residue

        x = self.output_layer(x)

        if clip is not None:
            x = torch.clip(x,max=clip)

        return x


class MLP(nn.Module):
    def __init__(self,config):
        super(MLP,self).__init__()
        # initialize constants and layers

        if True:
            act_func = 'gelu'

        self.inputLength = config.max_sample_length
        self.layers = config.proxy_model_layers
        self.filters = config.proxy_model_width
        self.classes = int(config.dataset_dict_size + 1)
        self.init_layer_depth = int(self.inputLength * self.classes)

        # build input and output layers
        self.initial_layer = nn.Linear(int(self.inputLength * self.classes), self.filters) # layer which takes in our sequence in one-hot encoding
        self.activation1 = Activation(act_func,self.filters,config)

        self.output_layers = nn.Linear(self.filters, 1, bias=False)

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        self.norms = []
        self.dropouts = []

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters,self.filters))
            self.activations.append(Activation(act_func, self.filters))
            if config.proxy_norm == 'batch':
                self.norms.append(nn.BatchNorm1d(self.filters))
            else:
                self.norms.append(nn.Identity())
            if config.proxy_dropout_prob != 0:
                self.dropouts.append(nn.Dropout(config.proxy_dropout_prob))
            else:
                self.dropouts.append(nn.Identity())

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        self.norms = nn.ModuleList(self.norms)
        self.dropouts = nn.ModuleList(self.dropouts)


    def forward(self, x, clip = None):
        x = F.one_hot(x.long(),num_classes=self.classes)
        x = x.reshape(x.shape[0], self.init_layer_depth).float()
        x = self.activation1(self.initial_layer(x)) # apply linear transformation and nonlinear activation
        for i in range(self.layers):
            residue = x.clone()
            x = self.lin_layers[i](x)
            x = self.norms[i](x)
            x = self.dropouts[i](x)
            x = self.activations[i](x)
            x += residue

        x = self.output_layers(x) # each task has its own head

        if clip is not None:
            x = torch.clip(x,max=clip)

        return x


class kernelActivation(nn.Module): # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis)) # positive and negative values for Dirichlet Kernel
        gamma = 1/(6*(self.dict[-1]-self.dict[-2])**2) # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma',torch.ones(1) * gamma) #

        #self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1,1), groups=int(channels), bias=False)

        #nn.init.normal(self.linear.weight.data, std=0.1)


    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x)==2:
            x = x.reshape(2,self.channels,1)

        return torch.exp(-self.gamma*(x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1) # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]) # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1) # apply linear coefficients and sum

        #y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        #for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'gelu':
            self.activation = F.gelu
        elif activation_func == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)




class Oracle():
    def __init__(self, config):
        '''
        initialize the oracle
        :param config:
        '''
        self.config = config
        self.seqLen = self.config.max_sample_length

        self.initRands()


    def initRands(self):
        '''
        initialize random numbers for custom-made toy functions
        :return:
        '''
        np.random.seed(self.config.toy_oracle_seed)

        # set these to be always positive to play nice with gFlowNet sampling
        if True:#self.config.test_mode:
            self.linFactors = -np.ones(self.seqLen) # Uber-simple function, for testing purposes - actually nearly functionally identical to one-max, I believe
        else:
            self.linFactors = np.abs(np.random.randn(self.seqLen))  # coefficients for linear toy energy

        hamiltonian = np.random.randn(self.seqLen,self.seqLen) # energy function
        self.hamiltonian = np.tril(hamiltonian) + np.tril(hamiltonian, -1).T # random symmetric matrix

        pham = np.zeros((self.seqLen,self.seqLen,self.config.dataset_dict_size,self.config.dataset_dict_size))
        for i in range(pham.shape[0]):
            for j in range(i, pham.shape[1]):
                for k in range(pham.shape[2]):
                    for l in range(k, pham.shape[3]):
                        num =  - np.random.uniform(0,1)
                        pham[i, j, k, l] = num
                        pham[i, j, l, k] = num
                        pham[j, i, k, l] = num
                        pham[j, i, l, k] = num
        self.pottsJ = pham # multilevel spin Hamiltonian (Potts Hamiltonian) - coupling term
        self.pottsH = np.random.randn(self.seqLen,self.config.dataset_dict_size) # Potts Hamiltonian - onsite term

        # W-model parameters
        # first get the binary dimension size
        aa = np.arange(self.config.dataset_dict_size)
        if self.config.dataset_variable_length:
            aa = np.clip(aa, 1, self.config.dataset_dict_size) #  merge padding with class 1
        x0 = np.binary_repr(aa[-1])
        dimension = int(len(x0) * self.config.dataset_size)

        mu = np.random.randint(1, dimension + 1)
        v = np.random.randint(1, dimension + 1)
        m = np.random.randint(1, dimension)
        n = np.random.randint(1, dimension)
        gamma = np.random.randint(0, int(n * (n - 1 ) / 2))
        self.mu, self.v, self.m, self.n, self.gamma = [mu, v, m, n, gamma]


    def initializeDataset(self,save = True, returnData = False, customSize=None):
        '''
        generate an initial toy dataset with a given number of samples
        need an extra factor to speed it up (duplicate filtering is very slow)
        :param numSamples:
        :return:
        '''
        data = {}
        np.random.seed(self.config.dataset_seed)
        if customSize is None:
            datasetLength = self.config.dataset_size
        else:
            datasetLength = customSize

        if self.config.dataset_variable_length:
            samples = []
            while len(samples) < datasetLength:
                for i in range(self.config.min_sample_length, self.config.max_sample_length + 1):
                    samples.extend(np.random.randint(0 + 1, self.config.dataset_dict_size + 1, size=(int(100 * self.config.dataset_dict_size * i), i)))

                samples = self.numpy_fillna(np.asarray(samples, dtype = object)) # pad sequences up to maximum length
                samples = filterDuplicateSamples(samples) # this will naturally proportionally punish shorter sequences
                if len(samples) < datasetLength:
                    samples = samples.tolist()
            np.random.shuffle(samples) # shuffle so that sequences with different lengths are randomly distributed
            samples = samples[:datasetLength] # after shuffle, reduce dataset to desired size, with properly weighted samples
        else: # fixed sample size
            samples = np.random.randint(1, self.config.dataset_dict_size + 1,size=(datasetLength, self.config.max_sample_length))
            samples = filterDuplicateSamples(samples)
            while len(samples) < datasetLength:
                samples = np.concatenate((samples,np.random.randint(1, self.config.dataset_dict_size + 1, size=(datasetLength, self.config.max_sample_length))),0)
                samples = filterDuplicateSamples(samples)

        data['samples'] = samples
        data['scores'] = self.score(data['samples'])

        if save:
            np.save('nupack_dataset', data)
        if returnData:
            np.save('nupack_dataset', data)
            return data


    def score(self, queries):
        '''
        assign correct scores to selected sequences
        :param queries: sequences to be scored
        :return: computed scores
        '''
        if isinstance(queries, list):
            queries = np.asarray(queries) # convert queries to array
        block_size = int(1e4) # score in blocks of maximum 10000
        scores_list = []
        scores_dict = {}
        for idx in tqdm.tqdm(range(len(queries) // block_size + bool(len(queries) % block_size))):
            queryBlock = queries[idx * block_size:(idx + 1) * block_size]
            scores_block = self.getScore(queryBlock)
            if isinstance(scores_block, dict):
                for k, v in scores_block.items():
                    if k in scores_dict:
                        scores_dict[k].extend(list(v))
                    else:
                        scores_dict.update({k: list(v)})
            else:
                scores_list.extend(self.getScore(queryBlock))
        if len(scores_list) > 0:
            return np.asarray(scores_list)
        else:
            return {k: np.asarray(v) for k, v in scores_dict.items()}


    def getScore(self,queries):
        if self.config.oracle == 'nupack energy':
            return self.nupackScore(queries, returnFunc = 'energy')
        elif self.config.oracle == 'nupack pins':
            return -self.nupackScore(queries, returnFunc = 'pins')
        elif self.config.oracle == 'nupack pairs':
            return -self.nupackScore(queries, returnFunc = 'pairs')
        elif isinstance(self.config.oracle, list) and all(["nupack " in el for el in self.config.dataset_oracle]):
            return self.nupackScore(queries, returnFunc=[el.replace("nupack ", "") for el in self.config.oracle])
        else:
            raise NotImplementedError("Unknown orackle type")

    def numbers2letters(self, sequences):  # Tranforming letters to numbers (1234 --> ATGC)
        '''
        Converts numerical values to ATCG-format
        :param sequences: numerical DNA sequences to be converted
        :return: DNA sequences in ATCG format
        '''
        if type(sequences) != np.ndarray:
            sequences = np.asarray(sequences)

        my_seq = ["" for x in range(len(sequences))]
        row = 0
        for j in range(len(sequences)):
            seq = sequences[j, :]
            assert type(seq) != str, 'Function inputs must be a list of equal length strings'
            for i in range(len(sequences[0])):
                na = seq[i]
                if na == 1:
                    my_seq[row] += 'A'
                elif na == 2:
                    my_seq[row] += 'T'
                elif na == 3:
                    my_seq[row] += 'C'
                elif na == 4:
                    my_seq[row] += 'G'
            row += 1
        return my_seq


    def numpy_fillna(self, data):
        '''
        function to pad uneven-length vectors up to the max with zeros
        :param data:
        :return:
        '''
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        out = np.zeros(mask.shape, dtype=object)
        out[mask] = np.concatenate(data)
        return out


    def nupackScore(self, queries, returnFunc='energy'):
        # Nupack requires Linux OS.
        #use nupack instead of seqfold - more stable and higher quality predictions in general
        #returns the energy of the most probable structure only
        #:param queries:
        #:param returnFunct 'energy' 'pins' 'pairs'
        #:return:

        temperature = 310.0  # Kelvin
        ionicStrength = 1.0 # molar
        sequences = self.numbers2letters(queries)

        energies = np.zeros(len(sequences))
        strings = []
        nPins = np.zeros(len(sequences)).astype(int)
        nPairs = 0
        ssStrings = np.zeros(len(sequences),dtype=object)

        # parallel evaluation - fast
        strandList = []
        comps = []
        i = -1
        for sequence in sequences:
            i += 1
            strandList.append(Strand(sequence, name='strand{}'.format(i)))
            comps.append(Complex([strandList[-1]], name='comp{}'.format(i)))

        set = ComplexSet(strands=strandList, complexes=SetSpec(max_size=1, include=comps))
        model1 = Model(material='dna', celsius=temperature - 273, sodium=ionicStrength)
        results = complex_analysis(set, model=model1, compute=['mfe'])
        for i in range(len(energies)):
            energies[i] = results[comps[i]].mfe[0].energy
            ssStrings[i] = str(results[comps[i]].mfe[0].structure)

        dict_return = {}
        if 'pins' in returnFunc:
            for i in range(len(ssStrings)):
                indA = 0  # hairpin completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == '(':
                        indA += 1
                    elif ssStrings[i][j] == ')':
                        indA -= 1
                        if indA == 0:  # if we come to the end of a distinct hairpin
                            nPins[i] += 1
            dict_return.update({"pins": nPins})
        if 'pairs' in returnFunc:
            nPairs = np.asarray([ssString.count('(') for ssString in ssStrings]).astype(int)
            dict_return.update({"pairs": nPairs})
        if 'energy' in returnFunc:
            dict_return.update({"energy": energies})

        if isinstance(returnFunc, list):
            if len(returnFunc) > 1:
                return dict_return
            else:
                return dict_return[returnFunc[0]]
        else:
            return dict_return[returnFunc]





class Trainer():
    def __init__(self, config):
        '''
        setup: dataset, comet
        '''
        self.config = config

        # get dataset
        try:
            self.dataset = np.load('nupack_dataset.npy',allow_pickle=True).item()
            print('Loaded premade dataset')
        except:
            print('Started dataset generation')
            oracle = Oracle(config)
            t0 = time.time()
            self.dataset = oracle.initializeDataset(returnData=True)
            print('Building dataset took {} seconds'.format(time.time()-t0))

        datasetBuilder = buildDataset(self.config, self.dataset)
        self.mean, self.std = datasetBuilder.getStandardization()

        if config.comet_project:
            self.comet = Experiment(
                project_name=config.comet_project, display_summary_level=0,
            )
            if config.comet_tags:
                if isinstance(config.comet_tags, list):
                    self.comet.add_tags(config.comet_tags)
                else:
                    self.comet.add_tag(config.comet_tags)

            self.comet.set_name("run {}".format(config.run_num))

            self.comet.log_parameters(vars(config))
        else:
            self.comet = None


    def train(self):
        test_loss = []
        train_loss = []
        for j in range(self.config.proxy_model_ensemble_size):
            printRecord('Training model {}'.format(j))
            self.resetModel(j)
            if j == 0:
                nParams = get_n_params(self.model.model)
                printRecord('Proxy model has {} parameters'.format(int(nParams)))

            err_tr_hist, err_te_hist = self.model.converge(self.dataset, returnHist=True)
            train_loss.append(err_tr_hist)
            test_loss.append(err_te_hist)

            if self.comet:
                epochs = len(err_tr_hist)
                for i in range(epochs):
                    self.comet.log_metric('proxy train loss iter {}'.format(j), step=i, value=err_tr_hist[i])
                    self.comet.log_metric('proxy test loss iter {}'.format(j), step=i, value=err_te_hist[i])



    def resetModel(self,ensembleIndex, returnModel = False):
        '''
        load a new instance of the model with reset parameters
        :return:
        '''
        try: # if we have a model already, delete it
            del self.model
        except:
            pass
        self.model = nupackModel(self.config,ensembleIndex, self.mean, self.std, self.comet)
        if returnModel:
            return self.model




if __name__ == "__main__":
    # Handle command line arguments and configuration
    parser = ArgumentParser()
    parser = add_args(parser)
    config = parser.parse_args()
    config = process_config(config)

    trainer = Trainer(config)
    trainer.train()
