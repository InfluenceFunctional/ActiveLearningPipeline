import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

#from lib.model.gfn_transformer import GFNTransformer
from gfntransformer import GFNTransformer


class AMPDropoutTransformerRegressor(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.init_model()
        # self.model = GFNTransformer(num_tokens=num_tokens,
        #                     num_outputs=1,
        #                     num_hid=args.proxy_num_hid,
        #                     num_layers=args.proxy_num_layers, # TODO: add these as hyperparameters?
        #                     num_head=args.num_heads,
        #                     dropout=args.proxy_dropout,
        #                     max_len=max_len)

        # self.model.to(args.device)
        self.sigmoid = nn.Sigmoid()
        self.proxy_num_iterations = args.proxy_num_iterations

        self.device = args.device
        if args.task == "amp":
            self.eos_tok = self.tokenizer.numericalize(self.tokenizer.eos_token).item()
        elif args.task == "tfbind":
            self.eos_tok = 4

    def init_model(self):
        self.model = GFNTransformer(num_tokens=self.num_tokens,
                                    num_outputs=1,
                                    num_hid=self.args.proxy_num_hid,
                                    num_layers=self.args.proxy_num_layers,  # TODO: add these as hyperparameters?
                                    num_head=self.args.num_heads,
                                    dropout=self.args.proxy_dropout,
                                    max_len=self.max_len)

        self.model.to(self.args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), self.args.proxy_learning_rate,
                                    weight_decay=self.args.proxy_L2)

    def fit(self, data, reset=False):
        losses = []
        accs = []
        test_losses = []
        test_accs = []
        best_params = None
        best_accuracy = 0
        best_loss = 1e6
        early_stop_tol = 15  # self.args.proxy_early_stop_tol
        early_stop_count = 0
        epoch_length = 100
        test_random_seqs = []
        if reset:
            self.init_model()

        for it in tqdm(range(self.proxy_num_iterations)):
            x, y = data.sample(self.args.proxy_num_per_minibatch)
            x = self.tokenizer.process(x).to(self.device)
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)
            if self.args.task == "tfbind":
                output = self.model(x.swapaxes(0, 1), None).squeeze(1)
            else:
                output = self.model(x.swapaxes(0, 1), x.lt(self.eos_tok)).squeeze(1)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            losses.append(loss.item())

            self.args.logger.add_scalar("proxy_train_loss", loss.item())

            if not it % epoch_length:
                vx, vy = data.validation_set()
                vlosses = []
                for j in range(len(vx) // 256):
                    x = self.tokenizer.process(vx[j * 256:(j + 1) * 256]).to(self.device)
                    # import pdb; pdb.set_trace();
                    y = torch.tensor(vy[j * 256:(j + 1) * 256], device=self.device, dtype=torch.float).reshape(-1)
                    if self.args.task == "tfbind":
                        output = self.model(x.swapaxes(0, 1), None).squeeze(1)
                    else:
                        output = self.model(x.swapaxes(0, 1), x.lt(self.eos_tok)).squeeze(1)
                    loss = (output - y).pow(2)
                    vlosses.append(loss.sum().item())

                test_loss = np.sum(vlosses) / len(vx)
                test_losses.append(test_loss)

                self.args.logger.add_scalar("proxy_test_loss", test_loss)
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = [i.data.cpu().numpy() for i in self.model.parameters()]
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if early_stop_count >= early_stop_tol:
                    print(best_loss)
                    print('early stopping')
                    break

        if self.args.proxy_early_stop_to_best_params:
            # Put best parameters back in
            for i, besti in zip(self.model.parameters(), best_params):
                i.data = torch.tensor(besti).to(self.device)
        self.args.logger.save(self.args.save_path, self.args)
        return {}

    def forward(self, x, uncertainty_call=False):
        x = self.tokenizer.process(x).to(self.device)
        if uncertainty_call:
            if self.args.task == "tfbind":
                ys = self.model(x.swapaxes(0, 1), None)  # eos_tok == 2
            else:
                ys = self.model(x.swapaxes(0, 1), x.lt(self.eos_tok))  # eos_tok == 2
        else:
            self.model.eval()
            if self.args.task == "tfbind":
                ys = self.model(x.swapaxes(0, 1), None)  # eos_tok == 2
            else:
                ys = self.model(x.swapaxes(0, 1), x.lt(self.eos_tok))  # eos_tok == 2
            self.model.train()
        return ys

    def forward_with_uncertainty(self, x):
        self.model.train()
        with torch.no_grad():
            outputs = torch.cat([self(x, True) for _ in range(self.args.proxy_num_dropout_samples)])
        return outputs.mean(dim=0), outputs.std(dim=0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(path)