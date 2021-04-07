import os
import math
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from torch.autograd import Variable

import datetime, shutil, argparse, logging, sys

import utils

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument("--data_scale", default=60, type=float)
    parser.add_argument("--dec_size", default=[1024, 512, 1024], type=list)
    parser.add_argument("--enc_dest_size", default=[256, 128], type=list)
    parser.add_argument("--enc_latent_size", default=[256, 512], type=list)
    parser.add_argument("--enc_past_size", default=[512, 256], type=list)
    parser.add_argument("--predictor_hidden_size", default=[1024, 512, 256], type=list)
    parser.add_argument("--non_local_theta_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_phi_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_g_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_dim", default=128, type=int)
    parser.add_argument("--fdim", default=16, type=int)
    parser.add_argument("--future_length", default=12, type=int)
    parser.add_argument("--device", default=1, type=int)
    parser.add_argument("--kld_coeff", default=0.8, type=float)
    parser.add_argument("--future_loss_coeff", default=1, type=float)
    parser.add_argument("--dest_loss_coeff", default=2, type=float)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--lr_decay_step_size", default=1, type=int)
    parser.add_argument("--lr_decay_gamma", default=0.5, type=float)
    parser.add_argument("--mu", default=0, type=float)
    parser.add_argument("--n_values", default=20, type=int)
    parser.add_argument("--nonlocal_pools", default=3, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--past_length", default=8, type=int)
    parser.add_argument("--sigma", default=1.3, type=float)
    parser.add_argument("--zdim", default=16, type=int)
    parser.add_argument("--print_log", default=6, type=int)
    parser.add_argument("--sub_goal_indexes", default=[2, 5, 8, 11], type=list)


    parser.add_argument('--e_prior_sig', type=float, default=2, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=2, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='lrelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_steps_pcd', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')
    parser.add_argument('--e_lr', default=0.00003, type=float)
    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--e_max_norm', type=float, default=25, help='max norm allowed')
    parser.add_argument('--e_decay', default=1e-4, help='weight decay for ebm')
    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--e_beta1', default=0.9, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)
    parser.add_argument('--memory_size', default=200000, type=int)


    parser.add_argument('--dataset_name', type=str, default='hotel')
    parser.add_argument('--dataset_folder', type=str, default='dataset')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--batch_size',type=int,default=70)

    parser.add_argument('--ny', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='saved_models/lbebm_hotel.pt')


    return parser.parse_args()


def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))

def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]

def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def main():

    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)

    args = parse_args()
    set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)
    args.way_points = list(set(list(range(args.future_length))) - set(args.sub_goal_indexes))

    logger = setup_logging('job{}'.format(0), output_dir, console=True)
    logger.info(args)

    if args.val_size==0:
        train_dataset, _ = utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True, verbose=True)
        val_dataset, _ = utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False, verbose=True)
    else:
        train_dataset, val_dataset = utils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs, args.preds, delim=args.delim, train=True, verbose=args.verbose)

    test_dataset, _ =  utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True, verbose=True)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size*10, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*10, shuffle=False, num_workers=0)

    
    def initial_pos(traj_batches):
        batches = []
        for b in traj_batches:
            starting_pos = b[:,7,:].copy()/1000
            batches.append(starting_pos)
        return batches

    def sample_p_0(n, nz=16):
        return args.e_init_sig * torch.randn(*[n, nz]).double().cuda()

    def calculate_loss(dest, dest_recon, mean, log_var, criterion, future, interpolated_future, sub_goal_indexes):
        dest_loss = criterion(dest, dest_recon)
        future_loss = criterion(future, interpolated_future)
        subgoal_reg = criterion(dest_recon, interpolated_future.view(dest.size(0), future.size(1)//2, 2)[:, sub_goal_indexes, :].view(dest.size(0), -1))
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return dest_loss, future_loss, kl, subgoal_reg

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
            super(MLP, self).__init__()
            dims = []
            dims.append(input_dim)
            dims.extend(hidden_size)
            dims.append(output_dim)
            self.layers = nn.ModuleList()
            for i in range(len(dims)-1):
                self.layers.append(nn.Linear(dims[i], dims[i+1]))

            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()

            self.sigmoid = nn.Sigmoid() if discrim else None
            self.dropout = dropout

        def forward(self, x):
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i != len(self.layers)-1:
                    x = self.activation(x)
                    if self.dropout != -1:
                        x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
                elif self.sigmoid:
                    x = self.sigmoid(x)
            return x


    class ReplayMemory(object):
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, input_memory):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = input_memory
            self.position = (self.position + 1) % self.capacity

        def sample(self, n=100):
            samples = random.sample(self.memory, n)
            return torch.cat(samples)

        def __len__(self):
            return len(self.memory)


    class LBEBM(nn.Module):
        def __init__(self, 
                    enc_past_size, 
                    enc_dest_size, 
                    enc_latent_size, 
                    dec_size, 
                    predictor_size, 
                    fdim, 
                    zdim, 
                    sigma, 
                    past_length, 
                    future_length):
            super(LBEBM, self).__init__()
            self.zdim = zdim
            self.sigma = sigma
            self.nonlocal_pools = args.nonlocal_pools
            non_local_dim = args.non_local_dim
            non_local_phi_size = args.non_local_phi_size
            non_local_g_size = args.non_local_g_size
            non_local_theta_size = args.non_local_theta_size

            self.encoder_past = MLP(input_dim=past_length*2, output_dim=fdim, hidden_size=enc_past_size)
            self.encoder_dest = MLP(input_dim=len(args.sub_goal_indexes)*2, output_dim=fdim, hidden_size=enc_dest_size)
            self.encoder_latent = MLP(input_dim=2*fdim, output_dim=2*zdim, hidden_size=enc_latent_size)
            self.decoder = MLP(input_dim=fdim+zdim, output_dim=len(args.sub_goal_indexes)*2, hidden_size=dec_size)
            self.predictor = MLP(input_dim=2*fdim, output_dim=2*(future_length), hidden_size=predictor_size)

            self.non_local_theta = MLP(input_dim = fdim, output_dim = non_local_dim, hidden_size=non_local_theta_size)
            self.non_local_phi = MLP(input_dim = fdim, output_dim = non_local_dim, hidden_size=non_local_phi_size)
            self.non_local_g = MLP(input_dim = fdim, output_dim = fdim, hidden_size=non_local_g_size)

            self.EBM = nn.Sequential(
                nn.Linear(zdim + fdim, 200),
                nn.GELU(),
                nn.Linear(200, 200),
                nn.GELU(),
                nn.Linear(200, args.ny),
                )
                        
            self.replay_memory = ReplayMemory(args.memory_size)

        def forward(self, x, dest=None, mask=None, iteration=1, y=None):
            
            ftraj = self.encoder_past(x)

            if mask:
                for _ in range(self.nonlocal_pools):
                    ftraj = self.non_local_social_pooling(ftraj, mask)

            if self.training:
                pcd = True if len(self.replay_memory) == args.memory_size else False
                if pcd:
                    z_e_0 = self.replay_memory.sample(n=ftraj.size(0)).clone().detach().cuda()
                else:
                    z_e_0 = sample_p_0(n=ftraj.size(0), nz=self.zdim)
                z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=pcd, verbose=(iteration % 1000==0))
                for _z_e_k in z_e_k.clone().detach().cpu().split(1):
                    self.replay_memory.push(_z_e_k)
            else:
                z_e_0 = sample_p_0(n=ftraj.size(0), nz=self.zdim)
                z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=False, verbose=(iteration % 1000==0), y=y)                        
            z_e_k = z_e_k.double().cuda()

            if self.training:
                dest_features = self.encoder_dest(dest)
                features = torch.cat((ftraj, dest_features), dim=1)
                latent =  self.encoder_latent(features)
                mu = latent[:, 0:self.zdim]
                logvar = latent[:, self.zdim:]

                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_().cuda()
                z_g_k = eps.mul(var).add_(mu)
                z_g_k = z_g_k.double().cuda()

            if self.training:
                decoder_input = torch.cat((ftraj, z_g_k), dim=1)
            else:
                decoder_input = torch.cat((ftraj, z_e_k), dim=1)
            generated_dest = self.decoder(decoder_input)

            if self.training:
                generated_dest_features = self.encoder_dest(generated_dest)
                prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
                pred_future = self.predictor(prediction_features)

                en_pos = self.ebm(z_g_k, ftraj).mean()
                en_neg = self.ebm(z_e_k.detach().clone(), ftraj).mean()
                cd = en_pos - en_neg

                return generated_dest, mu, logvar, pred_future, cd, en_pos, en_neg, pcd

            return generated_dest

        def ebm(self, z, condition, cls_output=False):
            condition_encoding = condition.detach().clone()
            z_c = torch.cat((z, condition_encoding), dim=1)
            conditional_neg_energy = self.EBM(z_c)
            assert conditional_neg_energy.shape == (z.size(0), args.ny)
            if cls_output:
                return - conditional_neg_energy
            else:
                return - conditional_neg_energy.logsumexp(dim=1)
        
        def sample_langevin_prior_z(self, z, condition, pcd=False, verbose=False, y=None):
            z = z.clone().detach()
            z.requires_grad = True
            _e_l_steps = args.e_l_steps_pcd if pcd else args.e_l_steps
            _e_l_step_size = args.e_l_step_size
            for i in range(_e_l_steps):
                if y is None:
                    en = self.ebm(z, condition)
                else:
                    en = self.ebm(z, condition, cls_output=True)[range(z.size(0)), y]
                z_grad = torch.autograd.grad(en.sum(), z)[0]

                z.data = z.data - 0.5 * _e_l_step_size * _e_l_step_size * (z_grad + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
                if args.e_l_with_noise:
                    z.data += _e_l_step_size * torch.randn_like(z).data

                if (i % 5 == 0 or i == _e_l_steps - 1) and verbose:
                    if y is None:
                        print('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, _e_l_steps, en.sum().item()))
                    else:
                        logger.info('Conditional Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i + 1, _e_l_steps, en.sum().item()))

                z_grad_norm = z_grad.view(z_grad.size(0), -1).norm(dim=1).mean()

            return z.detach(), z_grad_norm


        def predict(self, past, generated_dest):
            ftraj = self.encoder_past(past)
            generated_dest_features = self.encoder_dest(generated_dest)
            prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
            interpolated_future = self.predictor(prediction_features)

            return interpolated_future


        def non_local_social_pooling(self, feat, mask):
            theta_x = self.non_local_theta(feat)
            phi_x = self.non_local_phi(feat).transpose(1,0)
            f = torch.matmul(theta_x, phi_x)
            f_weights = F.softmax(f, dim = -1)
            f_weights = f_weights * mask
            f_weights = F.normalize(f_weights, p=1, dim=1)
            pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

            return pooled_f + feat

    
    def train(model, optimizer, epoch, sub_goal_indexes):
        model.train()
        train_loss, total_dest_loss, total_future_loss = 0, 0, 0
        criterion = nn.MSELoss()

        for i, trajx in enumerate(tr_dl):
            x = trajx['src'][:, :, :2]
            y = trajx['trg'][:, :, :2]
            x = x - trajx['src'][:, -1:, :2]
            y = y - trajx['src'][:, -1:, :2]

            x *= args.data_scale
            y *= args.data_scale
            x = x.double().cuda()
            y = y.double().cuda()

            x = x.view(-1, x.shape[1]*x.shape[2])
            dest = y[:, sub_goal_indexes, :].detach().clone().view(y.size(0), -1)
            future = y.view(y.size(0),-1)

            dest_recon, mu, var, interpolated_future, cd, en_pos, en_neg, pcd = model.forward(x, dest=dest, mask=None, iteration=i)

            optimizer.zero_grad()
            dest_loss, future_loss, kld, subgoal_reg = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future, sub_goal_indexes)
            loss = args.dest_loss_coeff * dest_loss + args.future_loss_coeff * future_loss + args.kld_coeff * kld  + cd + subgoal_reg
            loss.backward()

            train_loss += loss.item()
            total_dest_loss += dest_loss.item()
            total_future_loss += future_loss.item()
            optimizer.step()

            if (i+1) % args.print_log == 0:
                logger.info('{:5d}/{:5d} '.format(i, epoch) +
                            'dest_loss={:8.6f} '.format(dest_loss.item()) +
                            'future_loss={:8.6f} '.format(future_loss.item()) +
                            'kld={:8.6f} '.format(kld.item()) +
                            'cd={:8.6f} '.format(cd.item()) +
                            'en_pos={:8.6f} '.format(en_pos.item()) +
                            'en_neg={:8.6f} '.format(en_neg.item()) +
                            'pcd={} '.format(pcd) +
                            'subgoal_reg={}'.format(subgoal_reg.detach().cpu().numpy())
                )

        return train_loss, total_dest_loss, total_future_loss

    def test(model, dataloader, dataset, sub_goal_indexes, best_of_n=20):
        model.eval()

        total_dest_err = 0.
        total_overall_err = 0.

        for i, trajx in enumerate(dataloader):
            x = trajx['src'][:, :, :2]
            y = trajx['trg'][:, :, :2]
            x = x - trajx['src'][:, -1:, :2]
            y = y - trajx['src'][:, -1:, :2]

            x *= args.data_scale
            y *= args.data_scale
            x = x.double().cuda()
            y = y.double().cuda()



            y = y.cpu().numpy()

            x = x.view(-1, x.shape[1]*x.shape[2])

            plan = y[:, sub_goal_indexes, :].reshape(y.shape[0],-1)
            all_plan_errs = []
            all_plans = []
            for _ in range(best_of_n):
                # dest_recon = model.forward(x, initial_pos, device=device)
                # modes = torch.tensor(k % args.ny, device=device).long().repeat(batch_size)
                plan_recon = model.forward(x, mask=None)
                plan_recon = plan_recon.detach().cpu().numpy()
                all_plans.append(plan_recon)
                plan_err = np.linalg.norm(plan_recon - plan, axis=-1)
                all_plan_errs.append(plan_err)

            all_plan_errs = np.array(all_plan_errs) 
            all_plans = np.array(all_plans) 
            indices = np.argmin(all_plan_errs, axis=0)
            best_plan = all_plans[indices, np.arange(x.shape[0]),  :]

            # FDE
            best_dest_err = np.linalg.norm(best_plan[:, -2:] - plan[:, -2:], axis=1).sum()

            best_plan = torch.DoubleTensor(best_plan).cuda()
            interpolated_future = model.predict(x, best_plan)
            interpolated_future = interpolated_future.detach().cpu().numpy()

            # ADE        
            predicted_future = np.reshape(interpolated_future, (-1, args.future_length, 2))
            overall_err = np.linalg.norm(y - predicted_future, axis=-1).mean(axis=-1).sum()

            overall_err /= args.data_scale
            best_dest_err /= args.data_scale

            total_overall_err += overall_err
            total_dest_err += best_dest_err

        total_overall_err /= len(dataset)
        total_dest_err /= len(dataset)


        return total_overall_err, total_dest_err


    def run_training(args):
        model = LBEBM(
            args.enc_past_size,
            args.enc_dest_size,
            args.enc_latent_size,
            args.dec_size,
            args.predictor_hidden_size,
            args.fdim,
            args.zdim,
            args.sigma,
            args.past_length,
            args.future_length)
        
        model = model.double().cuda()
        optimizer = optim.Adam(model.parameters(), lr= args.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_gamma)



        best_val_ade = 50
        best_val_fde = 50
        best_test_ade = 50
        best_test_fde = 50

        patience_epoch = 0

        for epoch in range(args.num_epochs):
            train_loss, dest_loss, overall_loss = train(model, optimizer, epoch, args.sub_goal_indexes)
            overall_err, dest_err = test(model, val_dl, val_dataset, args.sub_goal_indexes, args.n_values)
            test_overall_err, test_dest_err = test(model, test_dl, test_dataset, args.sub_goal_indexes, args.n_values)

            patience_epoch += 1
            if best_val_ade > overall_err:
                patience_epoch = 0
                best_val_ade = overall_err
                best_val_fde = dest_err

            if best_test_ade > test_overall_err:
                best_test_ade = test_overall_err
                best_test_fde = test_dest_err

            logger.info("Train Loss {}".format(train_loss))
            logger.info("Overall Loss {}".format(overall_loss))
            logger.info("Dest Loss {}".format(dest_loss))
            logger.info("Val ADE {}".format(overall_err))
            logger.info("Val FDE {}".format(dest_err))
            logger.info("Val Best ADE {}".format(best_val_ade))
            logger.info("Val Best FDE {}".format(best_val_fde))
            logger.info("Test ADE {}".format(test_overall_err))
            logger.info("Test FDE {}".format(test_dest_err))
            logger.info("Test Best ADE {}".format(best_test_ade))
            logger.info("Test Best FDE {}".format(best_test_fde))
            logger.info("----->learning rate {}".format(optimizer.param_groups[0]['lr'])) 

            scheduler.step()



    def run_eval(args):
        model = LBEBM(
            args.enc_past_size,
            args.enc_dest_size,
            args.enc_latent_size,
            args.dec_size,
            args.predictor_hidden_size,
            args.fdim,
            args.zdim,
            args.sigma,
            args.past_length,
            args.future_length)
        
        model = model.double().cuda()
        
        ckpt = torch.load(args.model_path, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt['model_state_dict'])
        overall_err, dest_err = test(model, test_dl, test_dataset, args.sub_goal_indexes, args.n_values)
        logger.info("Test ADE {}".format(overall_err))
        logger.info("Test FDE {}".format(dest_err))

    if args.model_path:
        run_eval(args)
    else:
        run_training(args)

if __name__ == '__main__':
    main()