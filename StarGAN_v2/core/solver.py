"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import pandas as pd
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from piq import SSIMLoss

from core.model import build_model, _GridExtractor, EdgeLoss
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics, obtain_samples
from core.texture_loss import _texture_loss


class Solver(nn.Module):
    def __init__(self, args, domains):
        super().__init__()
        self.args = args
        self.domains = domains
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        # ---------------------------------------------- #
        if self.args.texture_loss:
            self.texture_extractor = nn.DataParallel(_GridExtractor())
            # Wrap the texture_extractor in DataParallel if you have multiple GPUs
            """if torch.cuda.device_count() > 1:
                self.texture_extractor = nn.DataParallel(self.texture_extractor)"""
        else:
            self.texture_extractor = None

        #if self.args.edge_loss:
        # lo inizializzo sempre perchè mi serve a prescindere per calcolare le metriche
        if args.ssim:
            self.ssim_loss = SSIMLoss(data_range=1.0) # we put 2 as data_range because it represents the range of the imput data => our data are normalized in [-1,1] => 1 - (-1) = 2
        else:
            self.ssim_loss = None
        self.edge_loss = nn.DataParallel(EdgeLoss())
        #else:
        # ---------------------------------------------- #

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_modules():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                # print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims
        texture_extractor = self.texture_extractor
        edge_loss = self.edge_loss
        ssim_loss = self.ssim_loss


        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
            training_history_csv_path = os.path.join(args.loss_monitoring_dir, 'history.csv')
            training_history_csv_path_df = pd.read_csv(training_history_csv_path, index_col=False)
            training_history = training_history_csv_path_df.to_dict(orient='list')
        elif args.texture_loss and not args.edge_loss:
            training_history = {
                'loss_d_latent': [],
                'loss_d_reference': [],
                'loss_g_adv_latent': [],
                'loss_g_sty_latent': [],
                'loss_g_ds_latent': [],
                'loss_g_cyc_latent': [],
                'loss_g_texture_latent': [],
                'loss_g_adv_reference': [],
                'loss_g_sty_reference': [],
                'loss_g_ds_reference': [],
                'loss_g_cyc_reference': [],
                'loss_g_texture_reference': [],
                'loss_g_latent': [],
                'loss_g_reference': []
            }
        elif args.texture_loss and args.edge_loss:
            training_history = {
                'loss_d_latent': [],
                'loss_d_reference': [],
                'loss_g_adv_latent': [],
                'loss_g_sty_latent': [],
                'loss_g_ds_latent': [],
                'loss_g_cyc_latent': [],
                'loss_g_texture_latent': [],
                'loss_g_edge_latent': [],
                'loss_g_adv_reference': [],
                'loss_g_sty_reference': [],
                'loss_g_ds_reference': [],
                'loss_g_cyc_reference': [],
                'loss_g_texture_reference': [],
                'loss_g_edge_reference': [],
                'loss_g_latent': [],
                'loss_g_reference': []
            }
        else:
            training_history = {
                'loss_d_latent': [],
                'loss_d_reference': [],
                'loss_g_adv_latent': [],
                'loss_g_sty_latent': [],
                'loss_g_ds_latent': [],
                'loss_g_cyc_latent': [],
                'loss_g_adv_reference': [],
                'loss_g_sty_reference': [],
                'loss_g_ds_reference': [],
                'loss_g_cyc_reference': [],
                'loss_g_latent': [],
                'loss_g_reference': []
            }


        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            print("shape of x_real is:", x_real.shape)
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            print("shape of x_ref is:", x_ref.shape)
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2
            print("shape of z_trg is:", z_trg.shape)

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            # print("### STO TRAINANDO IL DISCRIMINATOR ###")
            # print("sto per eseguire la compute_d_loss")
            d_loss_lat, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            # print("ho appena finito di eseguire la compute_d_losses_latent")
            self._reset_grad()
            d_loss_lat.backward()
            optims.discriminator.step()

            d_loss_ref, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            # print("ho appena finito di eseguire la compute_d_losses_ref")
            self._reset_grad()
            d_loss_ref.backward()
            optims.discriminator.step()


            # train the generator
            # print("### STO TRAINANDO IL GENERATOR ###")
            if args.ssim:
                g_loss_lat, g_losses_latent = compute_g_loss(
                    nets, texture_extractor, ssim_loss, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            else:
                g_loss_lat, g_losses_latent = compute_g_loss(
                    nets, texture_extractor, edge_loss, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss_lat.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            if args.ssim:
                g_loss_ref, g_losses_ref = compute_g_loss(
                    nets, texture_extractor, ssim_loss, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            else:
                g_loss_ref, g_losses_ref = compute_g_loss(
                    nets, texture_extractor, edge_loss, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss_ref.backward()
            optims.generator.step()

            # compute moving average of network parameters
            # print("### STO CALCOLANDO LA MOVING AVERAGE DEI PARAMETRI DELLA RETE ###")
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i + 1) % args.print_every == 0:
                # save in the history

                training_history['loss_d_latent']+=([d_loss_lat.cpu().detach().item()])
                training_history['loss_d_reference']+=([d_loss_ref.cpu().detach().item()])

                training_history['loss_g_latent']+=([g_loss_lat.cpu().detach().item()])
                training_history['loss_g_reference']+=([g_loss_ref.cpu().detach().item()])

                if args.resume_iter == 0 or (args.resume_iter > 0 and 'loss_g_adv_latent' in training_history):
                    training_history['loss_g_adv_latent'].append(g_losses_latent.adv)
                    training_history['loss_g_sty_latent'].append(g_losses_latent.sty)
                    training_history['loss_g_ds_latent'].append(g_losses_latent.ds)
                    training_history['loss_g_cyc_latent'].append(g_losses_latent.cyc)
                    training_history['loss_g_adv_reference'].append(g_losses_ref.adv)
                    training_history['loss_g_sty_reference'].append(g_losses_ref.sty)
                    training_history['loss_g_ds_reference'].append(g_losses_ref.ds)
                    training_history['loss_g_cyc_reference'].append(g_losses_ref.cyc)
                    if args.texture_loss:
                        training_history['loss_g_texture_latent'].append(g_losses_latent.texture)
                        training_history['loss_g_texture_reference'].append(g_losses_ref.texture)
                    if args.edge_loss:
                        training_history['loss_g_edge_latent'].append(g_losses_latent.edge)
                        training_history['loss_g_edge_reference'].append(g_losses_ref.edge)

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            if (i + 1) % args.show_every == 0:
                training_history_df = pd.DataFrame(training_history)
                training_history_path = args.loss_monitoring_dir + '/history.csv'
                training_history_df.to_csv(training_history_path, index=False)

                x_axis = list(range(100, i + 2, 100))  # this because we update the history every 100 iterations

                if args.resume_iter == 0 or (args.resume_iter > 0 and 'loss_g_adv_latent' in training_history):
                    ### PLOT OF THE DISCRIMINATOR LOSSES ###
                    fig_d, axs = plt.subplots(1, 2, figsize=(15, 15))

                    axs[0].plot(x_axis, training_history['loss_d_latent'])
                    axs[0].set_title('loss_d_latent')

                    axs[1].plot(x_axis, training_history['loss_d_reference'])
                    axs[1].set_title('loss_d_reference')

                    plt.tight_layout()

                    d_losses_plots_img = args.loss_monitoring_dir + '/d_losses_' + str(i + 1) + '.png'
                    fig_d.savefig(d_losses_plots_img)
                    plt.close(fig_d)

                    ### PLOT OF THE GENERATOR LOSSES ###
                    g_keys_numb = (len(training_history.keys()) - 2) / 2
                    fig_g, axs = plt.subplots(int(g_keys_numb), 2, figsize=(15, 20))

                    axs[0, 0].plot(x_axis, training_history['loss_g_latent'])
                    axs[0, 0].set_title('loss_g_latent')

                    axs[0, 1].plot(x_axis, training_history['loss_g_reference'])
                    axs[0, 1].set_title('loss_g_reference')

                    axs[1, 0].plot(x_axis, training_history['loss_g_adv_latent'])
                    axs[1, 0].set_title('loss_g_adv_latent')

                    axs[1, 1].plot(x_axis, training_history['loss_g_adv_reference'])
                    axs[1, 1].set_title('loss_g_adv_reference')

                    axs[2, 0].plot(x_axis, training_history['loss_g_sty_latent'])
                    axs[2, 0].set_title('loss_g_sty_latent')

                    axs[2, 1].plot(x_axis, training_history['loss_g_sty_reference'])
                    axs[2, 1].set_title('loss_g_sty_reference')

                    axs[3, 0].plot(x_axis, training_history['loss_g_ds_latent'])
                    axs[3, 0].set_title('loss_g_ds_latent')

                    axs[3, 1].plot(x_axis, training_history['loss_g_ds_reference'])
                    axs[3, 1].set_title('loss_g_ds_reference')

                    axs[4, 0].plot(x_axis, training_history['loss_g_cyc_latent'])
                    axs[4, 0].set_title('loss_g_cyc_latent')

                    axs[4, 1].plot(x_axis, training_history['loss_g_cyc_reference'])
                    axs[4, 1].set_title('loss_g_cyc_reference')

                    if args.texture_loss:
                        axs[5, 0].plot(x_axis, training_history['loss_g_texture_latent'])
                        axs[5, 0].set_title('loss_g_texture_latent')

                        axs[5, 1].plot(x_axis, training_history['loss_g_texture_reference'])
                        axs[5, 1].set_title('loss_g_texture_reference')

                    if args.edge_loss:
                        axs[6, 0].plot(x_axis, training_history['loss_g_edge_latent'])
                        axs[6, 0].set_title('loss_g_edge_latent')

                        axs[6, 1].plot(x_axis, training_history['loss_g_edge_reference'])
                        axs[6, 1].set_title('loss_g_edge_reference')

                    plt.tight_layout()

                    losses_plots_img = args.loss_monitoring_dir + '/g_losses_' + str(i + 1) + '.png'
                    fig_g.savefig(losses_plots_img)
                    plt.close(fig_g)
                else:
                    fig, axs = plt.subplots(2, 2)

                    axs[0, 0].plot(x_axis, training_history['loss_d_latent'])
                    axs[0, 0].set_title('loss_d_latent')

                    axs[0, 1].plot(x_axis, training_history['loss_d_reference'])
                    axs[0, 1].set_title('loss_d_reference')

                    axs[1, 0].plot(x_axis, training_history['loss_g_latent'])
                    axs[1, 0].set_title('loss_g_latent')

                    axs[1, 1].plot(x_axis, training_history['loss_g_reference'])
                    axs[1, 1].set_title('loss_g_reference')

                    plt.tight_layout()

                    losses_plots = args.loss_monitoring_dir + '/losses_' + str(i + 1) + '.png'
                    fig.savefig(losses_plots)
                    plt.close(fig)



            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, edge_loss, inputs=inputs_val, step=i + 1, domains=self.domains)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            # compute FID and LPIPS if necessary
            if (i + 1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i + 1, mode='latent', domains=self.domains)
                calculate_metrics(nets_ema, args, i + 1, mode='reference', domains=self.domains)

        #shutil.rmtree(args.tifs_dir, ignore_errors=True)

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, src.y, ref.x, ref.y, fname, self.domains)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent', domains=self.domains)

        #calculate_metrics(nets_ema, args, step=resume_iter, mode='reference', domains=self.domains)

        #shutil.rmtree(args.tifs_dir, ignore_errors=True)

    @torch.no_grad()
    def print_samples(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        obtain_samples(nets_ema, args, domains=self.domains)


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    # print(type(x_real))
    x_real.requires_grad_()
    # print("shape of x_real inside compute_d_loss is:", x_real.shape)
    # print("la shape di y_org è:", y_org.shape)
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, texture_extractor, edge_loss, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    # print("sono prima del training del generator")
    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)

    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)

    #print(x_rec_prove.shape)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    
    # texture loss
    if args.texture_loss:
        start = 128 # (512-256)/2
        end = start + 256 # 128 + 256
        # ---------------------------------------------------------- #
        x_rec_1_ch = x_rec[:, 1, start:end, start:end].unsqueeze(1)
        x_real_1_ch = x_real[:, 1, start:end, start:end].unsqueeze(1)
        # ---------------------------------------------------------- #
        #print("la dimensione dell'img che passo alla loss texture è:", x_rec_prove.shape, x_real_prove.shape)
        loss_texture, laplace_x, laplace_y = _texture_loss(x_rec_1_ch, x_real_1_ch, args, texture_extractor, nets.attention_layer)
        if args.edge_loss:
            x_fake_1_ch = x_fake[:, 1, :, :].unsqueeze(1)
            x_real_1_ch = x_real[:, 1, :, :].unsqueeze(1)
            if args.ssim:
                x_fake_min = torch.min(x_fake_1_ch)
                x_fake_max = torch.max(x_fake_1_ch)

                x_real_min = torch.min(x_real_1_ch)
                x_real_max = torch.max(x_real_1_ch)

                max = torch.max(x_fake_max, x_real_max).item()
                min = torch.min(x_fake_min, x_real_min).item()

                x_fake_1_ch = ((x_fake_1_ch - min) / (max - min))
                x_real_1_ch = ((x_real_1_ch - min) / (max - min))

                """print(x_fake_1_ch)
                print(x_real_1_ch)"""
                loss_edge = edge_loss(x_real_1_ch, x_fake_1_ch)
                #loss_edge = loss_edge[0]
                #print(f"the first element of loss_edge is {loss_edge}")
            else:
                loss_edge, _, _ = edge_loss(x_real_1_ch, x_fake_1_ch)
                loss_edge = loss_edge.mean() # SSIMLoss compute automatically the average over all the pairs of images
            #print(f"the mean is {loss_edge}")
            loss = loss_adv + args.lambda_sty * loss_sty \
                   - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc + loss_texture + loss_edge
            return loss, Munch(adv=loss_adv.item(),
                               sty=loss_sty.item(),
                               ds=loss_ds.item(),
                               cyc=loss_cyc.item(),
                               texture=loss_texture.item(),
                               edge=loss_edge.item()
                               )
        else:
            loss = loss_adv + args.lambda_sty * loss_sty \
                   - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc + loss_texture
            return loss, Munch(adv=loss_adv.item(),
                               sty=loss_sty.item(),
                               ds=loss_ds.item(),
                               cyc=loss_cyc.item(),
                               texture=loss_texture.item(),
                               )
    else:
        loss = args.lambda_adv * loss_adv + args.lambda_sty * loss_sty \
               - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
        return loss, Munch(adv=loss_adv.item(),
                           sty=loss_sty.item(),
                           ds=loss_ds.item(),
                           cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
