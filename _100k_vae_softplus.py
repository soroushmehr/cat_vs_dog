import logging

from blocks.algorithms import GradientDescent, RMSProp
from blocks.bricks import MLP, Linear, Tanh, Rectifier
from blocks.bricks.parallel import Fork
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.roles import add_role, PARAMETER
from blocks.utils import shared_floatx
from blocks.extensions import SimpleExtension

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from _100k_fuel import _100k

import numpy
np = numpy
import theano
from theano import tensor
T = tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import function as F

import scipy
import matplotlib.pyplot as plt
from dispims_color import dispims_color
from dispims import dispims

from batch_norm import normalization_layer as bn_layer

theano_rng = MRG_RandomStreams(134663)

nvis, nhid, nlat, learn_prior = 16*16*3, 1024, 512, True
original_shape = (16, 16)
every_n_epoch = 20
save_every_n_epoch = 20
"""
True: softplus()+0.001
False: use exp for mu_phi and log_sigma_phi
"""
softplussmall = True
activation = Rectifier
# activation = Tanh
learning_rate = 10E-4
RMS_decay = 0.9


def readyToPlot(to_plot):
    return to_plot.reshape(-1, 16, 16, 3).transpose(0, 3, 2, 1).reshape(-1, 3, 16, 16).reshape(-1, 16, 16, 3)


class Generate(SimpleExtension):
    def __init__(self, decoder, prior_mu, prior_log_sigma, BN=True, **kwargs):
        super(Generate, self).__init__(**kwargs)
        self.decoder = decoder
        self.im_cnt = 0
        self.BN = BN
        self.prior_mu = prior_mu
        self.prior_log_sigma = prior_log_sigma

    def do(self, callback_name, *args):
        self.im_cnt += 1
        print "gen =>", str(self.im_cnt),
        rnd = theano_rng.normal(size=(nlat,), dtype='float32')
        if learn_prior:
            rnd = self.prior_mu + rnd * (tensor.nnet.softplus(self.prior_log_sigma)+0.01)
        gen_fn = F([], self.decoder(rnd))
        gen_inp = numpy.array([gen_fn() for i in range(100)])
        print gen_inp.shape
        dispims_color(readyToPlot(gen_inp), save_path='gen_im_100k'+str(self.im_cnt)+'.png')
        print "Done"


class Reconstruct(SimpleExtension):
    def __init__(self, reconstructor, inp, BN=True, **kwargs):
        super(Reconstruct, self).__init__(**kwargs)
        self.reconstructor = reconstructor
        self.inp = inp
        self.im_cnt = 0
        self.BN = BN

    def do(self, callback_name, *args):
        print "rec =>",
        if self.BN:
            inp_th = T.matrix('inp_th')
            rec_fn = F([inp_th], self.reconstructor(inp_th))
        self.im_cnt += 1
        rec_inp = rec_fn(self.inp)
        dispims_color(readyToPlot(rec_inp), save_path='rec_im_100k'+str(self.im_cnt)+'.png')
        print "Done"


class SaveParams(SimpleExtension):
    def __init__(self, cost, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        self.im_cnt = 0
        self.cost = cost
        self.path = save_path

    def do(self, callback_name, *args):
        print "save =>",
        self.im_cnt += 1
        cg = ComputationGraph(self.cost)
        # TODO: just Var...
        to_save = [p.get_value() for p in VariableFilter(roles=[PARAMETER])(cg.variables)]
        np.save(str(self.im_cnt)+self.path, to_save)
        print "Done"


def main():
    x = tensor.matrix('features')

    monitor_x_n_max = x.max()
    monitor_x_n_max.name = 'X.max'
    monitor_x_n_mean = x.mean(axis=0).min()
    monitor_x_n_mean.name = 'X.mean(0).min'
    monitor_x_n_std = (x.std(axis=0)).min()
    monitor_x_n_std.name = 'X.std(0).min'

    # Initialize prior
    prior_mu = shared_floatx(numpy.zeros(nlat), name='prior_mu')
    prior_log_sigma = shared_floatx(numpy.zeros(nlat), name='prior_log_sigma')
    if learn_prior:
        add_role(prior_mu, PARAMETER)
        add_role(prior_log_sigma, PARAMETER)

    # Initialize encoding network
    in_hid0 = Linear(name='in_hid0',
                     input_dim=nvis,
                     output_dim=nhid,
                     weights_init=IsotropicGaussian(std=0.001),
                     use_bias=False)
    in_hid0.initialize()
    preact0 = in_hid0.apply(x)
    # BN Layer
    preact0_n, [preact0_n_G, preact0_n_B] = bn_layer(preact0,
                                                     (nvis, nhid),
                                                     Gamma_scale=0.1)
    preact0_n_G.name = 'preact0_n_G'
    preact0_n_B.name = 'preact0_n_B'
    add_role(preact0_n_G, PARAMETER)
    add_role(preact0_n_B, PARAMETER)
    hPhi = activation().apply(preact0_n)

    hid0_muPhi = Linear(name='hid0_muPhi',
                        input_dim=nhid,
                        output_dim=nlat,
                        weights_init=IsotropicGaussian(std=0.001),
                        use_bias=False)
    hid0_muPhi.initialize()
    mu_phi = hid0_muPhi.apply(hPhi)
    mu_phi_n, [mu_phi_n_G, mu_phi_n_B] = bn_layer(mu_phi,
                                                  (nlat, nlat),
                                                  Gamma_scale=0.05)
    mu_phi_n_G.name = 'mu_phi_n_G'
    mu_phi_n_B.name = 'mu_phi_n_B'
    add_role(mu_phi_n_G, PARAMETER)
    add_role(mu_phi_n_B, PARAMETER)

    hid0_logSigmaPhi = Linear(name='hid0_logSigmaPhi',
                              input_dim=nhid,
                              output_dim=nlat,
                              weights_init=IsotropicGaussian(std=0.001),
                              use_bias=False)
    hid0_logSigmaPhi.initialize()
    log_sigma_phi = hid0_logSigmaPhi.apply(hPhi)
    log_sigma_phi_n, [log_sigma_phi_n_G, log_sigma_phi_n_B] = bn_layer(log_sigma_phi,
                                                                       (nlat, nlat),
                                                                       Gamma_scale=0.05)
    log_sigma_phi_n_G.name = 'log_sigma_phi_n_G'
    log_sigma_phi_n_B.name = 'log_sigma_phi_n_B'
    add_role(log_sigma_phi_n_G, PARAMETER)
    add_role(log_sigma_phi_n_B, PARAMETER)

    # Encode / decode
    epsilon = theano_rng.normal(size=mu_phi_n.shape, dtype=mu_phi_n.dtype)
    epsilon.name = 'epsilon'
    if softplussmall:
        z = mu_phi_n + epsilon * (tensor.nnet.softplus(log_sigma_phi_n)+0.01)
    else:  # exp
        z = mu_phi_n + epsilon * (tensor.exp(log_sigma_phi_n))  # +0.001)
    z.name = 'z'

    z_hid0 = Linear(name='z_hid0',
                    input_dim=nlat,
                    output_dim=nhid,
                    weights_init=IsotropicGaussian(std=0.001),
                    use_bias=False)
    z_hid0.initialize()
    preact_dec0 = z_hid0.apply(z)
    # BN Layer
    preact_dec0_n, [preact_dec0_n_G, preact_dec0_n_B] = bn_layer(preact_dec0,
                                                                 (nlat, nhid),
                                                                 Gamma_scale=0.1)
    preact_dec0_n_G.name = 'preact_dec0_n_G'
    preact_dec0_n_B.name = 'preact_dec0_n_B'
    add_role(preact_dec0_n_G, PARAMETER)
    add_role(preact_dec0_n_B, PARAMETER)
    hTheta = activation().apply(preact_dec0_n)
    hid0_muTheta = Linear(name='hid0_muTheta',
                          input_dim=nhid,
                          output_dim=nvis,
                          weights_init=IsotropicGaussian(std=0.001),
                          use_bias=False)
    hid0_muTheta.initialize()
    mu_theta = hid0_muTheta.apply(hTheta)
    mu_theta_n, [mu_theta_n_G, mu_theta_n_B] = bn_layer(mu_theta,
                                                        (nvis, nvis),
                                                        Gamma_scale=0.1)
    mu_theta_n_G.name = 'mu_theta_n_G'
    mu_theta_n_B.name = 'mu_theta_n_B'
    add_role(mu_phi_n_G, PARAMETER)
    add_role(mu_phi_n_B, PARAMETER)

    # TODO: fix this, w.r.t. new bn_layers
    def generator(r):
        r2 = r
        hr = z_hid0.apply(r2)
        hr_n = preact_dec0_n_G * hr + preact_dec0_n_B
        m_th = hid0_muTheta.apply(activation().apply(hr_n))
        return mu_theta_n_G * ((m_th - m_th.mean(axis=0, keepdims=True)) /
                               (m_th.std(axis=0, keepdims=True) + 1E-6)) + mu_theta_n_B

    # TODO: fix this, w.r.t new bn_layers
    def reconstructor(inp):
        pre0_inp_n = in_hid0.apply(inp)
        pre0_n = preact0_n_G * ((pre0_inp_n - pre0_inp_n.mean(axis=0, keepdims=True)) / (pre0_inp_n.std(axis=0, keepdims=True) + 1E-6)) + preact0_n_B
        after_rec0 = activation().apply(pre0_n)
        mu_enc = hid0_muPhi.apply(after_rec0)
        mu_enc_n = mu_phi_n_G * ((mu_enc - mu_enc.mean(axis=0, keepdims=True)) /
                                 (mu_enc.std(axis=0, keepdims=True) + 1E-6)) + mu_phi_n_B
        return generator(mu_enc_n)

    # Compute cost
    # TODO: check whether it is correct or not.
    if softplussmall:
        sigma_phi = tensor.nnet.softplus(log_sigma_phi_n)+0.01
    else:  # exp
        sigma_phi = tensor.exp(log_sigma_phi_n)  # +0.001
    t0 = tensor.log(sigma_phi**2).sum(axis=1).mean()
    t0.name = 'log(sig^2).sum.mean----->t0'
    t1 = (mu_phi_n**2).sum(axis=1).mean()
    t1.name = 'mu^2.sum.mean----->t1'
    t2 = (sigma_phi**2).mean()
    t2.name = 'sig^2.mean----->t2'
    t3 = (sigma_phi**2).max()
    t3.name = 'sig^2.max------>t3'
    log_logsigma = log_sigma_phi_n.max()
    log_logsigma.name = 'log_logsigma_n.max'
    kl_term = (
        prior_log_sigma - sigma_phi
        + 0.5 * (
            tensor.exp(2 * sigma_phi) + (mu_phi_n - prior_mu) ** 2
        ) / tensor.exp(2 * prior_log_sigma)
        - 0.5
    ).sum(axis=1)
    kl_term.name = 'kl_term'
    kl_term_mean = kl_term.mean()
    kl_term_mean.name = 'avg_kl_term=(0.5*(1+log(sig^2)-mu^2-sig^2)).sum.mean'
    reconstruction_term = (((x - mu_theta_n) ** 2).sum(axis=1))

    reconstruction_term.name = 'reconstruction_term'
    reconstruction_term_mean = reconstruction_term.mean()
    reconstruction_term_mean.name = 'avg_reconstruction_term'
    cost = (reconstruction_term + kl_term).mean()
    cost.name = '[***]cost=nll_upper_bound'

    # Datasets and data streams
    mil_valid = _100k('valid', sources=('features',))
    valid_monitor_stream = DataStream(
        dataset=mil_valid,
        iteration_scheme=SequentialScheme(mil_valid.num_examples, 20000))

    mil_train = _100k(which_set='train', sources=('features',))
    train_loop_stream = DataStream(
        dataset=mil_train,
        iteration_scheme=SequentialScheme(mil_train.num_examples, 50000))

    train_monitor_stream = DataStream(
        dataset=mil_train,
        iteration_scheme=SequentialScheme(mil_train.num_examples, 20000))

    first_sample = mil_valid.get_data(state=None, request=range(100))[0]
    print "First:"
    print "mean: ", first_sample.mean(), "std: ", first_sample.std()
    print "max: ", first_sample.max(), "min: ", first_sample.min()
    print "abs mean: ", np.abs(first_sample).mean(), "abs min: ", np.abs(first_sample).min(), "abs max: ", np.abs(first_sample).max()

    to_plot = first_sample
    dispims_color(readyToPlot(to_plot), save_path='MIL_ground_truth.png')

    # Get parameters
    computation_graph = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(computation_graph.variables)

    # Training loop
    step_rule = RMSProp(learning_rate=learning_rate, decay_rate=RMS_decay)
    algorithm = GradientDescent(cost=cost, params=params, step_rule=step_rule)
    monitored_quantities = [cost,reconstruction_term_mean,
                kl_term_mean,t0,t1,t2,t3,log_logsigma]  #,monitor_x_n_max,monitor_x_n_mean,monitor_x_n_std]
    main_loop = MainLoop(
        model=None, data_stream=train_loop_stream, algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=20000),
            DataStreamMonitoring(
                monitored_quantities, train_monitor_stream, prefix="++train"),
            DataStreamMonitoring(
                monitored_quantities, valid_monitor_stream, prefix="====valid"),
            Printing(),
            Generate(decoder=generator, prior_mu=prior_mu, prior_log_sigma=prior_log_sigma, BN=True, every_n_epochs=every_n_epoch),
            Reconstruct(reconstructor=reconstructor,
                        inp=first_sample.astype(theano.config.floatX),
                        BN=True, every_n_epochs=every_n_epoch),
            SaveParams(cost=cost, save_path='_params.npy',
                       every_n_epochs=save_every_n_epoch)
            ])
    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
